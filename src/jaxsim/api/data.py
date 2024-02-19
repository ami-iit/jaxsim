from __future__ import annotations

import contextlib
import dataclasses
import functools
import weakref
from typing import ContextManager

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
from jax_dataclasses import Static

import jaxsim.api.model
import jaxsim.physics.algos.aba
import jaxsim.physics.algos.crba
import jaxsim.physics.algos.forward_kinematics
import jaxsim.physics.algos.rnea
import jaxsim.physics.model.physics_model
import jaxsim.physics.model.physics_model_state
import jaxsim.typing as jtp
from jaxsim import sixd
from jaxsim.high_level.common import VelRepr
from jaxsim.physics.algos import soft_contacts
from jaxsim.simulation.ode_data import ODEState
from jaxsim.utils import JaxsimDataclass

from . import contact as Contact


@dataclasses.dataclass
class HashlessReferenceType:
    ref: weakref.ReferenceType

    def __hash__(self) -> int:
        return 0


@jax_dataclasses.pytree_dataclass
class JaxSimModelData(JaxsimDataclass):
    """
    Class containing the state of a `JaxSimModel` object.
    """

    state: ODEState

    gravity: jtp.Array

    soft_contacts_params: soft_contacts.SoftContactsParams = dataclasses.field(
        repr=False
    )

    time_ns: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.uint64)
    )

    velocity_representation: Static[VelRepr] = VelRepr.Inertial

    _model_ref: Static[HashlessReferenceType] = dataclasses.field(
        default=None, repr=False
    )

    @property
    def model(self) -> jaxsim.api.model.JaxSimModel:
        """
        The model associated with the current state.

        Returns:
            The model associated with the current state.

        Raises:
            RuntimeError: If the model has been deleted.

        Note:
            The model is stored as a weak reference to prevent garbage collection
            problems due to circular references. It is possible that the associated
            model has been deleted.
        """

        m = self._model_ref.ref()

        if m is None:
            raise RuntimeError("The model has been deleted")

        return m

    def valid(self, model: jaxsim.api.model.JaxSimModel) -> bool:
        """
        Check if the current state is valid for the given model.

        Args:
            model: The model to check against.

        Returns:
            `True` if the current state is valid for the given model, `False` otherwise.
        """

        valid = True
        valid = valid and self.model is not None
        valid = valid and self.state.valid(physics_model=model.physics_model)

        return valid

    @contextlib.contextmanager
    def switch_velocity_representation(
        self, velocity_representation: VelRepr
    ) -> ContextManager[JaxSimModelData]:
        """
        Context manager to temporarily switch the velocity representation.

        Args:
            velocity_representation: The new velocity representation.

        Yields:
            The same `JaxSimModelData` object with the new velocity representation.
        """

        original_representation = self.velocity_representation

        try:

            # First, we replace the velocity representation
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = velocity_representation

            # Then, we yield the data with changed representation.
            # We run this in a mutable context with restoration so that any exception
            # occurring, we restore the original object in case it was modified.
            with self.mutable_context(
                mutability=self._mutability(), restore_after_exception=True
            ):
                yield self

        finally:
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = original_representation

    @staticmethod
    def zero(
        model: jaxsim.api.model.JaxSimModel,
        velocity_representation: VelRepr = VelRepr.Inertial,
    ) -> JaxSimModelData:
        """
        Create a `JaxSimModelData` object with zero state.

        Args:
            model: The model for which to create the zero state.
            velocity_representation: The velocity representation to use.

        Returns:
            A `JaxSimModelData` object with zero state.
        """

        return JaxSimModelData.build(
            model=model, velocity_representation=velocity_representation
        )

    @staticmethod
    def build(
        model: jaxsim.api.model.JaxSimModel,
        base_position: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        joint_positions: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        gravity: jtp.Vector | None = None,
        soft_contacts_state: soft_contacts.SoftContactsState | None = None,
        soft_contacts_params: soft_contacts.SoftContactsParams | None = None,
        velocity_representation: VelRepr = VelRepr.Inertial,
        time: jtp.FloatLike | None = None,
    ) -> JaxSimModelData:
        """
        Create a `JaxSimModelData` object with the given state.

        Args:
            model: The model for which to create the state.
            base_position: The base position.
            base_quaternion: The base orientation as a quaternion.
            joint_positions: The joint positions.
            base_linear_velocity:
                The base linear velocity in the selected representation.
            base_angular_velocity:
                The base angular velocity in the selected representation.
            joint_velocities: The joint velocities.
            gravity: The gravity 3D vector.
            soft_contacts_state: The state of the soft contacts.
            soft_contacts_params: The parameters of the soft contacts.
            velocity_representation: The velocity representation to use.
            time: The time at which the state is created.

        Returns:
            A `JaxSimModelData` object with the given state.
        """

        base_position = jnp.array(
            base_position if base_position is not None else jnp.zeros(3)
        ).squeeze()

        base_quaternion = jnp.array(
            base_quaternion
            if base_quaternion is not None
            else jnp.array([1.0, 0, 0, 0])
        ).squeeze()

        base_linear_velocity = jnp.array(
            base_linear_velocity if base_linear_velocity is not None else jnp.zeros(3)
        ).squeeze()

        base_angular_velocity = jnp.array(
            base_angular_velocity if base_angular_velocity is not None else jnp.zeros(3)
        ).squeeze()

        gravity = jnp.array(
            gravity if gravity is not None else model.physics_model.gravity[0:3]
        ).squeeze()

        joint_positions = jnp.atleast_1d(
            joint_positions.squeeze()
            if joint_positions is not None
            else jnp.zeros(model.dofs())
        )

        joint_velocities = jnp.atleast_1d(
            joint_velocities.squeeze()
            if joint_velocities is not None
            else jnp.zeros(model.dofs())
        )

        time_ns = (
            jnp.array(time * 1e9, dtype=jnp.uint64)
            if time is not None
            else jnp.array(0, dtype=jnp.uint64)
        )

        soft_contacts_params = (
            soft_contacts_params
            if soft_contacts_params is not None
            else Contact.estimate_good_soft_contacts_parameters(model=model)
        )

        W_H_B = jaxlie.SE3.from_rotation_and_translation(
            translation=base_position,
            rotation=jaxlie.SO3.from_quaternion_xyzw(
                base_quaternion[jnp.array([1, 2, 3, 0])]
            ),
        ).as_matrix()

        v_WB = JaxSimModelData.other_representation_to_inertial(
            array=jnp.hstack([base_linear_velocity, base_angular_velocity]),
            other_representation=velocity_representation,
            base_transform=W_H_B,
            is_force=False,
        )

        ode_state = ODEState.build(
            physics_model=model.physics_model,
            physics_model_state=jaxsim.physics.model.physics_model.PhysicsModelState(
                base_position=base_position.astype(float),
                base_quaternion=base_quaternion.astype(float),
                joint_positions=joint_positions.astype(float),
                base_linear_velocity=v_WB[0:3].astype(float),
                base_angular_velocity=v_WB[3:6].astype(float),
                joint_velocities=joint_velocities.astype(float),
            ),
            soft_contacts_state=soft_contacts_state,
        )

        if not ode_state.valid(physics_model=model.physics_model):
            raise ValueError(ode_state)

        return JaxSimModelData(
            time_ns=time_ns,
            state=ode_state,
            gravity=gravity.astype(float),
            soft_contacts_params=soft_contacts_params,
            velocity_representation=velocity_representation,
            _model_ref=HashlessReferenceType(ref=weakref.ref(model)),
        )

    def joint_positions(self, joint_names: tuple[str, ...] | None = None) -> jtp.Vector:
        """
        Get the joint positions.

        Args:
            joint_names:
                The names of the joints for which to get the positions. If `None`, the
                positions of all joints are returned.

        Returns:
            The joint positions.
        """

        import jaxsim.api.joint

        joint_names = (
            joint_names if joint_names is not None else self.model.joint_names()
        )

        return self.state.physics_model.joint_positions[
            jaxsim.api.joint.names_to_idxs(joint_names=joint_names, model=self.model)
        ]

    def joint_velocities(
        self, joint_names: tuple[str, ...] | None = None
    ) -> jtp.Vector:
        """
        Get the joint velocities.

        Args:
            joint_names:
                The names of the joints for which to get the velocities. If `None`, the
                velocities of all joints are returned.

        Returns:
            The joint velocities.
        """

        import jaxsim.api.joint

        joint_names = (
            joint_names if joint_names is not None else self.model.joint_names()
        )

        return self.state.physics_model.joint_velocities[
            jaxsim.api.joint.names_to_idxs(joint_names=joint_names, model=self.model)
        ]

    @jax.jit
    def base_position(self) -> jtp.Vector:
        """
        Get the base position.

        Returns:
            The base position.
        """

        return self.state.physics_model.base_position.squeeze()

    @functools.partial(jax.jit, static_argnames=["dcm"])
    def base_orientation(self, dcm: jtp.BoolLike = False) -> jtp.Vector | jtp.Matrix:
        """
        Get the base orientation.

        Args:
            dcm: Whether to return the orientation as a SO(3) matrix or quaternion.

        Returns:
            The base orientation.
        """

        # Always normalize the quaternion to avoid numerical issues.
        # If the active scheme does not integrate the quaternion on its manifold,
        # we introduce a Baumgarte stabilization to let the quaternion converge to
        # a unit quaternion. In this case, it is not guaranteed that the quaternion
        # stored in the state is a unit quaternion.
        base_unit_quaternion = (
            self.state.physics_model.base_quaternion.squeeze()
            / jnp.linalg.norm(self.state.physics_model.base_quaternion)
        )

        # Slice to convert quaternion wxyz -> xyzw
        to_xyzw = np.array([1, 2, 3, 0])

        return (
            base_unit_quaternion
            if not dcm
            else sixd.so3.SO3.from_quaternion_xyzw(
                base_unit_quaternion[to_xyzw]
            ).as_matrix()
        )

    @jax.jit
    def base_transform(self) -> jtp.MatrixJax:
        """
        Get the base transform.

        Returns:
            The base transform as an SE(3) matrix.
        """

        W_R_B = self.base_orientation(dcm=True)
        W_p_B = jnp.vstack(self.base_position())

        return jnp.vstack(
            [
                jnp.block([W_R_B, W_p_B]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

    @jax.jit
    def base_velocity(self) -> jtp.Vector:
        """
        Get the base 6D velocity.

        Returns:
            The base 6D velocity in the active representation.
        """

        W_v_WB = jnp.hstack(
            [
                self.state.physics_model.base_linear_velocity,
                self.state.physics_model.base_angular_velocity,
            ]
        )

        W_H_B = self.base_transform()

        return (
            JaxSimModelData.inertial_to_other_representation(
                array=W_v_WB,
                other_representation=self.velocity_representation,
                base_transform=W_H_B,
                is_force=False,
            )
            .squeeze()
            .astype(float)
        )

    @jax.jit
    def generalized_position(self) -> tuple[jtp.Matrix, jtp.Vector]:
        """
        Get the generalized position.

        Returns:
            A tuple containing the base transform and the joint positions.
        """

        return self.base_transform(), self.joint_positions()

    @jax.jit
    def generalized_velocity(self) -> jtp.Vector:
        """
        Get the generalized velocity.

        Returns:
            The generalized velocity in the active representation.
        """

        return (
            jnp.hstack([self.base_velocity(), self.joint_velocities()])
            .squeeze()
            .astype(float)
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def inertial_to_other_representation(
        array: jtp.Array,
        other_representation: VelRepr,
        base_transform: jtp.Matrix,
        is_force: bool = False,
    ) -> jtp.Array:
        """
        Convert a 6D quantity from the inertial to another representation.

        Args:
            array: The 6D quantity to convert.
            other_representation: The representation to convert to.
            base_transform: The base transform.
            is_force: Whether the quantity is a 6D force or 6D velocity.

        Returns:
            The 6D quantity in the other representation.
        """

        W_array = array.squeeze()
        W_H_B = base_transform.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size, 6)

        if W_H_B.shape != (4, 4):
            raise ValueError(W_H_B.shape, (4, 4))

        match other_representation:

            case VelRepr.Inertial:
                return W_array

            case VelRepr.Body:

                if not is_force:
                    B_Xv_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
                    B_array = B_Xv_W @ W_array

                else:
                    B_Xf_W = sixd.se3.SE3.from_matrix(W_H_B).adjoint().T
                    B_array = B_Xf_W @ W_array

                return B_array

            case VelRepr.Mixed:
                W_p_B = W_H_B[0:3, 3]
                W_H_BW = jnp.eye(4).at[0:3, 3].set(W_p_B)

                if not is_force:
                    BW_Xv_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
                    BW_array = BW_Xv_W @ W_array

                else:
                    BW_Xf_W = sixd.se3.SE3.from_matrix(W_H_BW).adjoint().T
                    BW_array = BW_Xf_W @ W_array

                return BW_array

            case _:
                raise ValueError(other_representation)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def other_representation_to_inertial(
        array: jtp.Array,
        other_representation: VelRepr,
        base_transform: jtp.Matrix,
        is_force: bool = False,
    ) -> jtp.Array:
        """
        Convert a 6D quantity from another representation to the inertial.

        Args:
            array: The 6D quantity to convert.
            other_representation: The representation to convert from.
            base_transform: The base transform.
            is_force: Whether the quantity is a 6D force or 6D velocity.

        Returns:
            The 6D quantity in the inertial representation.
        """

        W_array = array.squeeze()
        W_H_B = base_transform.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size, 6)

        if W_H_B.shape != (4, 4):
            raise ValueError(W_H_B.shape, (4, 4))

        match other_representation:
            case VelRepr.Inertial:
                W_array = array
                return W_array

            case VelRepr.Body:
                B_array = array

                if not is_force:
                    W_Xv_B: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).adjoint()
                    W_array = W_Xv_B @ B_array

                else:
                    W_Xf_B = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint().T
                    W_array = W_Xf_B @ B_array

                return W_array

            case VelRepr.Mixed:
                BW_array = array
                W_p_B = W_H_B[0:3, 3]
                W_H_BW = jnp.eye(4).at[0:3, 3].set(W_p_B)

                if not is_force:
                    W_Xv_BW: jtp.Array = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
                    W_array = W_Xv_BW @ BW_array

                else:
                    W_Xf_BW = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint().T
                    W_array = W_Xf_BW @ BW_array

                return W_array

            case _:
                raise ValueError(other_representation)
