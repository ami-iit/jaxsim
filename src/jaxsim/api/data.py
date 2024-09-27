from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform
import jax_dataclasses

import jaxsim.api as js
import jaxsim.math
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim.rbda.contacts import SoftContacts
from jaxsim.utils import Mutability
from jaxsim.utils.tracing import not_tracing

from . import common
from .common import VelRepr
from .ode_data import ODEState

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxSimModelData(common.ModelDataWithVelocityRepresentation):
    """
    Class containing the data of a `JaxSimModel` object.
    """

    state: ODEState

    gravity: jtp.Array

    contacts_params: jaxsim.rbda.contacts.ContactsParams = dataclasses.field(repr=False)

    time_ns: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array(
            0, dtype=jnp.uint64 if jax.config.read("jax_enable_x64") else jnp.uint32
        ),
    )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                hash(self.state),
                HashedNumpyArray.hash_of_array(self.gravity),
                HashedNumpyArray.hash_of_array(self.time_ns),
                hash(self.contacts_params),
            )
        )

    def __eq__(self, other: JaxSimModelData) -> bool:

        if not isinstance(other, JaxSimModelData):
            return False

        return hash(self) == hash(other)

    def valid(self, model: js.model.JaxSimModel | None = None) -> bool:
        """
        Check if the current state is valid for the given model.

        Args:
            model: The model to check against.

        Returns:
            `True` if the current state is valid for the given model, `False` otherwise.
        """

        valid = True
        valid = valid and self.standard_gravity() > 0

        if model is not None:
            valid = valid and self.state.valid(model=model)

        return valid

    @staticmethod
    def zero(
        model: js.model.JaxSimModel,
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
        model: js.model.JaxSimModel,
        base_position: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        joint_positions: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        standard_gravity: jtp.FloatLike = jaxsim.math.StandardGravity,
        contact: jaxsim.rbda.contacts.ContactsState | None = None,
        contacts_params: jaxsim.rbda.contacts.ContactsParams | None = None,
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
            standard_gravity: The standard gravity constant.
            contact: The state of the soft contacts.
            contacts_params: The parameters of the soft contacts.
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

        gravity = jnp.zeros(3).at[2].set(-standard_gravity)

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
            jnp.array(
                time * 1e9,
                dtype=jnp.uint64 if jax.config.read("jax_enable_x64") else jnp.uint32,
            )
            if time is not None
            else jnp.array(
                0, dtype=jnp.uint64 if jax.config.read("jax_enable_x64") else jnp.uint32
            )
        )

        W_H_B = jaxsim.math.Transform.from_quaternion_and_translation(
            translation=base_position, quaternion=base_quaternion
        )

        v_WB = JaxSimModelData.other_representation_to_inertial(
            array=jnp.hstack([base_linear_velocity, base_angular_velocity]),
            other_representation=velocity_representation,
            transform=W_H_B,
            is_force=False,
        )

        ode_state = ODEState.build_from_jaxsim_model(
            model=model,
            base_position=base_position.astype(float),
            base_quaternion=base_quaternion.astype(float),
            joint_positions=joint_positions.astype(float),
            base_linear_velocity=v_WB[0:3].astype(float),
            base_angular_velocity=v_WB[3:6].astype(float),
            joint_velocities=joint_velocities.astype(float),
            tangential_deformation=(
                contact.tangential_deformation
                if contact is not None and isinstance(model.contact_model, SoftContacts)
                else None
            ),
        )

        if not ode_state.valid(model=model):
            raise ValueError(ode_state)

        if contacts_params is None:

            if isinstance(model.contact_model, jaxsim.rbda.contacts.SoftContacts):
                contacts_params = js.contact.estimate_good_soft_contacts_parameters(
                    model=model, standard_gravity=standard_gravity
                )
            else:
                contacts_params = model.contact_model.parameters

        return JaxSimModelData(
            time_ns=time_ns,
            state=ode_state,
            gravity=gravity.astype(float),
            contacts_params=contacts_params,
            velocity_representation=velocity_representation,
        )

    # ==================
    # Extract quantities
    # ==================

    def time(self) -> jtp.Float:
        """
        Get the simulated time.

        Returns:
            The simulated time in seconds.
        """

        return self.time_ns.astype(float) / 1e9

    def standard_gravity(self) -> jtp.Float:
        """
        Get the standard gravity constant.

        Returns:
            The standard gravity constant.
        """

        return -self.gravity[2]

    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def joint_positions(
        self,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> jtp.Vector:
        """
        Get the joint positions.

        Args:
            model: The model to consider.
            joint_names:
                The names of the joints for which to get the positions. If `None`,
                the positions of all joints are returned.

        Returns:
            If no model and no joint names are provided, the joint positions as a
            `(DoFs,)` vector corresponding to the serialization of the original
            model used to build the data object.
            If a model is provided and no joint names are provided, the joint positions
            as a `(DoFs,)` vector corresponding to the serialization of the
            provided model.
            If a model and joint names are provided, the joint positions as a
            `(len(joint_names),)` vector corresponding to the serialization of
            the passed joint names vector.
        """

        if model is None:
            if joint_names is not None:
                raise ValueError("Joint names cannot be provided without a model")

            return self.state.physics_model.joint_positions

        if not_tracing(self.state.physics_model.joint_positions) and not self.valid(
            model=model
        ):
            msg = "The data object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()

        return self.state.physics_model.joint_positions[
            js.joint.names_to_idxs(joint_names=joint_names, model=model)
        ]

    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def joint_velocities(
        self,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> jtp.Vector:
        """
        Get the joint velocities.

        Args:
            model: The model to consider.
            joint_names:
                The names of the joints for which to get the velocities. If `None`,
                the velocities of all joints are returned.

        Returns:
            If no model and no joint names are provided, the joint velocities as a
            `(DoFs,)` vector corresponding to the serialization of the original
            model used to build the data object.
            If a model is provided and no joint names are provided, the joint velocities
            as a `(DoFs,)` vector corresponding to the serialization of the
            provided model.
            If a model and joint names are provided, the joint velocities as a
            `(len(joint_names),)` vector corresponding to the serialization of
            the passed joint names vector.
        """

        if model is None:
            if joint_names is not None:
                raise ValueError("Joint names cannot be provided without a model")

            return self.state.physics_model.joint_velocities

        if not_tracing(self.state.physics_model.joint_velocities) and not self.valid(
            model=model
        ):
            msg = "The data object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()

        return self.state.physics_model.joint_velocities[
            js.joint.names_to_idxs(joint_names=joint_names, model=model)
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

        # Extract the base quaternion.
        W_Q_B = self.state.physics_model.base_quaternion.squeeze()

        # Always normalize the quaternion to avoid numerical issues.
        # If the active scheme does not integrate the quaternion on its manifold,
        # we introduce a Baumgarte stabilization to let the quaternion converge to
        # a unit quaternion. In this case, it is not guaranteed that the quaternion
        # stored in the state is a unit quaternion.
        W_Q_B = jax.lax.select(
            pred=jnp.allclose(jnp.linalg.norm(W_Q_B), 1.0, atol=1e-6, rtol=0.0),
            on_true=W_Q_B,
            on_false=W_Q_B / jnp.linalg.norm(W_Q_B),
        )

        return (W_Q_B if not dcm else jaxsim.math.Quaternion.to_dcm(W_Q_B)).astype(
            float
        )

    @jax.jit
    def base_transform(self) -> jtp.Matrix:
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
                transform=W_H_B,
                is_force=False,
            )
            .squeeze()
            .astype(float)
        )

    @jax.jit
    def generalized_position(self) -> tuple[jtp.Matrix, jtp.Vector]:
        r"""
        Get the generalized position
        :math:`\mathbf{q} = ({}^W \mathbf{H}_B, \mathbf{s}) \in \text{SO}(3) \times \mathbb{R}^n`.

        Returns:
            A tuple containing the base transform and the joint positions.
        """

        return self.base_transform(), self.joint_positions()

    @jax.jit
    def generalized_velocity(self) -> jtp.Vector:
        r"""
        Get the generalized velocity
        :math:`\boldsymbol{\nu} = (\boldsymbol{v}_{W,B};\, \boldsymbol{\omega}_{W,B};\, \mathbf{s}) \in \mathbb{R}^{6+n}`

        Returns:
            The generalized velocity in the active representation.
        """

        return (
            jnp.hstack([self.base_velocity(), self.joint_velocities()])
            .squeeze()
            .astype(float)
        )

    # ================
    # Store quantities
    # ================

    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def reset_joint_positions(
        self,
        positions: jtp.VectorLike,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> Self:
        """
        Reset the joint positions.

        Args:
            positions: The joint positions.
            model: The model to consider.
            joint_names: The names of the joints for which to set the positions.

        Returns:
            The updated `JaxSimModelData` object.
        """

        positions = jnp.array(positions)

        def replace(s: jtp.VectorLike) -> JaxSimModelData:
            return self.replace(
                validate=True,
                state=self.state.replace(
                    physics_model=self.state.physics_model.replace(
                        joint_positions=jnp.atleast_1d(s.squeeze()).astype(float)
                    )
                ),
            )

        if model is None:
            return replace(s=positions)

        if not_tracing(positions) and not self.valid(model=model):
            msg = "The data object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()

        return replace(
            s=self.state.physics_model.joint_positions.at[
                js.joint.names_to_idxs(joint_names=joint_names, model=model)
            ].set(positions)
        )

    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def reset_joint_velocities(
        self,
        velocities: jtp.VectorLike,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> Self:
        """
        Reset the joint velocities.

        Args:
            velocities: The joint velocities.
            model: The model to consider.
            joint_names: The names of the joints for which to set the velocities.

        Returns:
            The updated `JaxSimModelData` object.
        """

        velocities = jnp.array(velocities)

        def replace(ṡ: jtp.VectorLike) -> JaxSimModelData:
            return self.replace(
                validate=True,
                state=self.state.replace(
                    physics_model=self.state.physics_model.replace(
                        joint_velocities=jnp.atleast_1d(ṡ.squeeze()).astype(float)
                    )
                ),
            )

        if model is None:
            return replace(ṡ=velocities)

        if not_tracing(velocities) and not self.valid(model=model):
            msg = "The data object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()

        return replace(
            ṡ=self.state.physics_model.joint_velocities.at[
                js.joint.names_to_idxs(joint_names=joint_names, model=model)
            ].set(velocities)
        )

    @jax.jit
    def reset_base_position(self, base_position: jtp.VectorLike) -> Self:
        """
        Reset the base position.

        Args:
            base_position: The base position.

        Returns:
            The updated `JaxSimModelData` object.
        """

        base_position = jnp.array(base_position)

        return self.replace(
            validate=True,
            state=self.state.replace(
                physics_model=self.state.physics_model.replace(
                    base_position=jnp.atleast_1d(base_position.squeeze()).astype(float)
                )
            ),
        )

    @jax.jit
    def reset_base_quaternion(self, base_quaternion: jtp.VectorLike) -> Self:
        """
        Reset the base quaternion.

        Args:
            base_quaternion: The base orientation as a quaternion.

        Returns:
            The updated `JaxSimModelData` object.
        """

        W_Q_B = jnp.array(base_quaternion, dtype=float)

        W_Q_B = jax.lax.select(
            pred=jnp.allclose(jnp.linalg.norm(W_Q_B), 1.0, atol=1e-6, rtol=0.0),
            on_true=W_Q_B,
            on_false=W_Q_B / jnp.linalg.norm(W_Q_B),
        )

        return self.replace(
            validate=True,
            state=self.state.replace(
                physics_model=self.state.physics_model.replace(base_quaternion=W_Q_B)
            ),
        )

    @jax.jit
    def reset_base_pose(self, base_pose: jtp.MatrixLike) -> Self:
        """
        Reset the base pose.

        Args:
            base_pose: The base pose as an SE(3) matrix.

        Returns:
            The updated `JaxSimModelData` object.
        """

        base_pose = jnp.array(base_pose)

        W_p_B = base_pose[0:3, 3]

        W_Q_B = jaxsim.math.Quaternion.from_dcm(dcm=base_pose[0:3, 0:3])

        return self.reset_base_position(base_position=W_p_B).reset_base_quaternion(
            base_quaternion=W_Q_B
        )

    @functools.partial(jax.jit, static_argnames=["velocity_representation"])
    def reset_base_linear_velocity(
        self,
        linear_velocity: jtp.VectorLike,
        velocity_representation: VelRepr | None = None,
    ) -> Self:
        """
        Reset the base linear velocity.

        Args:
            linear_velocity: The base linear velocity as a 3D array.
            velocity_representation:
                The velocity representation in which the base velocity is expressed.
                If `None`, the active representation is considered.

        Returns:
            The updated `JaxSimModelData` object.
        """

        linear_velocity = jnp.array(linear_velocity)

        return self.reset_base_velocity(
            base_velocity=jnp.hstack(
                [
                    linear_velocity.squeeze(),
                    self.base_velocity()[3:6],
                ]
            ),
            velocity_representation=velocity_representation,
        )

    @functools.partial(jax.jit, static_argnames=["velocity_representation"])
    def reset_base_angular_velocity(
        self,
        angular_velocity: jtp.VectorLike,
        velocity_representation: VelRepr | None = None,
    ) -> Self:
        """
        Reset the base angular velocity.

        Args:
            angular_velocity: The base angular velocity as a 3D array.
            velocity_representation:
                The velocity representation in which the base velocity is expressed.
                If `None`, the active representation is considered.

        Returns:
            The updated `JaxSimModelData` object.
        """

        angular_velocity = jnp.array(angular_velocity)

        return self.reset_base_velocity(
            base_velocity=jnp.hstack(
                [
                    self.base_velocity()[0:3],
                    angular_velocity.squeeze(),
                ]
            ),
            velocity_representation=velocity_representation,
        )

    @functools.partial(jax.jit, static_argnames=["velocity_representation"])
    def reset_base_velocity(
        self,
        base_velocity: jtp.VectorLike,
        velocity_representation: VelRepr | None = None,
    ) -> Self:
        """
        Reset the base 6D velocity.

        Args:
            base_velocity: The base 6D velocity in the active representation.
            velocity_representation:
                The velocity representation in which the base velocity is expressed.
                If `None`, the active representation is considered.

        Returns:
            The updated `JaxSimModelData` object.
        """

        base_velocity = jnp.array(base_velocity)

        velocity_representation = (
            velocity_representation
            if velocity_representation is not None
            else self.velocity_representation
        )

        W_v_WB = self.other_representation_to_inertial(
            array=jnp.atleast_1d(base_velocity.squeeze()).astype(float),
            other_representation=velocity_representation,
            transform=self.base_transform(),
            is_force=False,
        )

        return self.replace(
            validate=True,
            state=self.state.replace(
                physics_model=self.state.physics_model.replace(
                    base_linear_velocity=W_v_WB[0:3].squeeze().astype(float),
                    base_angular_velocity=W_v_WB[3:6].squeeze().astype(float),
                )
            ),
        )


@functools.partial(jax.jit, static_argnames=["velocity_representation", "base_rpy_seq"])
def random_model_data(
    model: js.model.JaxSimModel,
    *,
    key: jax.Array | None = None,
    velocity_representation: VelRepr | None = None,
    base_pos_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = ((-1, -1, 0.5), 1.0),
    base_rpy_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-jnp.pi, jnp.pi),
    base_rpy_seq: str = "XYZ",
    joint_pos_bounds: (
        tuple[
            jtp.FloatLike | Sequence[jtp.FloatLike],
            jtp.FloatLike | Sequence[jtp.FloatLike],
        ]
        | None
    ) = None,
    base_vel_lin_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
    base_vel_ang_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
    joint_vel_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
    standard_gravity_bounds: tuple[jtp.FloatLike, jtp.FloatLike] = (
        jaxsim.math.StandardGravity,
        jaxsim.math.StandardGravity,
    ),
) -> JaxSimModelData:
    """
    Randomly generate a `JaxSimModelData` object.

    Args:
        model: The target model for the random data.
        key: The random key.
        velocity_representation: The velocity representation to use.
        base_pos_bounds: The bounds for the base position.
        base_rpy_bounds:
            The bounds for the euler angles used to build the base orientation.
        base_rpy_seq:
            The sequence of axes for rotation (using `Rotation` from scipy).
        joint_pos_bounds:
            The bounds for the joint positions (reading the joint limits if None).
        base_vel_lin_bounds: The bounds for the base linear velocity.
        base_vel_ang_bounds: The bounds for the base angular velocity.
        joint_vel_bounds: The bounds for the joint velocities.
        standard_gravity_bounds: The bounds for the standard gravity.

    Returns:
        A `JaxSimModelData` object with random data.
    """

    key = key if key is not None else jax.random.PRNGKey(seed=0)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, num=7)

    p_min = jnp.array(base_pos_bounds[0], dtype=float)
    p_max = jnp.array(base_pos_bounds[1], dtype=float)
    rpy_min = jnp.array(base_rpy_bounds[0], dtype=float)
    rpy_max = jnp.array(base_rpy_bounds[1], dtype=float)
    v_min = jnp.array(base_vel_lin_bounds[0], dtype=float)
    v_max = jnp.array(base_vel_lin_bounds[1], dtype=float)
    ω_min = jnp.array(base_vel_ang_bounds[0], dtype=float)
    ω_max = jnp.array(base_vel_ang_bounds[1], dtype=float)
    ṡ_min, ṡ_max = joint_vel_bounds

    random_data = JaxSimModelData.zero(
        model=model,
        **(
            dict(velocity_representation=velocity_representation)
            if velocity_representation is not None
            else {}
        ),
    )

    with random_data.mutable_context(
        mutability=Mutability.MUTABLE, restore_after_exception=False
    ):

        physics_model_state = random_data.state.physics_model

        physics_model_state.base_position = jax.random.uniform(
            key=k1, shape=(3,), minval=p_min, maxval=p_max
        )

        physics_model_state.base_quaternion = jaxsim.math.Quaternion.to_wxyz(
            xyzw=jax.scipy.spatial.transform.Rotation.from_euler(
                seq=base_rpy_seq,
                angles=jax.random.uniform(
                    key=k2, shape=(3,), minval=rpy_min, maxval=rpy_max
                ),
            ).as_quat()
        )

        if model.number_of_joints() > 0:

            s_min, s_max = (
                jnp.array(joint_pos_bounds, dtype=float)
                if joint_pos_bounds is not None
                else (None, None)
            )

            physics_model_state.joint_positions = (
                js.joint.random_joint_positions(model=model, key=k3)
                if (s_min is None or s_max is None)
                else jax.random.uniform(
                    key=k3, shape=(model.dofs(),), minval=s_min, maxval=s_max
                )
            )

            physics_model_state.joint_velocities = jax.random.uniform(
                key=k4, shape=(model.dofs(),), minval=ṡ_min, maxval=ṡ_max
            )

        if model.floating_base():
            physics_model_state.base_linear_velocity = jax.random.uniform(
                key=k5, shape=(3,), minval=v_min, maxval=v_max
            )

            physics_model_state.base_angular_velocity = jax.random.uniform(
                key=k6, shape=(3,), minval=ω_min, maxval=ω_max
            )

        random_data.gravity = (
            jnp.zeros(3, dtype=random_data.gravity.dtype)
            .at[2]
            .set(
                -jax.random.uniform(
                    key=k7,
                    shape=(),
                    minval=standard_gravity_bounds[0],
                    maxval=standard_gravity_bounds[1],
                )
            )
        )

    return random_data
