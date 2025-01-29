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

from . import common
from .common import VelRepr

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxSimModelData(common.ModelDataWithVelocityRepresentation):
    """
    Class storing the state of the physics model dynamics.

    Attributes:
        joint_positions: The vector of joint positions.
        joint_velocities: The vector of joint velocities.
        base_position: The 3D position of the base link.
        base_quaternion: The quaternion defining the orientation of the base link.
        base_linear_velocity:
            The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in inertial-fixed representation.
        base_transform: The base transform.
        joint_transforms: The joint transforms.
        link_transforms: The link transforms.
        link_velocities: The link velocities in inertial-fixed representation.
    """

    # Joint state
    joint_positions: jtp.Vector
    joint_velocities: jtp.Vector

    # Base state
    base_quaternion: jtp.Vector
    base_linear_velocity: jtp.Vector
    base_angular_velocity: jtp.Vector
    base_position: jtp.Vector

    # Cached computations.
    base_transform: jtp.Matrix = dataclasses.field(repr=False, default=None)
    joint_transforms: jtp.Matrix = dataclasses.field(repr=False, default=None)
    link_transforms: jtp.Matrix = dataclasses.field(repr=False, default=None)
    link_velocities: jtp.Matrix = dataclasses.field(repr=False, default=None)

    @staticmethod
    def build(
        model: js.model.JaxSimModel,
        base_position: jtp.VectorLike | None = None,
        base_quaternion: jtp.VectorLike | None = None,
        joint_positions: jtp.VectorLike | None = None,
        base_linear_velocity: jtp.VectorLike | None = None,
        base_angular_velocity: jtp.VectorLike | None = None,
        joint_velocities: jtp.VectorLike | None = None,
        velocity_representation: VelRepr = VelRepr.Mixed,
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
            velocity_representation: The velocity representation to use. It defaults to mixed if not provided.

        Returns:
            A `JaxSimModelData` initialized with the given state.
        """

        base_position = jnp.array(
            base_position if base_position is not None else jnp.zeros(3),
            dtype=float,
        ).squeeze()

        base_quaternion = jnp.array(
            (
                base_quaternion
                if base_quaternion is not None
                else jnp.array([1.0, 0, 0, 0])
            ),
            dtype=float,
        ).squeeze()

        base_linear_velocity = jnp.array(
            base_linear_velocity if base_linear_velocity is not None else jnp.zeros(3),
            dtype=float,
        ).squeeze()

        base_angular_velocity = jnp.array(
            (
                base_angular_velocity
                if base_angular_velocity is not None
                else jnp.zeros(3)
            ),
            dtype=float,
        ).squeeze()

        joint_positions = jnp.atleast_1d(
            jnp.array(
                (
                    joint_positions
                    if joint_positions is not None
                    else jnp.zeros(model.dofs())
                ),
                dtype=float,
            ).squeeze()
        )

        joint_velocities = jnp.atleast_1d(
            jnp.array(
                (
                    joint_velocities
                    if joint_velocities is not None
                    else jnp.zeros(model.dofs())
                ),
                dtype=float,
            ).squeeze()
        )

        W_H_B = jaxsim.math.Transform.from_quaternion_and_translation(
            translation=base_position, quaternion=base_quaternion
        )

        W_v_WB = JaxSimModelData.other_representation_to_inertial(
            array=jnp.hstack([base_linear_velocity, base_angular_velocity]),
            other_representation=velocity_representation,
            transform=W_H_B,
            is_force=False,
        ).astype(float)

        joint_transforms = model.kin_dyn_parameters.joint_transforms(
            joint_positions=joint_positions, base_transform=W_H_B
        )

        link_transforms, link_velocities_inertial = (
            jaxsim.rbda.forward_kinematics_model(
                model=model,
                base_position=base_position,
                base_quaternion=base_quaternion,
                joint_positions=joint_positions,
                base_linear_velocity_inertial=W_v_WB[0:3],
                base_angular_velocity_inertial=W_v_WB[3:6],
                joint_velocities=joint_velocities,
            )
        )

        model_data = JaxSimModelData(
            base_quaternion=base_quaternion,
            base_position=base_position,
            joint_positions=joint_positions,
            base_linear_velocity=W_v_WB[0:3],
            base_angular_velocity=W_v_WB[3:6],
            joint_velocities=joint_velocities,
            velocity_representation=velocity_representation,
            base_transform=W_H_B,
            joint_transforms=joint_transforms,
            link_transforms=link_transforms,
            link_velocities=link_velocities_inertial,
        )

        if not model_data.valid(model=model):
            raise ValueError(
                "The built state is not compatible with the model.", model_data
            )

        return model_data

    @staticmethod
    def zero(
        model: js.model.JaxSimModel,
        velocity_representation: VelRepr = VelRepr.Mixed,
    ) -> JaxSimModelData:
        """
        Create a `JaxSimModelData` object with zero state.

        Args:
            model: The model for which to create the state.
            velocity_representation: The velocity representation to use. It defaults to mixed if not provided.

        Returns:
            A `JaxSimModelData` initialized with zero state.
        """
        return JaxSimModelData.build(
            model=model, velocity_representation=velocity_representation
        )

    # ==================
    # Extract quantities
    # ==================

    @js.common.named_scope
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
        W_Q_B = self.base_quaternion.squeeze()

        # Always normalize the quaternion to avoid numerical issues.
        # If the active scheme does not integrate the quaternion on its manifold,
        # we introduce a Baumgarte stabilization to let the quaternion converge to
        # a unit quaternion. In this case, it is not guaranteed that the quaternion
        # stored in the state is a unit quaternion.
        norm = jaxsim.math.safe_norm(W_Q_B)
        W_Q_B = W_Q_B / (norm + jnp.finfo(float).eps * (norm == 0))

        return (W_Q_B if not dcm else jaxsim.math.Quaternion.to_dcm(W_Q_B)).astype(
            float
        )

    @js.common.named_scope
    @jax.jit
    def base_velocity(self) -> jtp.Vector:
        """
        Get the base 6D velocity.

        Returns:
            The base 6D velocity in the active representation.
        """

        W_v_WB = jnp.hstack(
            [
                self.base_linear_velocity,
                self.base_angular_velocity,
            ]
        )

        W_H_B = self.base_transform

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

    @js.common.named_scope
    @jax.jit
    def generalized_position(self) -> tuple[jtp.Matrix, jtp.Vector]:
        r"""
        Get the generalized position
        :math:`\mathbf{q} = ({}^W \mathbf{H}_B, \mathbf{s}) \in \text{SO}(3) \times \mathbb{R}^n`.

        Returns:
            A tuple containing the base transform and the joint positions.
        """

        return self.base_transform, self.joint_positions

    @js.common.named_scope
    @jax.jit
    def generalized_velocity(self) -> jtp.Vector:
        r"""
        Get the generalized velocity.

        :math:`\boldsymbol{\nu} = (\boldsymbol{v}_{W,B};\, \boldsymbol{\omega}_{W,B};\, \mathbf{s}) \in \mathbb{R}^{6+n}`

        Returns:
            The generalized velocity in the active representation.
        """

        return (
            jnp.hstack([self.base_velocity(), self.joint_velocities])
            .squeeze()
            .astype(float)
        )

    # ================
    # Store quantities
    # ================

    @js.common.named_scope
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

        norm = jaxsim.math.safe_norm(W_Q_B)
        W_Q_B = W_Q_B / (norm + jnp.finfo(float).eps * (norm == 0))

        return self.replace(validate=True, base_quaternion=W_Q_B)

    @js.common.named_scope
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
        return self.replace(
            base_position=W_p_B,
            base_quaternion=W_Q_B,
        )

    @js.common.named_scope
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

    @js.common.named_scope
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

    @js.common.named_scope
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
            transform=self.base_transform,
            is_force=False,
        )

        return self.replace(
            validate=True,
            base_linear_velocity=W_v_WB[0:3].squeeze().astype(float),
            base_angular_velocity=W_v_WB[3:6].squeeze().astype(float),
        )

    def replace(
        self,
        model: js.model.JaxSimModel,
        joint_positions: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        base_position: jtp.Vector | None = None,
        validate: bool = False,
    ) -> Self:
        """
        Replace the attributes of the `JaxSimModelData` object.
        """
        if joint_positions is None:
            joint_positions = self.joint_positions
        if joint_velocities is None:
            joint_velocities = self.joint_velocities
        if base_quaternion is None:
            base_quaternion = self.base_quaternion
        if base_linear_velocity is None:
            base_linear_velocity = self.base_linear_velocity
        if base_angular_velocity is None:
            base_angular_velocity = self.base_angular_velocity
        if base_position is None:
            base_position = self.base_position

        joint_positions = jnp.atleast_1d(joint_positions.squeeze()).astype(float)
        joint_velocities = jnp.atleast_1d(joint_velocities.squeeze()).astype(float)
        base_quaternion = jnp.atleast_1d(base_quaternion.squeeze()).astype(float)
        base_linear_velocity = jnp.atleast_1d(base_linear_velocity.squeeze())
        base_linear_velocity = base_linear_velocity.astype(float)
        base_angular_velocity = jnp.atleast_1d(base_angular_velocity.squeeze())
        base_angular_velocity = base_angular_velocity.astype(float)
        base_position = jnp.atleast_1d(base_position.squeeze())
        base_position = base_position.astype(float)

        base_transform = jaxsim.math.Transform.from_quaternion_and_translation(
            translation=base_position, quaternion=base_quaternion
        )
        joint_transforms = model.kin_dyn_parameters.joint_transforms(
            joint_positions=joint_positions, base_transform=base_transform
        )
        link_transforms, link_velocities = jaxsim.rbda.forward_kinematics_model(
            model=model,
            base_position=base_position,
            base_quaternion=base_quaternion,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=base_angular_velocity,
        )

        return super().replace(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_quaternion=base_quaternion,
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=base_angular_velocity,
            base_position=base_position,
            validate=validate,
            base_transform=base_transform,
            joint_transforms=joint_transforms,
            link_transforms=link_transforms,
            link_velocities=link_velocities,
        )

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `JaxSimModelData` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `JaxSimModelData` against.

        Returns:
            `True` if the `JaxSimModelData` is valid for the given model,
            `False` otherwise.
        """
        if self.joint_positions.shape != (model.dofs(),):
            return False
        if self.joint_velocities.shape != (model.dofs(),):
            return False
        if self.base_position.shape != (3,):
            return False
        if self.base_quaternion.shape != (4,):
            return False
        if self.base_linear_velocity.shape != (3,):
            return False
        if self.base_angular_velocity.shape != (3,):
            return False

        return True


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

    Returns:
        A `JaxSimModelData` object with random data.
    """

    key = key if key is not None else jax.random.PRNGKey(seed=0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, num=6)

    p_min = jnp.array(base_pos_bounds[0], dtype=float)
    p_max = jnp.array(base_pos_bounds[1], dtype=float)
    rpy_min = jnp.array(base_rpy_bounds[0], dtype=float)
    rpy_max = jnp.array(base_rpy_bounds[1], dtype=float)
    v_min = jnp.array(base_vel_lin_bounds[0], dtype=float)
    v_max = jnp.array(base_vel_lin_bounds[1], dtype=float)
    ω_min = jnp.array(base_vel_ang_bounds[0], dtype=float)
    ω_max = jnp.array(base_vel_ang_bounds[1], dtype=float)
    ṡ_min, ṡ_max = joint_vel_bounds

    base_position = jax.random.uniform(key=k1, shape=(3,), minval=p_min, maxval=p_max)

    base_quaternion = jaxsim.math.Quaternion.to_wxyz(
        xyzw=jax.scipy.spatial.transform.Rotation.from_euler(
            seq=base_rpy_seq,
            angles=jax.random.uniform(
                key=k2, shape=(3,), minval=rpy_min, maxval=rpy_max
            ),
        ).as_quat()
    )

    (
        joint_positions,
        joint_velocities,
        base_linear_velocity,
        base_angular_velocity,
    ) = (None,) * 4

    if model.number_of_joints() > 0:

        s_min, s_max = (
            jnp.array(joint_pos_bounds, dtype=float)
            if joint_pos_bounds is not None
            else (None, None)
        )

        joint_positions = (
            js.joint.random_joint_positions(model=model, key=k3)
            if (s_min is None or s_max is None)
            else jax.random.uniform(
                key=k3, shape=(model.dofs(),), minval=s_min, maxval=s_max
            )
        )

        joint_velocities = jax.random.uniform(
            key=k4, shape=(model.dofs(),), minval=ṡ_min, maxval=ṡ_max
        )

    if model.floating_base():
        base_linear_velocity = jax.random.uniform(
            key=k5, shape=(3,), minval=v_min, maxval=v_max
        )

        base_angular_velocity = jax.random.uniform(
            key=k6, shape=(3,), minval=ω_min, maxval=ω_max
        )

    return JaxSimModelData.build(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        **(
            {"velocity_representation": velocity_representation}
            if velocity_representation is not None
            else {}
        ),
    )
