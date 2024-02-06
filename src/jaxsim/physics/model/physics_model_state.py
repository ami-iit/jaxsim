import jax.numpy as jnp
import jax_dataclasses

import jaxsim.physics.model.physics_model
import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class PhysicsModelState(JaxsimDataclass):
    """
    A class representing the state of a physics model.

    This class stores the joint positions, joint velocities, and the base state (position, orientation, linear velocity,
    and angular velocity) of a physics model.

    Attributes:
        joint_positions (jtp.Vector): An array representing the joint positions.
        joint_velocities (jtp.Vector): An array representing the joint velocities.
        base_position (jtp.Vector): An array representing the base position (default: zeros).
        base_quaternion (jtp.Vector): An array representing the base quaternion (default: [1.0, 0, 0, 0]).
        base_linear_velocity (jtp.Vector): An array representing the base linear velocity (default: zeros).
        base_angular_velocity (jtp.Vector): An array representing the base angular velocity (default: zeros).
    """

    # Joint state
    joint_positions: jtp.Vector
    joint_velocities: jtp.Vector

    # Base state
    base_position: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )
    base_quaternion: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.array([1.0, 0, 0, 0])
    )
    base_linear_velocity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )
    base_angular_velocity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )

    @staticmethod
    def zero(
        physics_model: "jaxsim.physics.model.physics_model.PhysicsModel",
    ) -> "PhysicsModelState":
        return PhysicsModelState(
            joint_positions=jnp.zeros(physics_model.dofs()),
            joint_velocities=jnp.zeros(physics_model.dofs()),
        )

    def position(self) -> jtp.Vector:
        return jnp.hstack(
            [self.base_position, self.base_quaternion, self.joint_positions]
        )

    def velocity(self) -> jtp.Vector:
        # W_v_WB: inertial-fixed representation of the base velocity
        return jnp.hstack(
            [
                self.base_linear_velocity,
                self.base_angular_velocity,
                self.joint_velocities,
            ]
        )

    def xfb(self) -> jtp.Vector:
        return jnp.hstack(
            [
                self.base_quaternion,
                self.base_position,
                self.base_angular_velocity,
                self.base_linear_velocity,
            ]
        )

    def valid(
        self, physics_model: "jaxsim.physics.model.physics_model.PhysicsModel"
    ) -> bool:
        from jaxsim.simulation.utils import check_valid_shape

        valid = True

        valid = check_valid_shape(
            what="joint_positions",
            shape=self.joint_positions.shape,
            expected_shape=(physics_model.dofs(),),
            valid=valid,
        )

        valid = check_valid_shape(
            what="joint_velocities",
            shape=self.joint_velocities.shape,
            expected_shape=(physics_model.dofs(),),
            valid=valid,
        )

        valid = check_valid_shape(
            what="base_position",
            shape=self.base_position.shape,
            expected_shape=(3,),
            valid=valid,
        )

        valid = check_valid_shape(
            what="base_quaternion",
            shape=self.base_quaternion.shape,
            expected_shape=(4,),
            valid=valid,
        )

        valid = check_valid_shape(
            what="base_linear_velocity",
            shape=self.base_linear_velocity.shape,
            expected_shape=(3,),
            valid=valid,
        )

        valid = check_valid_shape(
            what="base_angular_velocity",
            shape=self.base_angular_velocity.shape,
            expected_shape=(3,),
            valid=valid,
        )

        return valid


@jax_dataclasses.pytree_dataclass
class PhysicsModelInput(JaxsimDataclass):
    """
    A class representing the input to a physics model.

    This class stores the joint torques and external forces acting on the bodies of a physics model.

    Attributes:
        tau: An array representing the joint torques.
        f_ext: A matrix representing the external forces acting on the bodies of the physics model.
    """

    tau: jtp.VectorJax
    f_ext: jtp.MatrixJax

    @staticmethod
    def zero(
        physics_model: "jaxsim.physics.model.physics_model.PhysicsModel",
    ) -> "PhysicsModelInput":
        ode_input = PhysicsModelInput(
            tau=jnp.zeros(physics_model.dofs()),
            f_ext=jnp.zeros(shape=(physics_model.NB, 6)),
        )

        assert ode_input.valid(physics_model)
        return ode_input

    def replace(self, validate: bool = True, **kwargs) -> "PhysicsModelInput":
        with jax_dataclasses.copy_and_mutate(self, validate=validate) as updated_input:
            _ = [updated_input.__setattr__(k, v) for k, v in kwargs.items()]

        return updated_input

    def valid(
        self, physics_model: "jaxsim.physics.model.physics_model.PhysicsModel"
    ) -> bool:
        from jaxsim.simulation.utils import check_valid_shape

        valid = True

        valid = check_valid_shape(
            what="tau",
            shape=self.tau.shape,
            expected_shape=(physics_model.dofs(),),
            valid=valid,
        )

        valid = check_valid_shape(
            what="f_ext",
            shape=self.f_ext.shape,
            expected_shape=(physics_model.NB, 6),
            valid=valid,
        )

        return valid
