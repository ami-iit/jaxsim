import dataclasses

import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.data import JaxSimModelData
from jaxsim.math import Adjoint, Transform


def semi_implicit_euler_integration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    base_acceleration_inertial: jtp.Vector,
    joint_accelerations: jtp.Vector,
) -> JaxSimModelData:
    """Integrate the system state using the semi-implicit Euler method."""
    # Step the dynamics forward.
    with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

        dt = model.time_step
        W_v̇_WB = base_acceleration_inertial
        s̈ = joint_accelerations

        B_H_W = Transform.inverse(data._base_transform).at[:3, :3].set(jnp.eye(3))
        BW_X_W = Adjoint.from_transform(B_H_W)

        new_generalized_acceleration = jnp.hstack([W_v̇_WB, s̈])

        new_generalized_velocity = (
            data.generalized_velocity + dt * new_generalized_acceleration
        )

        new_base_velocity_inertial = new_generalized_velocity[0:6]
        new_joint_velocities = new_generalized_velocity[6:]

        base_lin_velocity_inertial = new_base_velocity_inertial[0:3]

        new_base_velocity_mixed = BW_X_W @ new_generalized_velocity[0:6]
        base_lin_velocity_mixed = new_base_velocity_mixed[0:3]
        base_ang_velocity_mixed = new_base_velocity_mixed[3:6]

        base_quaternion_derivative = jaxsim.math.Quaternion.derivative(
            quaternion=data.base_orientation,
            omega=base_ang_velocity_mixed,
            omega_in_body_fixed=False,
        ).squeeze()

        new_base_position = data.base_position + dt * base_lin_velocity_mixed
        new_base_quaternion = data.base_orientation + dt * base_quaternion_derivative

        base_quaternion_norm = jaxsim.math.safe_norm(new_base_quaternion)

        new_base_quaternion = new_base_quaternion / jnp.where(
            base_quaternion_norm == 0, 1.0, base_quaternion_norm
        )

        new_joint_position = data.joint_positions + dt * new_joint_velocities

    # TODO: Avoid double replace, e.g. by computing cached value here
    data = dataclasses.replace(
        data,
        _base_quaternion=new_base_quaternion,
        _base_position=new_base_position,
        _joint_positions=new_joint_position,
        _joint_velocities=new_joint_velocities,
        _base_linear_velocity=base_lin_velocity_inertial,
        # Here we use the base angular velocity in mixed representation since
        # it's equivalent to the one in inertial representation
        # See: S. Traversaro and A. Saccon, “Multibody Dynamics Notation (Version 2), pg.9
        _base_angular_velocity=base_ang_velocity_mixed,
    )
    data = data.replace(model=model)  # update cache

    return data
