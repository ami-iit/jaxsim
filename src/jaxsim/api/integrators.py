import dataclasses
from collections.abc import Callable

import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
from jaxsim.api.integrator_types import IntegratorType
import jaxsim.typing as jtp
from jaxsim.api.data import JaxSimModelData
from jaxsim.math import Adjoint, Transform


def get_integrator(integrator_type: IntegratorType) -> Callable:
    """Get the integrator function based on the integrator type."""
    if integrator_type == IntegratorType.SEMI_IMPLICIT:
        return semi_implicit_euler_integration
    elif integrator_type == IntegratorType.HEUN2:
        return heun2_integration


def semi_implicit_euler_integration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    base_acceleration_inertial: jtp.Vector,
    joint_accelerations: jtp.Vector,
) -> JaxSimModelData:
    """Integrate the system state using the semi-implicit Euler method."""
    # Step the dynamics forward.
    dt = model.time_step
    W_v̇_WB = base_acceleration_inertial
    s̈ = joint_accelerations

    B_H_W = Transform.inverse(data._base_transform).at[:3, :3].set(jnp.eye(3))
    BW_X_W = Adjoint.from_transform(B_H_W)

    base_velocity = jnp.hstack([data._base_linear_velocity, data._base_angular_velocity])
    new_base_velocity_inertial = base_velocity + dt * W_v̇_WB
    new_joint_velocities = data._joint_velocities + dt * s̈

    base_lin_velocity_inertial = new_base_velocity_inertial[0:3]

    new_base_velocity_mixed = BW_X_W @ new_base_velocity_inertial
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

    # The base angular velocity in mixed representation is equivalent to the one in inertial representation
    # See: S. Traversaro and A. Saccon, “Multibody Dynamics Notation (Version 2), pg.9"
    base_ang_velocity_inertial = base_ang_velocity_mixed

    link_transforms, link_velocities = jaxsim.rbda.forward_kinematics_model(
        model=model,
        base_position=new_base_position,
        base_quaternion=new_base_quaternion,
        joint_positions=new_joint_position,
        joint_velocities=new_joint_velocities,
        base_linear_velocity_inertial=base_lin_velocity_inertial,
        base_angular_velocity_inertial=base_ang_velocity_inertial,
    )

    return dataclasses.replace(
        data,
        _base_quaternion=new_base_quaternion,
        _base_position=new_base_position,
        _joint_positions=new_joint_position,
        _joint_velocities=new_joint_velocities,
        _base_linear_velocity=base_lin_velocity_inertial,
        _base_angular_velocity=base_ang_velocity_inertial,
        _link_transforms=link_transforms,
        _link_velocities=link_velocities,
    )


def heun2_integration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    **kwargs,
) -> js.data.JaxSimModelData:
    """
    Integrate the system state using Heun's second order method.

    This implementation follows:
      k₁ = f(x₀)
      x_trial = x₀ + Δt · k₁
      k₂ = f(x_trial)
      x_new = x₀ + Δt · 0.5 · (k₁ + k₂)

    The state fields have been moved to data itself (i.e. _joint_positions,
    _joint_velocities, _base_position, _base_quaternion, _base_linear_velocity,
    _base_angular_velocity), so we perform arithmetic on these fields.
    """
    dt = model.time_step

    # Compute the first slope: k₁ = f(x₀)
    k1 = js.ode.system_dynamics(model, data)

    # Compute trial state: x_trial = x₀ + dt * k₁
    base_velocity = data.base_velocity + dt * k1.base_velocity
    trial_data = data.replace(
        model=model,
        # Pass along the same velocity representation and cached fields.
        joint_positions=data.joint_positions + dt * k1.joint_positions,
        joint_velocities=data.joint_velocities + dt * k1.joint_velocities,
        base_position=data.base_position + dt * k1.base_position,
        base_quaternion=data.base_quaternion + dt * k1.base_quaternion,
        base_linear_velocity=base_velocity[:3],
        base_angular_velocity=base_velocity[3:],
    )

    # Compute the second slope: k₂ = f(x_trial)
    k2 = js.ode.system_dynamics(model, trial_data)

    # Combine slopes: x_new = x₀ + dt * 0.5*(k₁ + k₂)
    new_joint_positions = data.joint_positions + dt * 0.5 * (k1.joint_positions + k2.joint_positions)
    new_joint_velocities = data.joint_velocities + dt * 0.5 * (k1.joint_velocities + k2.joint_velocities)
    new_base_position = data.base_position + dt * 0.5 * (k1.base_position + k2.base_position)
    new_base_quaternion = data.base_quaternion + dt * 0.5 * (k1.base_quaternion + k2.base_quaternion)
    base_velocity = data.base_velocity + dt * 0.5 * (k1.base_velocity + k2.base_velocity)
    return data.replace(
        model=model,
        joint_positions=new_joint_positions,
        joint_velocities=new_joint_velocities,
        base_position=new_base_position,
        base_quaternion=new_base_quaternion,
        base_linear_velocity=base_velocity[:3],
        base_angular_velocity=base_velocity[3:],
    )
