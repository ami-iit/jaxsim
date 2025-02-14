import dataclasses
from collections.abc import Callable

import jax
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


def rk4_integration(
    model: js.model.JaxSimModel,
    data: JaxSimModelData,
    base_acceleration_inertial: jtp.Vector,
    joint_accelerations: jtp.Vector,
    link_forces: jtp.Vector,
    joint_torques: jtp.Vector,
) -> JaxSimModelData:
    """Integrate the system state using the Runge-Kutta 4 method."""

    dt = model.time_step

    def get_state_derivative(data_ode: JaxSimModelData) -> dict:

        # Safe normalize the quaternion.
        base_quaternion_norm = jaxsim.math.safe_norm(data_ode.base_quaternion)
        base_quaternion = data_ode.base_quaternion / jnp.where(
            base_quaternion_norm == 0, 1.0, base_quaternion_norm
        )

        return dict(
            base_position=data_ode.base_position,
            base_quaternion=base_quaternion,
            joint_positions=data_ode.joint_positions,
            base_linear_velocity=data_ode.base_velocity[0:3],
            base_angular_velocity=data_ode.base_velocity[3:6],
            joint_velocities=data_ode.joint_velocities,
        )

    def f(x) -> dict[str, jtp.Matrix]:

        with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

            data_ti = data.replace(
                model=model, **{k.lstrip("_"): v for k, v in x.items()}
            )

            return js.ode.system_dynamics(
                model=model,
                data=data_ti,
                link_forces=link_forces,
                joint_torques=joint_torques,
            )

    with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):
        x_t0 = get_state_derivative(data)

    euler_mid = lambda x, dxdt: x + (0.5 * dt) * dxdt
    euler_fin = lambda x, dxdt: x + dt * dxdt

    k1 = f(x_t0)
    k2 = f(jax.tree.map(euler_mid, x_t0, k1))
    k3 = f(jax.tree.map(euler_mid, x_t0, k2))
    k4 = f(jax.tree.map(euler_fin, x_t0, k3))

    # Average the slopes and compute the RK4 state derivative.
    average = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) / 6

    dxdt = jax.tree_util.tree_map(average, k1, k2, k3, k4)

    # Integrate the dynamics
    x_tf = jax.tree_util.tree_map(euler_fin, x_t0, dxdt)

    data_tf = dataclasses.replace(data, **{"_" + k: v for k, v in x_tf.items()})

    return data_tf.replace(model=model)


_INTEGRATORS_MAP: dict[js.model.Integrator, Callable[..., js.data.JaxSimModelData]] = {
    js.model.Integrator.SemiImplicitEuler: semi_implicit_euler_integration,
    js.model.Integrator.RungeKutta4: rk4_integration,
}
