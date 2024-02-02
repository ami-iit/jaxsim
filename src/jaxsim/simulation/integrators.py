from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import jaxsim.typing as jtp
from jaxsim.math.quaternion import Quaternion
from jaxsim.physics.algos.soft_contacts import SoftContactsState
from jaxsim.physics.model.physics_model_state import PhysicsModelState
from jaxsim.simulation.ode_data import ODEState
from jaxsim.sixd import se3, so3

Time = jtp.FloatLike
TimeStep = jtp.FloatLike
TimeHorizon = jtp.VectorLike

State = jtp.PyTree
StateDerivative = jtp.PyTree

StateDerivativeCallable = Callable[
    [State, Time], Tuple[StateDerivative, Dict[str, Any]]
]


# =======================
# Single-step integration
# =======================


def odeint_euler_one_step(
    dx_dt: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    num_sub_steps: int = 1,
) -> Tuple[State, Dict[str, Any]]:
    """
    Forward Euler integrator.

    Args:
        dx_dt: Callable that computes the state derivative.
        x0: Initial state.
        t0: Initial time.
        tf: Final time.
        num_sub_steps: Number of sub-steps to break the integration into.

    Returns:
        The final state and a dictionary including auxiliary data at t0.
    """

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Initialize the carry
    Carry = Tuple[State, Time]
    carry_init: Carry = (x0, t0)

    def body_fun(carry: Carry, xs: None) -> Tuple[Carry, None]:
        # Unpack the carry
        x_t0, t0 = carry

        # Compute the state derivative
        dxdt_t0, _ = dx_dt(x_t0, t0)

        # Integrate the dynamics
        x_tf = jax.tree_util.tree_map(
            lambda x, dxdt: x + sub_step_dt * dxdt, x_t0, dxdt_t0
        )

        # Update the time
        tf = t0 + sub_step_dt

        # Pack the carry
        carry = (x_tf, tf)

        return carry, None

    # Integrate over the given horizon
    (x_tf, _), _ = jax.lax.scan(
        f=body_fun, init=carry_init, xs=None, length=num_sub_steps
    )

    # Compute the aux dictionary at t0
    _, aux_t0 = dx_dt(x0, t0)

    return x_tf, aux_t0


def odeint_rk4_one_step(
    dx_dt: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    num_sub_steps: int = 1,
) -> Tuple[State, Dict[str, Any]]:
    """
    Runge-Kutta 4 integrator.

    Args:
        dx_dt: Callable that computes the state derivative.
        x0: Initial state.
        t0: Initial time.
        tf: Final time.
        num_sub_steps: Number of sub-steps to break the integration into.

    Returns:
        The final state and a dictionary including auxiliary data at t0.
    """

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Initialize the carry
    Carry = Tuple[State, Time]
    carry_init: Carry = (x0, t0)

    def body_fun(carry: Carry, xs: None) -> Tuple[Carry, None]:
        # Unpack the carry
        x_t0, t0 = carry

        # Helper to forward the state to compute k2 and k3 at midpoint and k4 at final
        euler_mid = lambda x, dxdt: x + (0.5 * sub_step_dt) * dxdt
        euler_fin = lambda x, dxdt: x + sub_step_dt * dxdt

        # Compute the RK4 slopes
        k1, _ = dx_dt(x_t0, t0)
        k2, _ = dx_dt(tree_map(euler_mid, x_t0, k1), t0 + 0.5 * sub_step_dt)
        k3, _ = dx_dt(tree_map(euler_mid, x_t0, k2), t0 + 0.5 * sub_step_dt)
        k4, _ = dx_dt(tree_map(euler_fin, x_t0, k3), t0 + sub_step_dt)

        # Average the slopes and compute the RK4 state derivative
        average = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dxdt = jax.tree_util.tree_map(average, k1, k2, k3, k4)

        # Integrate the dynamics
        x_tf = jax.tree_util.tree_map(euler_fin, x_t0, dxdt)

        # Update the time
        tf = t0 + sub_step_dt

        # Pack the carry
        carry = (x_tf, tf)

        return carry, None

    # Integrate over the given horizon
    (x_tf, _), _ = jax.lax.scan(
        f=body_fun, init=carry_init, xs=None, length=num_sub_steps
    )

    # Compute the aux dictionary at t0
    _, aux_t0 = dx_dt(x0, t0)

    return x_tf, aux_t0


def odeint_euler_semi_implicit_one_step(
    dx_dt: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,
    num_sub_steps: int = 1,
) -> Tuple[ODEState, Dict[str, Any]]:
    """
    Semi-implicit Euler integrator.

    Args:
        dx_dt: Callable that computes the state derivative.
        x0: Initial state as ODEState object.
        t0: Initial time.
        tf: Final time.
        num_sub_steps: Number of sub-steps to break the integration into.

    Returns:
        A tuple having as first element the final state as ODEState object,
        and as second element a dictionary including auxiliary data at t0.
    """

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Initialize the carry
    Carry = Tuple[ODEState, Time]
    carry_init: Carry = (x0, t0)

    def body_fun(carry: Carry, xs: None) -> Tuple[Carry, None]:
        # Unpack the carry
        x_t0, t0 = carry

        # Compute the state derivative.
        # We only keep the quantities related to the acceleration and discard those
        # related to the velocity since we are going to use those implicitly integrated
        # from the accelerations.
        StateDerivative = ODEState
        dxdt_t0: StateDerivative = dx_dt(x_t0, t0)[0]

        # Extract the initial position ∈ ℝ⁷⁺ⁿ and initial velocity ∈ ℝ⁶⁺ⁿ.
        # This integrator, contrarily to most of the other ones, is not generic.
        # It expects to operate on an x object of class ODEState.
        pos_t0 = x_t0.physics_model.position()
        vel_t0 = x_t0.physics_model.velocity()

        # Extract the velocity derivative
        d_vel_dt = dxdt_t0.physics_model.velocity()

        # =============================================
        # Perform semi-implicit Euler integration [1-4]
        # =============================================

        # 1. Integrate the accelerations obtaining the implicit velocities
        # 2. Compute the derivative of the generalized position
        # 3. Integrate the implicit velocities
        # 4. Integrate the remaining state

        # ----------------------------------------------------------------
        # 1. Integrate the accelerations obtaining the implicit velocities
        # ----------------------------------------------------------------

        vel_tf = vel_t0 + sub_step_dt * d_vel_dt

        # -----------------------------------------------------
        # 2. Compute the derivative of the generalized position
        # -----------------------------------------------------

        # Extract the implicit angular velocity and the initial base quaternion
        W_ω_WB = vel_tf[3:6]
        W_Q_B = x_t0.physics_model.base_quaternion

        # Compute the quaternion derivative and the base position derivative
        W_Qd_B = Quaternion.derivative(
            quaternion=W_Q_B, omega=W_ω_WB, omega_in_body_fixed=False
        ).squeeze()

        # Compute the transform of the mixed base frame at t0
        W_H_BW = jnp.vstack(
            [
                jnp.block([jnp.eye(3), jnp.vstack(x_t0.physics_model.base_position)]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

        # The derivative W_ṗ_B of the base position is the linear component of the
        # mixed velocity B[W]_v_WB. We need to compute it from the velocity in
        # inertial-fixed representation W_vl_WB.
        W_v_WB = vel_tf[0:6]
        BW_Xv_W = se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
        BW_vl_WB = (BW_Xv_W @ W_v_WB)[0:3]

        # Compute the derivative of the generalized position
        d_pos_tf = jnp.hstack([BW_vl_WB, W_Qd_B, vel_tf[6:]])

        # ------------------------------------
        # 3. Integrate the implicit velocities
        # ------------------------------------

        pos_tf = pos_t0 + sub_step_dt * d_pos_tf

        # ---------------------------------
        # 4. Integrate the remaining state
        # ---------------------------------

        # Integrate the derivative of the tangential material deformation
        m = x_t0.soft_contacts.tangential_deformation
        ṁ = dxdt_t0.soft_contacts.tangential_deformation
        tangential_deformation_tf = m + sub_step_dt * ṁ

        # Pack the new state into an ODEState object
        x_tf = ODEState(
            physics_model=PhysicsModelState(
                base_position=pos_tf[0:3],
                base_quaternion=pos_tf[3:7],
                joint_positions=pos_tf[7:],
                base_linear_velocity=vel_tf[0:3],
                base_angular_velocity=vel_tf[3:6],
                joint_velocities=vel_tf[6:],
            ),
            soft_contacts=SoftContactsState(
                tangential_deformation=tangential_deformation_tf
            ),
        )

        # Update the time
        tf = t0 + sub_step_dt

        # Pack the carry
        carry = (x_tf, tf)

        return carry, None

    # Integrate over the given horizon
    (x_tf, _), _ = jax.lax.scan(
        f=body_fun, init=carry_init, xs=None, length=num_sub_steps
    )

    # Compute the aux dictionary at t0
    _, aux_t0 = dx_dt(x0, t0)

    return x_tf, aux_t0


def odeint_euler_semi_implicit_manifold_one_step(
    dx_dt: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,
    num_sub_steps: int = 1,
) -> Tuple[ODEState, Dict[str, Any]]:
    """
    Semi-implicit Euler integrator with quaternion integration on SO(3).

    Args:
        dx_dt: Callable that computes the state derivative.
        x0: Initial state as ODEState object.
        t0: Initial time.
        tf: Final time.
        num_sub_steps: Number of sub-steps to break the integration into.

    Returns:
        A tuple having as first element the final state as ODEState object,
        and as second element a dictionary including auxiliary data at t0.
    """

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Integrate the quaternion on its manifold using the new angular velocity

    # Initialize the carry
    Carry = Tuple[ODEState, Time]
    carry_init: Carry = (x0, t0)

    def body_fun(carry: Carry, xs: None) -> Tuple[Carry, None]:
        # Unpack the carry
        x_t0, t0 = carry

        # Compute the state derivative.
        # We only keep the quantities related to the acceleration and discard those
        # related to the velocity since we are going to use those implicitly integrated
        # from the accelerations.
        StateDerivative = ODEState
        dxdt_t0: StateDerivative = dx_dt(x_t0, t0)[0]

        # Extract the initial position ∈ ℝ⁷⁺ⁿ and initial velocity ∈ ℝ⁶⁺ⁿ.
        # This integrator, contrarily to most of the other ones, is not generic.
        # It expects to operate on an x object of class ODEState.
        pos_t0 = x_t0.physics_model.position()
        vel_t0 = x_t0.physics_model.velocity()

        # Extract the velocity derivative
        d_vel_dt = dxdt_t0.physics_model.velocity()

        # =============================================
        # Perform semi-implicit Euler integration [1-4]
        # =============================================

        # 1. Integrate the accelerations obtaining the implicit velocities
        # 2. Compute the derivative of the generalized position (w/o quaternion)
        # 3. Integrate the implicit velocities (w/o quaternion)
        # 4. Integrate the remaining state
        # 5. Outside the loop: integrate the quaternion on SO(3) manifold

        # ----------------------------------------------------------------
        # 1. Integrate the accelerations obtaining the implicit velocities
        # ----------------------------------------------------------------

        vel_tf = vel_t0 + sub_step_dt * d_vel_dt

        # ----------------------------------------------------------------------
        # 2. Compute the derivative of the generalized position (w/o quaternion)
        # ----------------------------------------------------------------------

        # Compute the transform of the mixed base frame at t0
        W_H_BW = jnp.vstack(
            [
                jnp.block([jnp.eye(3), jnp.vstack(x_t0.physics_model.base_position)]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

        # The derivative W_ṗ_B of the base position is the linear component of the
        # mixed velocity B[W]_v_WB. We need to compute it from the velocity in
        # inertial-fixed representation W_vl_WB.
        W_v_WB = vel_tf[0:6]
        BW_Xv_W = se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
        BW_vl_WB = (BW_Xv_W @ W_v_WB)[0:3]

        # Compute the derivative of the generalized position excluding the quaternion
        pos_no_quat_t0 = jnp.hstack([pos_t0[0:3], pos_t0[7:]])
        d_pos_no_quat_tf = jnp.hstack([BW_vl_WB, vel_tf[6:]])

        # -----------------------------------------------------
        # 3. Integrate the implicit velocities (w/o quaternion)
        # -----------------------------------------------------

        pos_no_quat_tf = pos_no_quat_t0 + sub_step_dt * d_pos_no_quat_tf

        # ---------------------------------
        # 4. Integrate the remaining state
        # ---------------------------------

        # Integrate the derivative of the tangential material deformation
        m = x_t0.soft_contacts.tangential_deformation
        ṁ = dxdt_t0.soft_contacts.tangential_deformation
        tangential_deformation_tf = m + sub_step_dt * ṁ

        # Pack the new state into an ODEState object.
        # We store a zero quaternion as placeholder, it will be replaced later.
        x_tf = ODEState(
            physics_model=PhysicsModelState(
                base_position=pos_no_quat_tf[0:3],
                base_quaternion=jnp.zeros_like(x_t0.physics_model.base_quaternion),
                joint_positions=pos_no_quat_tf[3:],
                base_linear_velocity=vel_tf[0:3],
                base_angular_velocity=vel_tf[3:6],
                joint_velocities=vel_tf[6:],
            ),
            soft_contacts=SoftContactsState(
                tangential_deformation=tangential_deformation_tf
            ),
        )

        # Update the time
        tf = t0 + sub_step_dt

        # Pack the carry
        carry = (x_tf, tf)

        return carry, None

    # Integrate over the given horizon
    (x_no_quat_tf, _), _ = jax.lax.scan(
        f=body_fun, init=carry_init, xs=None, length=num_sub_steps
    )

    # ---------------------------------------------
    # 5. Integrate the quaternion on SO(3) manifold
    # ---------------------------------------------

    # Indices to convert quaternions between serializations
    to_xyzw = jnp.array([1, 2, 3, 0])
    to_wxyz = jnp.array([3, 0, 1, 2])

    # Get the initial quaternion and the implicitly integrated angular velocity
    W_ω_WB_tf = x_no_quat_tf.physics_model.base_angular_velocity
    W_Q_B_t0 = so3.SO3.from_quaternion_xyzw(x0.physics_model.base_quaternion[to_xyzw])

    # Integrate the quaternion on its manifold using the implicit angular velocity,
    # transformed in body-fixed representation since jaxlie uses this convention
    B_R_W = W_Q_B_t0.inverse().as_matrix()
    W_Q_B_tf = W_Q_B_t0 @ so3.SO3.exp(tangent=dt * B_R_W @ W_ω_WB_tf)

    # Store the quaternion in the final state
    x_tf = x_no_quat_tf.replace(
        physics_model=x_no_quat_tf.physics_model.replace(
            base_quaternion=W_Q_B_tf.as_quaternion_xyzw()[to_wxyz]
        )
    )

    # Compute the aux dictionary at t0
    _, aux_t0 = dx_dt(x0, t0)

    return x_tf, aux_t0


# ===============================
# Adapter: single step -> horizon
# ===============================


def integrate_single_step_over_horizon(
    integrator_single_step: Callable[[Time, Time, State], Tuple[State, Dict[str, Any]]],
    t: TimeHorizon,
    x0: State,
) -> Tuple[State, Dict[str, Any]]:
    """
    Integrate a single-step integrator over a given horizon.

    Args:
        integrator_single_step: A single-step integrator.
        t: The vector of time instants of the integration horizon.
        x0: The initial state of the integration horizon.

    Returns:
        The final state and auxiliary data produced by the integrator.
    """

    # Initialize the carry
    carry_init = (x0, t)

    def body_fun(carry: Tuple, idx: int) -> Tuple[Tuple, jtp.PyTree]:
        # Unpack the carry
        x_t0, horizon = carry

        # Get the integration interval
        t0 = horizon[idx]
        tf = horizon[idx + 1]

        # Perform a single-step integration of the ODE
        x_tf, aux_t0 = integrator_single_step(t0, tf, x_t0)

        # Prepare returned data
        out = (x_t0, aux_t0)
        carry = (x_tf, horizon)

        return carry, out

    # Integrate over the given horizon
    _, (x_horizon, aux_horizon) = jax.lax.scan(
        f=body_fun, init=carry_init, xs=jnp.arange(start=0, stop=len(t), dtype=int)
    )

    return x_horizon, aux_horizon


# ===================================================================
# Integration over horizon (same APIs of jax.experimental.ode.odeint)
# ===================================================================


def odeint_euler(
    func,
    y0: State,
    t: TimeHorizon,
    *args,
    num_sub_steps: int = 1,
    return_aux: bool = False
) -> Union[State, Tuple[State, Dict[str, Any]]]:
    """
    Integrate a system of ODEs using the Euler method.

    Args:
        func: A function that computes the time-derivative of the state.
        y0: The initial state.
        t: The vector of time instants of the integration horizon.
        *args: Additional arguments to be passed to the function func.
        num_sub_steps: The number of sub-steps to be performed within each integration step.
        return_aux: Whether to return the auxiliary data produced by the integrator.

    Returns:
        The state of the system at the end of the integration horizon, and optionally
        the auxiliary data produced by the integrator.
    """

    # Close func over additional inputs and parameters
    dx_dt_closure_aux = lambda x, ts: func(x, ts, *args)

    # Close one-step integration over its arguments
    integrator_single_step = lambda t0, tf, x0: odeint_euler_one_step(
        dx_dt=dx_dt_closure_aux, x0=x0, t0=t0, tf=tf, num_sub_steps=num_sub_steps
    )

    # Integrate the state and compute optional auxiliary data over the horizon
    out, aux = integrate_single_step_over_horizon(
        integrator_single_step=integrator_single_step, t=t, x0=y0
    )

    return (out, aux) if return_aux else out


def odeint_euler_semi_implicit(
    func,
    y0: ODEState,
    t: TimeHorizon,
    *args,
    num_sub_steps: int = 1,
    return_aux: bool = False
) -> Union[ODEState, Tuple[ODEState, Dict[str, Any]]]:
    """
    Integrate a system of ODEs using the Semi-Implicit Euler method.

    Args:
        func: A function that computes the time-derivative of the state.
        y0: The initial state as ODEState object.
        t: The vector of time instants of the integration horizon.
        *args: Additional arguments to be passed to the function func.
        num_sub_steps: The number of sub-steps to be performed within each integration step.
        return_aux: Whether to return the auxiliary data produced by the integrator.

    Returns:
        The state of the system at the end of the integration horizon as ODEState object,
        and optionally the auxiliary data produced by the integrator.
    """

    # Close func over additional inputs and parameters
    dx_dt_closure_aux = lambda x, ts: func(x, ts, *args)

    # Close one-step integration over its arguments
    integrator_single_step = lambda t0, tf, x0: odeint_euler_semi_implicit_one_step(
        dx_dt=dx_dt_closure_aux, x0=x0, t0=t0, tf=tf, num_sub_steps=num_sub_steps
    )

    # Integrate the state and compute optional auxiliary data over the horizon
    out, aux = integrate_single_step_over_horizon(
        integrator_single_step=integrator_single_step, t=t, x0=y0
    )

    return (out, aux) if return_aux else out


def odeint_rk4(
    func,
    y0: State,
    t: TimeHorizon,
    *args,
    num_sub_steps: int = 1,
    return_aux: bool = False
) -> Union[State, Tuple[State, Dict[str, Any]]]:
    """
    Integrate a system of ODEs using the Runge-Kutta 4 method.

    Args:
        func: A function that computes the time-derivative of the state.
        y0: The initial state.
        t: The vector of time instants of the integration horizon.
        *args: Additional arguments to be passed to the function func.
        num_sub_steps: The number of sub-steps to be performed within each integration step.
        return_aux: Whether to return the auxiliary data produced by the integrator.

    Returns:
        The state of the system at the end of the integration horizon, and optionally
        the auxiliary data produced by the integrator.
    """

    # Close func over additional inputs and parameters
    dx_dt_closure = lambda x, ts: func(x, ts, *args)

    # Close one-step integration over its arguments
    integrator_single_step = lambda t0, tf, x0: odeint_rk4_one_step(
        dx_dt=dx_dt_closure, x0=x0, t0=t0, tf=tf, num_sub_steps=num_sub_steps
    )

    # Integrate the state and compute optional auxiliary data over the horizon
    out, aux = integrate_single_step_over_horizon(
        integrator_single_step=integrator_single_step, t=t, x0=y0
    )

    return (out, aux) if return_aux else out
