from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.experimental.loops
import jax.flatten_util
import jax.numpy as jnp

import jaxsim.typing as jtp

Time = float
TimeHorizon = jtp.Vector

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

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Initial value of the loop carry
    init_val = (x0, t0, dx_dt(x0, t0)[1])

    def body_fun(idx: int, carry: Tuple) -> Tuple:

        # Unpack data from the carry
        x, t, _ = carry

        # Compute the state derivative
        dxdt, aux = dx_dt(x, t)

        # Integrate the dynamics and update the time
        x = jax.tree_map(lambda _x, _dxdt: _x + sub_step_dt * _dxdt, x, dxdt)
        t = t + sub_step_dt

        # Pack the carry data
        return x, t, aux

    x_tf, tf, aux_tf = jax.lax.fori_loop(
        lower=0, upper=num_sub_steps, body_fun=body_fun, init_val=init_val
    )

    return x_tf, aux_tf


def odeint_rk4_one_step(
    dx_dt: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    num_sub_steps: int = 1,
) -> Tuple[State, Dict[str, Any]]:

    # Compute the sub-step size.
    # We break dt in configurable sub-steps.
    dt = tf - t0
    sub_step_dt = dt / num_sub_steps

    # Initial value of the loop carry
    init_val = (x0, t0, dx_dt(x0, t0)[1])

    def body_fun(idx: int, carry: Tuple) -> Tuple:

        # Unpack data from the carry
        x, t, _ = carry

        # Helper to forward the state to compute k2 and k3 at midpoint and k4 at final
        euler_mid = lambda _x, _dxdt: _x + (0.5 * sub_step_dt) * _dxdt
        euler_fin = lambda _x, _dxdt: _x + sub_step_dt * _dxdt

        # Compute the RK4 slopes
        k1, aux = dx_dt(x, t)
        k2, _ = dx_dt(jax.tree_map(euler_mid, x, k1), t + 0.5 * sub_step_dt)
        k3, _ = dx_dt(jax.tree_map(euler_mid, x, k2), t + 0.5 * sub_step_dt)
        k4, _ = dx_dt(jax.tree_map(euler_fin, x, k3), t + sub_step_dt)

        # Average the slopes and compute the RK4 state derivative
        average = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dxdt = jax.tree_map(average, k1, k2, k3, k4)

        # Integrate the dynamics and update the time
        x = jax.tree_map(euler_fin, x, dxdt)
        t = t + sub_step_dt

        # Pack the carry data
        return x, t, aux

    x_tf, tf, aux_tf = jax.lax.fori_loop(
        lower=0, upper=num_sub_steps, body_fun=body_fun, init_val=init_val
    )

    return x_tf, aux_tf


# ===============================
# Adapter: single step -> horizon
# ===============================


def integrate_single_step_over_horizon(
    integrator_single_step: Callable[[Time, Time, State], Tuple[State, Dict[str, Any]]],
    t: TimeHorizon,
    x0: State,
) -> Tuple[State, Dict[str, Any]]:

    # We cannot use fori_loop because it does not support indexing buffer elements
    # using the index passed as body_fun input argument.
    # Assuming the state pytree not having any static attribute, we operate on its
    # flattened representation.
    with jax.experimental.loops.Scope() as s:

        # Dummy run to get the flattened dimensions
        _, aux0 = integrator_single_step(t[0], t[1], x0)
        x0_flat, restore_x_fn = jax.flatten_util.ravel_pytree(x0)
        aux0_flat, restore_aux_fn = jax.flatten_util.ravel_pytree(aux0)

        # Allocate the buffers to store flattened data over the entire horizon
        s.aux = jnp.zeros(shape=(t.size, aux0_flat.size))
        s.x_horizon = jnp.zeros(shape=(t.size, x0_flat.size))

        # Store the initial state and aux data
        s.aux = s.aux.at[0].set(aux0_flat)
        s.x_horizon = s.x_horizon.at[0].set(x0_flat)

        for i in s.range(0, t.size - 1):

            # Get the indices of the integration step
            t_start_idx, t_end_idx = (i, i + 1)

            # Define the integration interval
            t_start, t_end = t[t_start_idx], t[t_end_idx]

            # Define the initial condition of the ODE
            x_start = s.x_horizon[t_start_idx]

            # Integrate the ODE for a single step
            x_start_pytree = restore_x_fn(x_start)
            x_end_pytree, aux_pytree = integrator_single_step(
                t_start, t_end, x_start_pytree
            )

            # Flatten the pytrees
            x_end, _ = jax.flatten_util.ravel_pytree(x_end_pytree)
            aux, _ = jax.flatten_util.ravel_pytree(aux_pytree)

            # Store the final state
            s.x_horizon = s.x_horizon.at[t_end_idx].set(x_end.squeeze())

            # Store the flattened auxiliary data
            s.aux = s.aux.at[t_end_idx].set(aux.squeeze())

        x_horizon = jax.vmap(lambda x: restore_x_fn(x))(s.x_horizon)
        aux = jax.vmap(lambda a: restore_aux_fn(a))(s.aux)

    return x_horizon, aux


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


def odeint_rk4(
    func,
    y0: State,
    t: TimeHorizon,
    *args,
    num_sub_steps: int = 1,
    return_aux: bool = False
) -> Union[State, Tuple[State, Dict[str, Any]]]:

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
