import enum
import functools
from typing import Any, Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

from jaxsim import typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.simulation.integrators import Time, TimeHorizon, TimeStep
from jaxsim.simulation.ode_data import ODEState
from jaxsim.sixd import so3

RTOL_DEFAULT = 1.4e-8
ATOL_DEFAULT = 1.4e-8

SAFETY_DEFAULT = 0.9
BETA_MIN_DEFAULT = 1.0 / 10
BETA_MAX_DEFAULT = 2.5
MAX_STEP_REJECTIONS_DEFAULT = 5

# Contrarily to the fixed-step integrators that operate on generic PyTrees,
# these variable-step integrators operate only on arrays (that could be the
# flatted PyTree).
State = jtp.Vector
StateNext = State
StateDerivative = jtp.Vector
StateDerivativeCallable = Callable[
    [State, Time], tuple[StateDerivative, dict[str, Any]]
]


class AdaptiveIntegratorType(enum.IntEnum):
    HeunEuler = enum.auto()
    BogackiShampine = enum.auto()


class VariableStepIntegratorFactory:
    @staticmethod
    def get(integrator_type: AdaptiveIntegratorType) -> tuple[Callable, int, int]:
        """"""

        match integrator_type:
            case AdaptiveIntegratorType.HeunEuler:
                p = int(2)
                p̂ = int(p - 1)
                return heun_euler, p, p̂

            case AdaptiveIntegratorType.BogackiShampine:
                p = int(3)
                p̂ = int(p - 1)
                return bogacki_shampine, p, p̂

            case _:
                raise ValueError(integrator_type)


# =================
# Utility functions
# =================


@functools.partial(jax.jit, static_argnames=["f"])
def estimate_step_size(
    x0: State,
    t0: Time,
    f: StateDerivativeCallable,
    order: jtp.IntLike,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
) -> tuple[jtp.Float, StateDerivative]:
    """
    Compute the initial step size to warm-start an adaptive integrator.

    Args:
        x0: The initial state.
        t0: The initial time.
        f: The state derivative function $f(x, t)$.
        order: The order $p$ of an integrator with truncation error $\mathcal{O}(\Delta t^{p+1})$.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        A tuple containing the computed initial step size
        and the state derivative $\dot{x} = f(x_0, t_0)$.

    Note:
        Refer to the following reference for the implementation details:

        Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        E. Hairer, S. P. Norsett G. Wanner.
    """

    # Compute the state derivative at the initial state.
    ẋ0 = f(x0, t0)[0]

    # Scale the initial state and its derivative.
    scale0 = atol + jnp.abs(x0) * rtol
    scale1 = atol + jnp.abs(ẋ0) * rtol
    d0 = jnp.linalg.norm(jnp.abs(x0) / scale0, ord=jnp.inf)  # noqa
    d1 = jnp.linalg.norm(jnp.abs(ẋ0) / scale1, ord=jnp.inf)  # noqa

    # Compute the first guess of the initial step size.
    h0 = jnp.where(jnp.minimum(d0, d1) <= 1e-5, 1e-6, 0.01 * d0 / d1)

    # Compute the next state and its derivative.
    x1 = x0 + h0 * ẋ0
    ẋ1 = f(x1, t0 + h0)[0]

    # Scale the difference of the state derivatives.
    scale2 = atol + jnp.maximum(jnp.abs(ẋ1), jnp.abs(ẋ0)) * rtol
    d2 = jnp.linalg.norm(jnp.abs((ẋ1 - ẋ0) / scale2), ord=jnp.inf) / h0  # noqa

    # Compute the second guess of the initial step size.
    h1 = jnp.where(
        jnp.maximum(d1, d2) <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    # Propose the final guess of the initial step size.
    # Also return the state derivative computed at the initial state since
    # it is likely a quantity that needs to be computed again later.
    return jnp.array(jnp.minimum(100.0 * h0, h1), dtype=float), ẋ0


def scale_array(
    x1: State,
    x2: State | StateNext | None = None,
    rtol: jax.typing.ArrayLike = RTOL_DEFAULT,
    atol: jax.typing.ArrayLike = ATOL_DEFAULT,
) -> jax.Array:
    """
    Compute the component-wise state scale to use for the error estimate of
    the local integration error.

    Args:
        x1: The first state, usually $x(t_0)$.
        x2: The optional second state, usually $x(t_f)$.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        The component-wise state scale to use for the error estimate of
        the local integration error.
    """

    # Use a zeroed second state if not provided.
    x2 = x2 if x2 is not None else jnp.zeros_like(x1)

    # Return: atol + max(|x1|, |x2|) * rtol.
    return (
        atol
        + jnp.vstack(
            [
                jnp.abs(jnp.atleast_1d(x1.squeeze())),
                jnp.abs(jnp.atleast_1d(x2.squeeze())),
            ]
        ).max(axis=0)
        * rtol
    )


def error_local(
    x0: State,
    xf: StateNext,
    error_estimate: State | None = None,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    norm_ord: jtp.IntLike | jtp.FloatLike = jnp.inf,
) -> jtp.Float:
    """
    Compute the local integration error.

    Args:
        x0: The initial state $x(t_0)$.
        xf: The final state $x(t_f)$.
        error_estimate: The optional error estimate. In not given, it is computed as the
            absolute value of the difference between the final and initial states.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.
        norm_ord: The norm to use to compute the error. Default is the infinity norm.

    Returns:
        The local integration error.
    """

    # First compute the component-wise scale using the initial and final states.
    sc = scale_array(x1=x0, x2=xf, rtol=rtol, atol=atol)

    # Compute the error estimate if not given.
    error_estimate = error_estimate if error_estimate is not None else jnp.abs(xf - x0)

    # Then, compute the local error by properly scaling the given error estimate and apply
    # the desired norm (default is infinity norm, that is the maximum absolute value).
    return jnp.linalg.norm(error_estimate / sc, ord=norm_ord)


@functools.partial(jax.jit, static_argnames=["f"])
def runge_kutta_from_butcher_tableau(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    c: jax.Array,
    b: jax.Array,
    A: jax.Array,
    dxdt0: StateDerivative | None = None,
) -> tuple[State, StateDerivative, State, dict[str, Any]]:
    """
    Advance a state vector by integrating a system dynamics with a Runge-Kutta integrator.

    Args:
        x0: The initial state.
        t0: The initial time.
        dt: The integration time step.
        f: The state derivative function :math:`f(x, t)`.
        c: The :math:`\mathbf{c}` parameter of the Butcher tableau.
        b: The :math:`\mathbf{b}` parameter of the Butcher tableau.
        A: The :math:`\mathbf{A}` parameter of the Butcher tableau.
        dxdt0: The optional pre-computed state derivative at the
            initial :math:`(x_0, t_0)`, useful for FSAL schemes.

    Returns:
        A tuple containing the next state, the intermediate state derivatives
        :math:`\mathbf{k}_i`, the component-wise error estimate, and the auxiliary
        dictionary returned by `f`.

    Note:
        If `b.T` has multiple rows (used e.g. in embedded Runge-Kutta methods), the first
        returned argument is a 2D array having as many rows as `b.T`. Each i-th row
        corresponds to the solution computed with coefficients of the i-th row of `b.T`.
    """

    # Adjust sizes of Butcher tableau arrays.
    c = jnp.atleast_1d(c.squeeze())
    b = jnp.atleast_2d(b.squeeze())
    A = jnp.atleast_2d(A.squeeze())

    # Use a symbol for the time step.
    Δt = dt

    # Initialize the carry of the for loop with the stacked kᵢ vectors.
    carry0 = jnp.zeros(shape=(c.size, x0.size), dtype=float)

    # Allow FSAL (first-same-as-last) property by passing ẋ0 = f(x0, t0) from
    # the previous iteration.
    get_ẋ0 = lambda: dxdt0 if dxdt0 is not None else f(x0, t0)[0]

    # We use a `jax.lax.scan` to have only a single instance of the compiled `f` function.
    # Otherwise, if we compute e.g. for RK4 sequentially, the jit-compiled code
    # would include 4 repetitions of the `f` logic, making everything extremely slow.
    def scan_body(carry: jax.Array, i: int | jax.Array) -> tuple[Any, None]:
        """"""

        # Unpack the carry
        k = carry

        def compute_ki():
            xi = x0 + Δt * jnp.dot(A[i, :], k)
            ti = t0 + c[i] * Δt
            return f(xi, ti)[0]

        # This selector enables FSAL property in the first iteration (i=0).
        ki = jax.lax.select(
            pred=(i == 0),
            on_true=get_ẋ0(),
            on_false=compute_ki(),
        )

        k = k.at[i].set(ki)
        return k, None

    # Compute the state derivatives k
    k, _ = jax.lax.scan(
        f=scan_body,
        init=carry0,
        xs=jnp.arange(c.size),
    )

    # Compute the output state.
    # Note that z contains as many new states as the rows of `b.T`.
    z = x0 + Δt * jnp.dot(b.T, k)

    # Compute the error estimate if `b.T` has multiple rows, otherwise return 0.
    error_estimate = jax.lax.select(
        pred=b.T.shape[0] == 1,
        on_true=jnp.zeros_like(x0, dtype=float),
        on_false=dt * jnp.dot(b.T[-1] - b.T[0], k),
    )

    # TODO: populate the auxiliary dictionary
    return z, k, error_estimate, dict()


@functools.partial(jax.jit, static_argnames=["f", "tf_next_state"])
def heun_euler(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
) -> tuple[State, jtp.Float, dict[str, Any]]:
    """"""

    # b parameter of Butcher tableau.
    b = jnp.array(
        [
            [1 / 2, 1 / 2],
            [1, 0],
        ]
    ).T

    # c parameter of Butcher tableau.
    c = jnp.array([0, 1])

    # A parameter of Butcher tableau.
    A = jnp.array(
        [
            [0, 0],
            [1, 0],
        ]
    )

    # Integrate the state with the resulting integrator.
    (
        (xf_higher, xf_lower),
        (_, k2),
        error_estimate,
        aux_dict,
    ) = runge_kutta_from_butcher_tableau(
        x0=x0,
        t0=t0,
        dt=dt,
        f=f,
        c=c,
        b=b,
        A=A,
        f0=aux_dict.get("f0", None) if aux_dict is not None else None,
    )

    # Take the higher-order solution as the next state, and optionally apply
    # the user-defined transformation.
    x_next = tf_next_state(x0, xf_higher, t0, dt)

    # Calculate the local integration error.
    error = error_local(
        x0=x0, xf=x_next, error_estimate=error_estimate, rtol=rtol, atol=atol
    )

    # Enable FSAL (first-same-as-last) property by returning k2.
    aux_dict = dict(f0=k2)

    return x_next, error, aux_dict


@functools.partial(jax.jit, static_argnames=["f", "tf_next_state"])
def bogacki_shampine(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
) -> tuple[State, jtp.Float, dict[str, Any]]:
    """"""

    # b parameter of Butcher tableau.
    b = jnp.array(
        [
            [2 / 9, 1 / 3, 4 / 9, 0],
            [7 / 24, 1 / 4, 1 / 3, 1 / 8],
        ]
    ).T

    # c parameter of Butcher tableau.
    c = jnp.array([0, 1 / 2, 3 / 4, 1])

    # A parameter of Butcher tableau.
    A = jnp.array(
        [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 3 / 4, 0, 0],
            [2 / 9, 1 / 3, 4 / 9, 0],
        ]
    )

    # Integrate the state with the resulting integrator.
    (
        (xf_higher, xf_lower),
        (_, _, _, k4),
        error_estimate,
        aux_dict,
    ) = runge_kutta_from_butcher_tableau(
        x0=x0,
        t0=t0,
        dt=dt,
        f=f,
        c=c,
        b=b,
        A=A,
        dxdt0=aux_dict.get("f0", None) if aux_dict is not None else None,
    )

    # Take the higher-order solution as the next state, and optionally apply the
    # user-defined transformation.
    x_next = tf_next_state(x0, xf_higher, t0, dt)

    # Calculate the local integration error.
    error = error_local(
        x0=x0, xf=x_next, error_estimate=error_estimate, rtol=rtol, atol=atol
    )

    # Enable FSAL (first-same-as-last) property by returning k4.
    aux_dict = dict(f0=k4)

    return x_next, error, aux_dict


# ==========================================
# Variable-step RK integrators (single step)
# ==========================================


@functools.partial(
    jax.jit,
    static_argnames=["f", "integrator_type", "debug_buffers_size", "tf_next_state"],
)
def odeint_embedded_rk_one_step(
    f: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    *,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    dt0: jtp.FloatLike | None = None,
    dt_min: jtp.FloatLike = -jnp.inf,
    dt_max: jtp.FloatLike = jnp.inf,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    debug_buffers_size: int | None = None,
    max_step_rejections: jtp.IntLike = MAX_STEP_REJECTIONS_DEFAULT,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
) -> tuple[State, dict[str, Any]]:
    """"""

    # Get the integrator and its order
    rk_method, p, p̂ = VariableStepIntegratorFactory.get(integrator_type=integrator_type)
    q = jnp.minimum(p, p̂)

    # Close the integrator over its optional arguments
    rk_method_closed = lambda x0, t0, Δt, aux_dict: rk_method(
        x0=x0,
        t0=t0,
        dt=Δt,
        f=f,
        rtol=rtol,
        atol=atol,
        aux_dict=aux_dict,
        tf_next_state=tf_next_state,
    )

    # Compute the initial step size considering the order of the integrator,
    # and clip it to the given bounds, if necessary.
    # The function also returns the evaluation of the state derivative at the
    # initial state, saving a call to the f function.
    Δt0, ẋ0 = jax.lax.cond(
        pred=jnp.where(dt0 is None, 0.0, dt0) == 0.0,
        true_fun=lambda _: estimate_step_size(
            x0=x0, t0=t0, f=f, order=p, atol=atol, rtol=rtol
        ),
        false_fun=lambda _: (dt0, f(x0, t0)[0]),
        operand=None,
    )

    # Clip the initial step size to the given bounds, if necessary.
    Δt0 = jnp.clip(
        a=Δt0,
        a_min=jnp.minimum(dt_min, tf - t0),
        a_max=jnp.minimum(dt_max, tf - t0),
    )

    # Initialize the size of the debug buffers.
    debug_buffers_size = debug_buffers_size if debug_buffers_size is not None else 0

    # Allocate the debug buffers.
    debug_buffers = (
        dict(
            idx=jnp.array(0, dtype=int),
            x_steps=-jnp.inf
            * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
            t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
            dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
        )
        if debug_buffers_size > 0
        else dict()
    )

    # Initialize the debug buffers with the initial state and time.
    if debug_buffers_size > 0:
        debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
        debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t0)

    # =========================================================
    # While loop to reach tf from t0 using an adaptive timestep
    # =========================================================

    # Initialize the carry of the while loop.
    Carry = tuple
    carry0: Carry = (
        Δt0,
        x0,
        t0,
        dict(f0=ẋ0),
        jnp.array(0, dtype=int),
        False,
        debug_buffers,
    )

    def cond_outer(carry: Carry) -> jtp.Bool:
        _, _, _, _, _, break_loop, _ = carry
        return jnp.logical_not(break_loop)

    # Each loop is an integration step with variable Δt.
    # Depending on the integration error, the step could be discarded and the while body
    # ran again from the same (x0, t0) but with a smaller Δt.
    # We run these loops until the final time tf is reached.
    def body_outer(carry: Carry) -> Carry:
        """While loop body."""

        # Unpack the carry.
        Δt0, x0, t0, carry_integrator, discarded_steps, _, debug_buffers = carry

        # Let's take care of the final (variable) step.
        # We want the final Δt to let us reach tf exactly.
        # Then we can exit the while loop.
        Δt0 = jnp.where(t0 + Δt0 < tf, Δt0, tf - t0)
        break_loop = jnp.where(t0 + Δt0 < tf, False, True)

        # Calculate the next initial state and the corresponding integration error.
        # We enable FSAL (first-same-as-last) through the aux_dict (carry_integrator).
        x0_next, error, carry_integrator_next = rk_method_closed(
            x0, t0, Δt0, carry_integrator
        )

        # Shrink the Δt every time by the safety factor.
        # The β parameters define the bounds of the timestep update factor.
        s = jnp.clip(safety, a_min=0.0, a_max=1.0)
        β_min = jnp.maximum(0.0, beta_min)
        β_max = jnp.maximum(β_min, beta_max)

        # Compute the next Δt from the desired integration error.
        # This new time step is accepted if error <= 1.0, otherwise it is rejected.
        Δt_next = Δt0 * jnp.clip(
            a=s * jnp.power(1 / error, 1 / (q + 1)),
            a_min=β_min,
            a_max=β_max,
        )

        def accept_step(debug_buffers: dict[str, Any]):
            if debug_buffers_size > 0:
                idx = debug_buffers["idx"]
                x_steps = debug_buffers["x_steps"]
                t_steps = debug_buffers["t_steps"]
                dt_steps = debug_buffers["dt_steps"]
                #
                idx = jnp.minimum(idx + 1, len(t_steps) - 1)
                x_steps = x_steps.at[idx].set(x0_next)
                t_steps = t_steps.at[idx].set(t0 + Δt0)
                dt_steps = dt_steps.at[idx - 1].set(Δt0)
                #
                debug_buffers = dict(
                    idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
                )

            return (
                x0_next,
                t0 + Δt0,
                jnp.clip(Δt_next, dt_min, dt_max),
                carry_integrator_next,
                jnp.array(0, dtype=int),
                debug_buffers,
            )

        def reject_step(debug_buffers):
            return (
                x0,
                t0,
                jnp.clip(Δt_next, dt_min, dt_max),
                carry_integrator,
                discarded_steps + 1,
                debug_buffers,
            )

        (
            x0_next,
            t0_next,
            Δt_next,
            carry_integrator,
            discarded_steps,
            debug_buffers,
        ) = jax.lax.cond(
            pred=jnp.logical_or(
                jnp.logical_or(error <= 1.0, Δt_next < dt_min),
                discarded_steps >= max_step_rejections,
            ),
            true_fun=accept_step,
            true_operand=debug_buffers,
            false_fun=reject_step,
            false_operand=debug_buffers,
        )

        # Even if we thought that this while loop was the last one, maybe the step was
        # discarded and the Δt shrank
        break_loop = jnp.where(t0_next + Δt_next < tf, False, break_loop)

        # If this is the last while loop, we want to make sure that the returned Δt
        # is not the one that got shrank for reaching tf, but the last one computed
        # from the desired integration error to properly warm-start the next call.
        Δt_next = jnp.where(break_loop, Δt0, Δt_next)

        return (
            Δt_next,
            x0_next,
            t0_next,
            carry_integrator,
            discarded_steps,
            break_loop,
            debug_buffers,
        )

    Δt_final, x0_final, t0_final, _, _, _, debug_buffers = jax.lax.while_loop(
        cond_fun=cond_outer,
        body_fun=body_outer,
        init_val=carry0,
    )

    xf = x0_final
    Δt = Δt_final
    tf = t0_final + Δt_final

    return xf, dict(dt=Δt) | debug_buffers


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "semi_implicit_quaternion_integration",
        "debug_buffers_size",
    ],
)
def odeint_embedded_rk_manifold_one_step(
    f: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,
    *,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    physics_model: PhysicsModel,
    dt0: jtp.FloatLike | None = None,
    dt_min: jtp.FloatLike = -jnp.inf,
    dt_max: jtp.FloatLike = jnp.inf,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    semi_implicit_quaternion_integration: jtp.BoolLike = True,
    debug_buffers_size: int | None = None,
    max_step_rejections: jtp.IntLike = MAX_STEP_REJECTIONS_DEFAULT,
) -> tuple[ODEState, dict[str, Any]]:
    """"""

    def tf_next_state(x0: State, xf: State, t0: Time, dt: TimeStep) -> State:
        """
        Replace the quaternion in the final state with the one implicitly integrated
        on the SO(3) manifold.
        """

        # Convert the flat state to an ODEState pytree
        x0_ode_state = ODEState.deserialize(data=x0, physics_model=physics_model)
        xf_ode_state = ODEState.deserialize(data=xf, physics_model=physics_model)

        # Indices to convert quaternions between serializations
        to_xyzw = jnp.array([1, 2, 3, 0])
        to_wxyz = jnp.array([3, 0, 1, 2])

        # Get the initial quaternion and the inertial-fixed angular velocity
        W_ω_WB_t0 = x0_ode_state.physics_model.base_angular_velocity
        W_ω_WB_tf = xf_ode_state.physics_model.base_angular_velocity
        W_Q_B_t0 = so3.SO3.from_quaternion_xyzw(
            x0_ode_state.physics_model.base_quaternion[to_xyzw]
        )

        # Integrate implicitly the quaternion on its manifold using the angular velocity
        # transformed in body-fixed representation since jaxlie uses this convention
        B_R_W = W_Q_B_t0.inverse().as_matrix()
        W_ω_WB = W_ω_WB_tf if semi_implicit_quaternion_integration else W_ω_WB_t0
        W_Q_B_tf = W_Q_B_t0 @ so3.SO3.exp(tangent=dt * B_R_W @ W_ω_WB)

        # Store the quaternion in the final state
        xf_ode_state_manifold = xf_ode_state.replace(
            physics_model=xf_ode_state.physics_model.replace(
                base_quaternion=W_Q_B_tf.as_quaternion_xyzw()[to_wxyz]
            )
        )

        return xf_ode_state_manifold.flatten()

    # Flatten the ODEState. We use the unflatten_fn to convert the flat state back.
    x0_flat, unflatten_fn = jax.flatten_util.ravel_pytree(x0)

    # Integrate the flat ODEState with the embedded Runge-Kutta integrator.
    xf_flat, aux_dict = odeint_embedded_rk_one_step(
        f=f,
        x0=x0_flat,
        t0=t0,
        tf=tf,
        integrator_type=integrator_type,
        dt0=dt0,
        dt_min=dt_min,
        dt_max=dt_max,
        rtol=rtol,
        atol=atol,
        safety=safety,
        beta_min=beta_min,
        beta_max=beta_max,
        debug_buffers_size=debug_buffers_size,
        max_step_rejections=max_step_rejections,
        tf_next_state=tf_next_state,
    )

    # Convert the flat state back to ODEState.
    # Note that the aux_dict might contain flattened data that is not post-processed.
    xf = unflatten_fn(xf_flat)

    return xf, aux_dict


# ===============================
# Integration over a time horizon
# ===============================


@functools.partial(
    jax.jit,
    static_argnames=["odeint_adaptive_one_step", "debug_buffers_size_per_step"],
)
def _ode_integration_adaptive_template(
    x0: State,
    t: TimeHorizon,
    *,
    odeint_adaptive_one_step: Callable[
        [State, Time, Time, TimeStep], tuple[StateNext, dict[str, Any]]
    ],
    dt0: jax.Array | float | None = None,
    debug_buffers_size_per_step: int | None = None,
) -> tuple[StateNext, dict[str, Any]]:
    """"""

    # Adjust some of the input arguments.
    x0 = jnp.array(x0, dtype=float)
    dt0 = (
        jnp.array(dt0, dtype=float) if dt0 is not None else jnp.array(0.0, dtype=float)
    )

    debug_buffers_size = (
        debug_buffers_size_per_step * len(t)
        if debug_buffers_size_per_step is not None
        else 0
    )

    debug_buffers = (
        dict(
            idx=jnp.array(0, dtype=int),
            x_steps=-jnp.inf
            * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
            t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
            dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
        )
        if debug_buffers_size > 0
        else dict()
    )

    if debug_buffers_size > 0:
        debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
        debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t[0])

    # =================================================
    # For loop to integrate on the horizon defined by t
    # =================================================

    Carry = tuple
    carry0 = (x0, dt0, debug_buffers)

    def body(carry: Carry, i: float | jax.Array) -> tuple[Carry, jax.Array]:
        """For loop body."""

        # Unpack the carry.
        x0, dt0, debug_buffers = carry

        # Calculate the final state (the integrator can take an arbitrary number of steps)
        # and the auxiliary data (e.g. the last dt and the debug buffers).
        xf, dict_aux = odeint_adaptive_one_step(x0, t[i], t[i + 1], dt0)

        if debug_buffers_size > 0:
            # Get the horizon data
            idx = debug_buffers["idx"]
            x_steps = debug_buffers["x_steps"]
            t_steps = debug_buffers["t_steps"]
            dt_steps = debug_buffers["dt_steps"]

            # Get the single-step data
            x_odeint = dict_aux["x_steps"]
            t_odeint = dict_aux["t_steps"]
            dt_odeint = dict_aux["dt_steps"]

            # Merge the buffers
            x_steps = jax.lax.dynamic_update_slice(x_steps, x_odeint, (idx, 0))
            t_steps = jax.lax.dynamic_update_slice(t_steps, t_odeint, (idx,))
            dt_steps = jax.lax.dynamic_update_slice(dt_steps, dt_odeint, (idx,))

            # Advance the index
            idx_odeint = dict_aux["idx"]
            idx += idx_odeint

            debug_buffers = dict(
                idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
            )

        return (xf, dict_aux["dt"], debug_buffers), xf

    (_, dt_final, debug_buffers), X = jax.lax.scan(
        f=body,
        init=carry0,
        xs=jnp.arange(start=0, stop=len(t) - 1, dtype=int),
    )

    return (
        jnp.vstack([jnp.atleast_1d(x0.squeeze()), jnp.atleast_2d(X.squeeze())]),
        debug_buffers | dict(dt=dt_final),
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "debug_buffers_size_per_step",
        "tf_next_state",
    ],
)
def ode_integration_embedded_rk(
    x0: State,
    t: TimeHorizon,
    *,
    f: StateDerivativeCallable,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    dt0: jax.Array | float | None = None,
    dt_min: float = -jnp.inf,
    dt_max: float = jnp.inf,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    debug_buffers_size_per_step: int | None = None,
    max_step_rejections: int = MAX_STEP_REJECTIONS_DEFAULT,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
) -> tuple[StateNext, dict[str, Any]]:
    """"""

    # Select the target one-step integrator.
    odeint_adaptive_one_step = lambda x0, t0, tf, dt0: odeint_embedded_rk_one_step(
        f=f,
        x0=x0,
        t0=t0,
        tf=tf,
        integrator_type=integrator_type,
        dt0=dt0,
        dt_min=dt_min,
        dt_max=dt_max,
        rtol=rtol,
        atol=atol,
        safety=safety,
        beta_min=beta_min,
        beta_max=beta_max,
        debug_buffers_size=debug_buffers_size_per_step,
        max_step_rejections=max_step_rejections,
        tf_next_state=tf_next_state,
    )

    # Integrate the state with an adaptive timestep over the horizon defined by `t`.
    return _ode_integration_adaptive_template(
        x0=x0,
        t=t,
        odeint_adaptive_one_step=odeint_adaptive_one_step,
        dt0=dt0,
        debug_buffers_size_per_step=debug_buffers_size_per_step,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "semi_implicit_quaternion_integration",
        "debug_buffers_size_per_step",
    ],
)
def ode_integration_embedded_rk_manifold(
    x0: ODEState,
    t: TimeHorizon,
    *,
    f: StateDerivativeCallable,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    physics_model: PhysicsModel,
    dt0: jax.Array | float | None = None,
    dt_min: float = -jnp.inf,
    dt_max: float = jnp.inf,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    semi_implicit_quaternion_integration: jtp.BoolLike = True,
    debug_buffers_size_per_step: int | None = None,
    max_step_rejections: int = MAX_STEP_REJECTIONS_DEFAULT,
) -> tuple[ODEState, dict[str, Any]]:
    """"""

    # Define the functions to flatten and unflatten ODEState objects.
    flatten_fn = lambda ode_state: x0.flatten_fn()(ode_state)
    unflatten_fn = lambda x: x0.unflatten_fn()(x)

    # Select the target one-step integrator.
    def odeint_adaptive_one_step(x0, t0, tf, dt0):
        out = odeint_embedded_rk_manifold_one_step(
            f=f,
            x0=unflatten_fn(x0),
            t0=t0,
            tf=tf,
            integrator_type=integrator_type,
            physics_model=physics_model,
            dt0=dt0,
            dt_min=dt_min,
            dt_max=dt_max,
            rtol=rtol,
            atol=atol,
            safety=safety,
            beta_min=beta_min,
            beta_max=beta_max,
            semi_implicit_quaternion_integration=semi_implicit_quaternion_integration,
            debug_buffers_size=debug_buffers_size_per_step,
            max_step_rejections=max_step_rejections,
        )
        return flatten_fn(out[0]), out[1]

    # Integrate the state with an adaptive timestep over the horizon defined by `t`.
    X_flat, dict_aux = _ode_integration_adaptive_template(
        x0=flatten_fn(x0),
        t=t,
        odeint_adaptive_one_step=odeint_adaptive_one_step,
        dt0=dt0,
        debug_buffers_size_per_step=debug_buffers_size_per_step,
    )

    # Unflatten the integrated flat data.
    X = jax.vmap(unflatten_fn)(X_flat)

    # Unflatten the optional debug data included in dict_aux.
    dict_aux_unflattened = dict_aux | (
        dict()
        if "x_steps" not in dict_aux
        else dict(x_steps=jax.vmap(unflatten_fn)(dict_aux["x_steps"]))
    )

    # Return the unflattened output.
    return X, dict_aux_unflattened
