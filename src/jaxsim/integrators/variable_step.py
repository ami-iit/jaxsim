import functools
from typing import Any, ClassVar, Generic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.utils.tracing
from jaxsim import typing as jtp

from .common import (
    ExplicitRungeKutta,
    ExplicitRungeKuttaSO3Mixin,
    NextState,
    PyTreeType,
    State,
    StateDerivative,
    SystemDynamics,
    Time,
    TimeStep,
)

# For robot dynamics, the following default tolerances are already pretty accurate.
# Users can either decrease them and pay the price of smaller Δt, or increase
# them and pay the price of less accurate dynamics.
RTOL_DEFAULT = 0.000_100  # 0.01%
ATOL_DEFAULT = 0.000_010  # 10μ

# Default parameters of Embedded RK schemes.
SAFETY_DEFAULT = 0.9
BETA_MIN_DEFAULT = 1.0 / 10
BETA_MAX_DEFAULT = 2.5
MAX_STEP_REJECTIONS_DEFAULT = 5


# =================
# Utility functions
# =================


@functools.partial(jax.jit, static_argnames=["f"])
def estimate_step_size(
    x0: jtp.PyTree,
    t0: Time,
    f: SystemDynamics,
    order: jtp.IntLike,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
) -> tuple[jtp.Float, jtp.PyTree]:
    r"""
    Compute the initial step size to warm-start variable-step integrators.

    Args:
        x0: The initial state.
        t0: The initial time.
        f: The state derivative function :math:`f(x, t)`.
        order:
            The order :math:`p` of an integrator with truncation error
            :math:`\mathcal{O}(\Delta t^{p+1})`.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        A tuple containing the computed initial step size
        and the state derivative :math:`\dot{x} = f(x_0, t_0)`.

    Note:
        Interested readers could find implementation details in:

        Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        E. Hairer, S. P. Norsett G. Wanner.
    """

    # Helper to flatten a pytree to a 1D vector.
    def flatten(pytree) -> jax.Array:
        return jax.flatten_util.ravel_pytree(pytree=pytree)[0]

    # Compute the state derivative at the initial state.
    ẋ0 = f(x0, t0)[0]

    # Compute the scaling factors of the initial state and its derivative.
    compute_scale = lambda x: atol + jnp.abs(x) * rtol
    scale0 = jax.tree.map(compute_scale, x0)
    scale1 = jax.tree.map(compute_scale, ẋ0)

    # Scale the initial state and its derivative.
    scale_pytree = lambda x, scale: jnp.abs(x) / scale
    x0_scaled = jax.tree.map(scale_pytree, x0, scale0)
    ẋ0_scaled = jax.tree.map(scale_pytree, ẋ0, scale1)

    # Get the maximum of the scaled pytrees.
    d0 = jnp.linalg.norm(flatten(x0_scaled), ord=jnp.inf)
    d1 = jnp.linalg.norm(flatten(ẋ0_scaled), ord=jnp.inf)

    # Compute the first guess of the initial step size.
    h0 = jnp.where(jnp.minimum(d0, d1) <= 1e-5, 1e-6, 0.01 * d0 / d1)

    # Compute the next state (explicit Euler step) and its derivative.
    x1 = jax.tree.map(lambda x0, ẋ0: x0 + h0 * ẋ0, x0, ẋ0)
    ẋ1 = f(x1, t0 + h0)[0]

    # Compute the scaling factor of the state derivatives.
    compute_scale_2 = lambda ẋ0, ẋ1: atol + jnp.maximum(jnp.abs(ẋ0), jnp.abs(ẋ1)) * rtol
    scale2 = jax.tree.map(compute_scale_2, ẋ0, ẋ1)

    # Scale the difference of the state derivatives.
    scale_ẋ_difference = lambda ẋ0, ẋ1, scale: jnp.abs((ẋ0 - ẋ1) / scale)
    ẋ_difference_scaled = jax.tree.map(scale_ẋ_difference, ẋ0, ẋ1, scale2)

    # Get the maximum of the scaled derivatives difference.
    d2 = jnp.linalg.norm(flatten(ẋ_difference_scaled), ord=jnp.inf) / h0

    # Compute the second guess of the initial step size.
    h1 = jnp.where(
        jnp.maximum(d1, d2) <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    # Propose the final guess of the initial step size.
    # Also return the state derivative computed at the initial state since
    # likely it is a quantity that needs to be computed again later.
    return jnp.array(jnp.minimum(100.0 * h0, h1), dtype=float), ẋ0


@jax.jit
def compute_pytree_scale(
    x1: jtp.PyTree,
    x2: jtp.PyTree | None = None,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
) -> jtp.PyTree:
    """
    Compute the component-wise state scale factors to scale dynamical states.

    Args:
        x1: The first state (often the initial state).
        x2: The optional second state (often the final state).
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        A pytree with the same structure of the state containing the scaling factors.
    """

    # Consider a zero second pytree, if not given.
    x2 = jax.tree.map(jnp.zeros_like, x1) if x2 is None else x2

    # Compute the scaling factors of the initial state and its derivative.
    compute_scale = lambda l1, l2: atol + jnp.maximum(jnp.abs(l1), jnp.abs(l2)) * rtol
    scale = jax.tree.map(compute_scale, x1, x2)

    return scale


@jax.jit
def local_error_estimation(
    xf: jtp.PyTree,
    xf_estimate: jtp.PyTree | None = None,
    x0: jtp.PyTree | None = None,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    norm_ord: jtp.IntLike | jtp.FloatLike = jnp.inf,
) -> jtp.Float:
    """
    Estimate the local integration error, often used in Embedded RK schemes.

    Args:
        xf: The final state, often computed with the most accurate integrator.
        xf_estimate:
            The estimated final state, often computed with the less accurate integrator.
            If missing, it is initialized to zero.
        x0:
            The initial state to compute the scaling factors. If missing, it is
            initialized to zero.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.
        norm_ord:
            The norm to use to compute the error. Default is the infinity norm.

    Returns:
        The estimated local integration error.
    """

    # Helper to flatten a pytree to a 1D vector.
    def flatten(pytree) -> jax.Array:
        return jax.flatten_util.ravel_pytree(pytree=pytree)[0]

    # Compute the scale considering the initial and final states.
    scale = compute_pytree_scale(x1=xf, x2=x0, rtol=rtol, atol=atol)

    # Consider a zero estimated final state, if not given.
    xf_estimate = (
        jax.tree.map(jnp.zeros_like, xf) if xf_estimate is None else xf_estimate
    )

    # Estimate the error.
    estimate_error = lambda l, l̂, sc: jnp.abs(l - l̂) / sc
    error_estimate = jax.tree.map(estimate_error, xf, xf_estimate, scale)

    # Return the highest element of the error estimate.
    return jnp.linalg.norm(flatten(error_estimate), ord=norm_ord)


# ================================
# Embedded Runge-Kutta integrators
# ================================


@jax_dataclasses.pytree_dataclass
class EmbeddedRungeKutta(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):
    """
    An Embedded Runge-Kutta integrator.

    This class implements a general-purpose Embedded Runge-Kutta integrator
    that can be used to solve ordinary differential equations with adaptive
    step sizes.

    The integrator is based on an Explicit Runge-Kutta method, and it uses
    two different solutions to estimate the local integration error. The
    error is then used to adapt the step size to reach a desired accuracy.
    """

    AfterInitKey: ClassVar[str] = "after_init"
    InitializingKey: ClassVar[str] = "initializing"

    # Define the row of the integration output corresponding to the solution estimate.
    # This is the row of b.T that produces the state used e.g. by embedded methods to
    # implement the adaptive timestep logic.
    row_index_of_solution_estimate: ClassVar[int | None] = None

    # Bounds of the adaptive Δt.
    dt_max: Static[jtp.FloatLike] = jnp.inf
    dt_min: Static[jtp.FloatLike] = -jnp.inf

    # Tolerances used to scale the two states corresponding to the high-order solution
    # and the low-order estimate during the computation of the local integration error.
    rtol: Static[jtp.FloatLike] = RTOL_DEFAULT
    atol: Static[jtp.FloatLike] = ATOL_DEFAULT

    # Parameters of the adaptive timestep logic.
    # Refer to Eq. (4.13) pag. 168 of Hairer93.
    safety: Static[jtp.FloatLike] = SAFETY_DEFAULT
    beta_max: Static[jtp.FloatLike] = BETA_MAX_DEFAULT
    beta_min: Static[jtp.FloatLike] = BETA_MIN_DEFAULT

    # Maximum number of rejected steps when the Δt needs to be reduced.
    max_step_rejections: Static[jtp.IntLike] = MAX_STEP_REJECTIONS_DEFAULT

    def init(
        self,
        x0: State,
        t0: Time,
        dt: TimeStep,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Initialize the integrator and get the metadata.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt: The time step of the integration.
            **kwargs: Additional parameters.

        Returns:
            The metadata of the integrator to be passed to the first step.
        """

        if jaxsim.utils.tracing(var=jnp.zeros(0)):
            raise RuntimeError("This method cannot be used within a JIT context")

        with self.editable(validate=False) as integrator:

            # Inject this key to signal that the integrator is initializing.
            # This is used to allocate the arrays of the metadata dictionary,
            # that are then filled with NaNs.
            metadata = {EmbeddedRungeKutta.InitializingKey: jnp.array(True)}

            # Run a dummy call of the integrator.
            # It is used only to get the metadata so that we know the structure
            # of the corresponding pytree.
            _ = integrator(
                x0,
                jnp.array(t0, dtype=float),
                jnp.array(dt, dtype=float),
                **(kwargs | {"metadata": metadata}),
            )

        # Remove the injected key.
        _ = metadata.pop(EmbeddedRungeKutta.InitializingKey)

        # Make sure that all leafs of the dictionary are JAX arrays.
        # Also, since these are dummy parameters, set them all to NaN.
        metadata_after_init = jax.tree.map(
            lambda l: jnp.nan * jnp.zeros_like(l), metadata
        )

        return metadata_after_init

    def __call__(
        self, x0: State, t0: Time, dt: TimeStep, **kwargs
    ) -> tuple[NextState, dict[str, Any]]:
        """
        Integrate the system for a single step.
        """

        # This method is called differently in three stages:
        #
        # 1. During initialization, to allocate a dummy metadata dictionary.
        #    The metadata is a dictionary of float JAX arrays, that are initialized
        #    with the right shape and filled with NaNs.
        # 2. During the first step, this method operates on the Nan-filled
        #    `metadata` argument, and it populates with the actual metadata.
        # 3. After the first step, this method operates on the actual metadata.
        #
        # In particular, we store the following information in the metadata:
        # - The first attempt of the step size, `dt0`. This is either estimated during
        #   phase 2, or taken from the previous step during phase 3.
        # - For integrators that support FSAL, the derivative at the initial state
        #   computed during the previous step. This can be done because FSAL integrators
        #   evaluate the dynamics at the final state of the previous step, that matches
        #   the initial state of the current step.
        #
        metadata = kwargs.pop("metadata", {})

        integrator_init = jnp.array(
            metadata.get(self.InitializingKey, False), dtype=bool
        )

        # Close f over optional kwargs.
        f = lambda x, t: self.dynamics(x=x, t=t, **kwargs)

        # Define the final time.
        tf = t0 + dt

        # Initialize solution orders.
        p = self.order_of_solution
        p̂ = self.order_of_solution_estimate
        q = jnp.minimum(p, p̂)

        # The value of dt0 is NaN (or, at least, it should be) only after initialization
        # and before the first step.
        metadata["dt0"], metadata["dxdt0"] = jax.lax.cond(
            pred=("dt0" in metadata) & ~jnp.isnan(metadata.get("dt0", 0.0)).any(),
            true_fun=lambda metadata: (
                metadata.get("dt0", jnp.array(0.0, dtype=float)),
                metadata.get("dxdt0", f(x0, t0)[0]),
            ),
            false_fun=lambda aux: estimate_step_size(
                x0=x0, t0=t0, f=f, order=p, atol=self.atol, rtol=self.rtol
            ),
            operand=metadata,
        )

        # Clip the estimated initial step size to the given bounds, if necessary.
        metadata["dt0"] = jnp.clip(
            metadata["dt0"],
            jnp.minimum(self.dt_min, metadata["dt0"]),
            jnp.minimum(self.dt_max, metadata["dt0"]),
        )

        # =========================================================
        # While loop to reach tf from t0 using an adaptive timestep
        # =========================================================

        # Initialize the carry of the while loop.
        Carry = tuple[Any, ...]
        carry0: Carry = (
            x0,
            jnp.array(t0).astype(float),
            metadata,
            jnp.array(0, dtype=int),
            jnp.array(False).astype(bool),
        )

        def while_loop_cond(carry: Carry) -> jtp.Bool:
            _, _, _, _, break_loop = carry
            return jnp.logical_not(break_loop)

        # Each loop is an integration step with variable Δt.
        # Depending on the integration error, the step could be discarded and the
        # while body ran again from the same (x0, t0) but with a smaller Δt.
        # We run these loops until the final time tf is reached.
        def while_loop_body(carry: Carry) -> Carry:

            # Unpack the carry.
            x0, t0, metadata, discarded_steps, _ = carry

            # Take care of the final adaptive step.
            # We want the final Δt to let us reach tf exactly.
            # Then we can exit the while loop.
            Δt0 = metadata["dt0"]
            Δt0 = jnp.where(t0 + Δt0 < tf, Δt0, tf - t0)
            break_loop = jnp.where(t0 + Δt0 < tf, False, True)

            # Run the underlying explicit RK integrator.
            # The output z contains multiple solutions (depending on the rows of b.T).
            with self.editable(validate=True) as integrator:
                z, aux_dict = integrator._compute_next_state(
                    x0=x0, t0=t0, dt=Δt0, **kwargs
                )
                metadata_next = aux_dict["metadata"]

            # Extract the high-order solution xf and the low-order estimate x̂f.
            xf = jax.tree.map(lambda l: l[self.row_index_of_solution], z)
            x̂f = jax.tree.map(lambda l: l[self.row_index_of_solution_estimate], z)

            # Calculate the local integration error.
            local_error = local_error_estimation(
                x0=x0, xf=xf, xf_estimate=x̂f, rtol=self.rtol, atol=self.atol
            )

            # Shrink the Δt every time by the safety factor (even when accepted).
            # The β parameters define the bounds of the timestep update factor.
            safety = jnp.clip(self.safety, 0.0, 1.0)
            β_min = jnp.maximum(0.0, self.beta_min)
            β_max = jnp.maximum(β_min, self.beta_max)

            # Compute the next Δt from the desired integration error.
            # The computed integration step is accepted if error <= 1.0,
            # otherwise it is rejected.
            #
            # In case of rejection, Δt_next is always smaller than Δt0.
            # In case of acceptance, Δt_next could either be larger than Δt0,
            # or slightly smaller than Δt0 depending on the safety factor.
            Δt_next = Δt0 * jnp.clip(
                safety * jnp.power(1 / local_error, 1 / (q + 1)),
                β_min,
                β_max,
            )

            def accept_step():
                # Use Δt_next in the next while loop.
                # If it is the last one, and Δt0 was clipped, return the initial Δt0.
                metadata_next_accepted = metadata_next | dict(
                    dt0=jnp.clip(
                        jax.lax.select(
                            pred=break_loop,
                            on_true=metadata["dt0"],
                            on_false=Δt_next,
                        ),
                        self.dt_min,
                        self.dt_max,
                    )
                )

                # Start the next while loop from the final state.
                x0_next = xf

                # Advance the starting time of the next adaptive step.
                t0_next = t0 + Δt0

                # Signal that the final time has been reached.
                break_loop_next = t0 + Δt0 >= tf

                return (
                    x0_next,
                    t0_next,
                    break_loop_next,
                    metadata_next_accepted,
                    jnp.array(0, dtype=int),
                )

            def reject_step():
                # Get back the original metadata.
                metadata_next_rejected = metadata

                # This time, with a reduced Δt.
                metadata_next_rejected["dt0"] = jnp.clip(
                    Δt_next, self.dt_min, self.dt_max
                )

                return (
                    x0,
                    t0,
                    False,
                    metadata_next_rejected,
                    discarded_steps + 1,
                )

            # Decide whether to accept or reject the step.
            (
                x0_next,
                t0_next,
                break_loop,
                metadata_next,
                discarded_steps,
            ) = jax.lax.cond(
                pred=(discarded_steps >= self.max_step_rejections)
                | (local_error <= 1.0)
                | (Δt_next < self.dt_min)
                | integrator_init,
                true_fun=accept_step,
                false_fun=reject_step,
            )

            return (
                x0_next,
                t0_next,
                metadata_next,
                discarded_steps,
                break_loop,
            )

        # Integrate with adaptive step until tf is reached.
        (
            xf,
            tf,
            metadata_tf,
            _,
            _,
        ) = jax.lax.while_loop(
            cond_fun=while_loop_cond,
            body_fun=while_loop_body,
            init_val=carry0,
        )

        return xf, {"metadata": metadata_tf}

    @property
    def order_of_solution(self) -> int:
        """
        The order of the solution.
        """
        return self.order_of_bT_rows[self.row_index_of_solution]

    @property
    def order_of_solution_estimate(self) -> int:
        """
        The order of the solution estimate.
        """
        return self.order_of_bT_rows[self.row_index_of_solution_estimate]

    @classmethod
    def build(
        cls: type[Self],
        *,
        dynamics: SystemDynamics[State, StateDerivative],
        fsal_enabled_if_supported: jtp.BoolLike = True,
        dt_max: jtp.FloatLike = jnp.inf,
        dt_min: jtp.FloatLike = -jnp.inf,
        rtol: jtp.FloatLike = RTOL_DEFAULT,
        atol: jtp.FloatLike = ATOL_DEFAULT,
        safety: jtp.FloatLike = SAFETY_DEFAULT,
        beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
        beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
        max_step_rejections: jtp.IntLike = MAX_STEP_REJECTIONS_DEFAULT,
        **kwargs,
    ) -> Self:
        """
        Build an Embedded Runge-Kutta integrator.

        Args:
            dynamics: The system dynamics function.
            fsal_enabled_if_supported:
                Whether to enable the FSAL property if supported by the integrator.
            dt_max: The maximum step size.
            dt_min: The minimum step size.
            rtol: The relative tolerance.
            atol: The absolute tolerance.
            safety: The safety factor to shrink the step size.
            beta_max: The maximum factor to increase the step size.
            beta_min: The minimum factor to increase the step size.
            max_step_rejections: The maximum number of step rejections.
            **kwargs: Additional parameters.
        """

        # Check that b.T has enough rows based on the configured index of the
        # solution estimate. This is necessary for embedded methods.
        if (
            cls.row_index_of_solution_estimate is not None
            and cls.row_index_of_solution_estimate >= cls.b.T.shape[0]
        ):
            msg = "The index of the solution estimate ({}-th row of `b.T`) "
            msg += "is out of range ({})."
            raise ValueError(
                msg.format(cls.row_index_of_solution_estimate, cls.b.T.shape[0])
            )

        integrator = super().build(
            # Integrator:
            dynamics=dynamics,
            # ExplicitRungeKutta:
            fsal_enabled_if_supported=bool(fsal_enabled_if_supported),
            # EmbeddedRungeKutta:
            dt_max=float(dt_max),
            dt_min=float(dt_min),
            rtol=float(rtol),
            atol=float(atol),
            safety=float(safety),
            beta_max=float(beta_max),
            beta_min=float(beta_min),
            max_step_rejections=int(max_step_rejections),
            **kwargs,
        )

        return integrator


@jax_dataclasses.pytree_dataclass
class HeunEulerSO3(EmbeddedRungeKutta[PyTreeType], ExplicitRungeKuttaSO3Mixin):
    """
    The Heun-Euler integrator for SO(3) dynamics.
    """

    A: ClassVar[jtp.Matrix] = jnp.array(
        [
            [0, 0],
            [1, 0],
        ]
    ).astype(float)

    b: ClassVar[jtp.Matrix] = (
        jnp.atleast_2d(
            jnp.array(
                [
                    [1 / 2, 1 / 2],
                    [1, 0],
                ]
            ),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jtp.Vector] = jnp.array(
        [0, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    row_index_of_solution_estimate: ClassVar[int | None] = 1

    order_of_bT_rows: ClassVar[tuple[int, ...]] = (2, 1)


@jax_dataclasses.pytree_dataclass
class BogackiShampineSO3(EmbeddedRungeKutta[PyTreeType], ExplicitRungeKuttaSO3Mixin):
    """
    The Bogacki-Shampine integrator for SO(3) dynamics.
    """

    A: ClassVar[jtp.Matrix] = jnp.array(
        [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 3 / 4, 0, 0],
            [2 / 9, 1 / 3, 4 / 9, 0],
        ]
    ).astype(float)

    b: ClassVar[jtp.Matrix] = (
        jnp.atleast_2d(
            jnp.array(
                [
                    [2 / 9, 1 / 3, 4 / 9, 0],
                    [7 / 24, 1 / 4, 1 / 3, 1 / 8],
                ]
            ),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jtp.Vector] = jnp.array(
        [0, 1 / 2, 3 / 4, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    row_index_of_solution_estimate: ClassVar[int | None] = 1

    order_of_bT_rows: ClassVar[tuple[int, ...]] = (3, 2)
