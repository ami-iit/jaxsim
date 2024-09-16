import abc
import dataclasses
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass, Mutability

try:
    from typing import override
except ImportError:
    from typing_extensions import override

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


# =============
# Generic types
# =============

Time = jtp.FloatLike
TimeStep = jtp.FloatLike
State = NextState = TypeVar("State")
StateDerivative = TypeVar("StateDerivative")
PyTreeType = TypeVar("PyTreeType", bound=jtp.PyTree)


class SystemDynamics(Protocol[State, StateDerivative]):
    def __call__(
        self, x: State, t: Time, **kwargs
    ) -> tuple[StateDerivative, dict[str, Any]]: ...


# =======================
# Base integrator classes
# =======================


@jax_dataclasses.pytree_dataclass
class Integrator(JaxsimDataclass, abc.ABC, Generic[State, StateDerivative]):

    AfterInitKey: ClassVar[str] = "after_init"
    InitializingKey: ClassVar[str] = "initializing"

    AuxDictDynamicsKey: ClassVar[str] = "aux_dict_dynamics"

    dynamics: Static[SystemDynamics[State, StateDerivative]] = dataclasses.field(
        repr=False, hash=False, compare=False, kw_only=True
    )

    params: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, hash=False, compare=False, kw_only=True
    )

    @classmethod
    def build(
        cls: type[Self],
        *,
        dynamics: SystemDynamics[State, StateDerivative],
        **kwargs,
    ) -> Self:
        """
        Build the integrator object.

        Args:
            dynamics: The system dynamics.
            **kwargs: Additional keyword arguments to build the integrator.

        Returns:
            The integrator object.
        """

        return cls(dynamics=dynamics, **kwargs)

    def step(
        self,
        x0: State,
        t0: Time,
        dt: TimeStep,
        *,
        params: dict[str, Any],
        **kwargs,
    ) -> tuple[State, dict[str, Any]]:
        """
        Perform a single integration step.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt: The time step of the integration.
            params: The auxiliary dictionary of the integrator.
            **kwargs: Additional keyword arguments.

        Returns:
            The final state of the system and the updated auxiliary dictionary.
        """

        with self.editable(validate=False) as integrator:
            integrator.params = params

        with integrator.mutable_context(mutability=Mutability.MUTABLE):
            xf, aux_dict = integrator(x0, t0, dt, **kwargs)

        return (
            xf,
            integrator.params
            | {Integrator.AfterInitKey: jnp.array(False).astype(bool)}
            | aux_dict,
        )

    @abc.abstractmethod
    def __call__(self, x0: State, t0: Time, dt: TimeStep, **kwargs) -> NextState:
        pass

    def init(
        self,
        x0: State,
        t0: Time,
        dt: TimeStep,
        *,
        include_dynamics_aux_dict: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Initialize the integrator.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt: The time step of the integration.

        Returns:
            The auxiliary dictionary of the integrator.

        Note:
            This method should have the same signature as the inherited `__call__`
            method, including additional kwargs.

        Note:
            If the integrator supports FSAL, the pair `(x0, t0)` must match the real
            initial state and time of the system, otherwise the initial derivative of
            the first step will be wrong.
        """

        with self.editable(validate=False) as integrator:

            # Initialize the integrator parameters.
            # For initialization purpose, the integrators can check if the
            # `Integrator.InitializingKey` is present in their parameters.
            # The AfterInitKey is used in the first step after initialization.
            integrator.params = {
                Integrator.InitializingKey: jnp.array(True),
                Integrator.AfterInitKey: jnp.array(False),
            }

            # Run a dummy call of the integrator.
            # It is used only to get the params so that we know the structure
            # of the corresponding pytree.
            _ = integrator(x0, t0, dt, **kwargs)

        # Remove the injected key.
        _ = integrator.params.pop(Integrator.InitializingKey)

        # Make sure that all leafs of the dictionary are JAX arrays.
        # Also, since these are dummy parameters, set them all to zero.
        params_after_init = jax.tree.map(lambda l: jnp.zeros_like(l), integrator.params)

        # Mark the next step as first step after initialization.
        params_after_init = params_after_init | {
            Integrator.AfterInitKey: jnp.array(True)
        }

        # Store the zero parameters in the integrator.
        # When the integrator is stepped, this is used to check if the passed
        # parameters are valid.
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self.params = params_after_init

        return params_after_init


@jax_dataclasses.pytree_dataclass
class ExplicitRungeKutta(Integrator[PyTreeType, PyTreeType], Generic[PyTreeType]):

    # The Runge-Kutta matrix.
    A: ClassVar[jtp.Matrix]

    # The weights coefficients.
    # Note that in practice we typically use its transpose `b.transpose()`.
    b: ClassVar[jtp.Matrix]

    # The nodes coefficients.
    c: ClassVar[jtp.Vector]

    # Define the order of the solution.
    # It should have as many elements as the number of rows of `b.transpose()`.
    order_of_bT_rows: ClassVar[tuple[int, ...]]

    # Define the row of the integration output corresponding to the final solution.
    # This is the row of b.T that produces the final state.
    row_index_of_solution: ClassVar[int]

    # Attributes of FSAL (first-same-as-last) property.
    fsal_enabled_if_supported: Static[bool] = dataclasses.field(repr=False)
    index_of_fsal: Static[jtp.IntLike | None] = dataclasses.field(repr=False)

    @property
    def has_fsal(self) -> bool:
        return self.fsal_enabled_if_supported and self.index_of_fsal is not None

    @property
    def order(self) -> int:
        return self.order_of_bT_rows[self.row_index_of_solution]

    @override
    @classmethod
    def build(
        cls: type[Self],
        *,
        dynamics: SystemDynamics[State, StateDerivative],
        fsal_enabled_if_supported: jtp.BoolLike = True,
        **kwargs,
    ) -> Self:
        """
        Build the integrator object.

        Args:
            dynamics: The system dynamics.
            fsal_enabled_if_supported:
                Whether to enable the FSAL property, if supported.
            **kwargs: Additional keyword arguments to build the integrator.

        Returns:
            The integrator object.
        """

        # Check validity of the Butcher tableau.
        if not ExplicitRungeKutta.butcher_tableau_is_valid(A=cls.A, b=cls.b, c=cls.c):
            raise ValueError("The Butcher tableau of this class is not valid.")

        # Check that b.T has enough rows based on the configured index of the solution.
        if cls.row_index_of_solution >= cls.b.T.shape[0]:
            msg = "The index of the solution ({}-th row of `b.T`) is out of range ({})."
            raise ValueError(msg.format(cls.row_index_of_solution, cls.b.T.shape[0]))

        # Check that the tuple containing the order of the b.T rows matches the number
        # of the b.T rows.
        if len(cls.order_of_bT_rows) != cls.b.T.shape[0]:
            msg = "Wrong size of 'order_of_bT_rows' ({}), should be {}."
            raise ValueError(msg.format(len(cls.order_of_bT_rows), cls.b.T.shape[0]))

        # Check if the Butcher tableau supports FSAL (first-same-as-last).
        # If it does, store the index of the intermediate derivative to be used as the
        # first derivative of the next iteration.
        has_fsal, index_of_fsal = (  # noqa: F841
            ExplicitRungeKutta.butcher_tableau_supports_fsal(
                A=cls.A, b=cls.b, c=cls.c, index_of_solution=cls.row_index_of_solution
            )
        )

        # Build the integrator object.
        integrator = super().build(
            dynamics=dynamics,
            index_of_fsal=index_of_fsal,
            fsal_enabled_if_supported=bool(fsal_enabled_if_supported),
            **kwargs,
        )

        return integrator

    def __call__(
        self, x0: State, t0: Time, dt: TimeStep, **kwargs
    ) -> tuple[NextState, dict[str, Any]]:

        # Here z is a batched state with as many batch elements as b.T rows.
        # Note that z has multiple batches only if b.T has more than one row,
        # e.g. in Butcher tableau of embedded schemes.
        z, aux_dict = self._compute_next_state(x0=x0, t0=t0, dt=dt, **kwargs)

        # The next state is the batch element located at the configured index of solution.
        next_state = jax.tree.map(lambda l: l[self.row_index_of_solution], z)

        return next_state, aux_dict

    @classmethod
    def integrate_rk_stage(
        cls, x0: State, t0: Time, dt: TimeStep, k: StateDerivative
    ) -> NextState:
        """
        Integrate a single stage of the Runge-Kutta method.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt:
                The time step of the RK integration scheme. Note that this is
                not the stage timestep, as it depends on the `A` matrix used
                to compute the `k` argument.
            k:
                The RK state derivative of the current stage, weighted with
                the `A` matrix.

        Returns:
            The state at the next stage of the integration.

        Note:
            In the most generic case, `k` could be an arbitrary composition
            of the kᵢ derivatives, depending on the RK matrix A.

        Note:
            Overriding this method allows users to use different classes
            defining `State` and `StateDerivative`. Be aware that the
            timestep `dt` is not the stage timestep, therefore the map
            used to convert the state derivative must be time-independent.
        """

        op = lambda x0_leaf, k_leaf: x0_leaf + dt * k_leaf
        return jax.tree.map(op, x0, k)

    @classmethod
    def post_process_state(
        cls, x0: State, t0: Time, xf: NextState, dt: TimeStep
    ) -> NextState:
        r"""
        Post-process the integrated state at :math:`t_f = t_0 + \Delta t`.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            xf: The final state of the system obtain through the integration.
            dt: The time step used for the integration.

        Returns:
            The post-processed integrated state.
        """

        return xf

    def _compute_next_state(
        self, x0: State, t0: Time, dt: TimeStep, **kwargs
    ) -> tuple[NextState, dict[str, Any]]:
        """
        Compute the next state of the system, returning all the output states.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt: The time step of the integration.
            **kwargs: Additional keyword arguments.

        Returns:
            A batched state with as many batch elements as `b.T` rows.
        """

        # Call variables with better symbols.
        Δt = dt
        c = self.c
        b = self.b
        A = self.A

        # Close f over optional kwargs.
        f = lambda x, t: self.dynamics(x=x, t=t, **kwargs)

        # Initialize the carry of the for loop with the stacked kᵢ vectors.
        carry0 = jax.tree.map(
            lambda l: jnp.repeat(jnp.zeros_like(l)[jnp.newaxis, ...], c.size, axis=0),
            x0,
        )

        # Apply FSAL property by passing ẋ0 = f(x0, t0) from the previous iteration.
        get_ẋ0_and_aux_dict = lambda: self.params.get("dxdt0", f(x0, t0))

        # We use a `jax.lax.scan` to compile the `f` function only once.
        # Otherwise, if we compute e.g. for RK4 sequentially, the jit-compiled code
        # would include 4 repetitions of the `f` logic, making everything extremely slow.
        def scan_body(
            carry: jax.Array, i: int | jax.Array
        ) -> tuple[jax.Array, dict[str, Any]]:
            """"""

            # Unpack the carry, i.e. the stacked kᵢ vectors.
            K = carry

            # Define the computation of the Runge-Kutta stage.
            def compute_ki() -> tuple[jax.Array, dict[str, Any]]:

                # Compute ∑ⱼ aᵢⱼ kⱼ.
                op_sum_ak = lambda k: jnp.einsum("s,s...->...", A[i], k)
                sum_ak = jax.tree.map(op_sum_ak, K)

                # Compute the next state for the kᵢ evaluation.
                # Note that this is not a Δt integration since aᵢⱼ could be fractional.
                xi = self.integrate_rk_stage(x0, t0, Δt, sum_ak)

                # Compute the next time for the kᵢ evaluation.
                ti = t0 + c[i] * Δt

                # This is kᵢ, aux_dict = f(xᵢ, tᵢ).
                return f(xi, ti)

            # This selector enables FSAL property in the first iteration (i=0).
            ki, aux_dict = jax.lax.cond(
                pred=jnp.logical_and(i == 0, self.has_fsal),
                true_fun=get_ẋ0_and_aux_dict,
                false_fun=compute_ki,
            )

            # Store the kᵢ derivative in K.
            op = lambda l_k, l_ki: l_k.at[i].set(l_ki)
            K = jax.tree.map(op, K, ki)

            carry = K
            return carry, aux_dict

        # Compute the state derivatives kᵢ.
        K, aux_dict = jax.lax.scan(
            f=scan_body,
            init=carry0,
            xs=jnp.arange(c.size),
        )

        # Update the FSAL property for the next iteration.
        if self.has_fsal:
            self.params["dxdt0"] = jax.tree.map(lambda l: l[self.index_of_fsal], K)

        # Compute the output state.
        # Note that z contains as many new states as the rows of `b.T`.
        op = lambda x0, k: x0 + Δt * jnp.einsum("zs,s...->z...", b.T, k)
        z = jax.tree.map(op, x0, K)

        # Transform the final state of the integration.
        # This allows to inject custom logic, if needed.
        z_transformed = jax.vmap(
            lambda xf: self.post_process_state(x0=x0, t0=t0, xf=xf, dt=dt)
        )(z)

        return z_transformed, aux_dict

    @staticmethod
    def butcher_tableau_is_valid(
        A: jtp.Matrix, b: jtp.Matrix, c: jtp.Vector
    ) -> jtp.Bool:
        """
        Check if the Butcher tableau is valid.

        Args:
            A: The Runge-Kutta matrix.
            b: The weights coefficients.
            c: The nodes coefficients.

        Returns:
            `True` if the Butcher tableau is valid, `False` otherwise.
        """

        valid = True
        valid = valid and A.ndim == 2
        valid = valid and b.ndim == 2
        valid = valid and c.ndim == 1
        valid = valid and b.T.shape[0] <= 2
        valid = valid and A.shape[0] == A.shape[1]
        valid = valid and A.shape == (c.size, b.T.shape[1])
        valid = valid and bool(jnp.all(b.T.sum(axis=1) == 1))

        return valid

    @staticmethod
    def butcher_tableau_is_explicit(A: jtp.Matrix) -> jtp.Bool:
        """
        Check if the Butcher tableau corresponds to an explicit integration scheme.

        Args:
            A: The Runge-Kutta matrix.

        Returns:
            `True` if the Butcher tableau is explicit, `False` otherwise.
        """

        return jnp.allclose(A, jnp.tril(A, k=-1))

    @staticmethod
    def butcher_tableau_supports_fsal(
        A: jtp.Matrix,
        b: jtp.Matrix,
        c: jtp.Vector,
        index_of_solution: jtp.IntLike = 0,
    ) -> tuple[bool, int | None]:
        """
        Check if the Butcher tableau supports the FSAL (first-same-as-last) property.

        Args:
            A: The Runge-Kutta matrix.
            b: The weights coefficients.
            c: The nodes coefficients.
            index_of_solution:
                The index of the row of `b.T` corresponding to the solution.

        Returns:
            A tuple containing a boolean indicating whether the Butcher tableau supports
            FSAL, and the index i of the intermediate kᵢ derivative corresponding to the
            initial derivative `f(x0, t0)` of the next step.
        """

        if not ExplicitRungeKutta.butcher_tableau_is_valid(A=A, b=b, c=c):
            raise ValueError("The Butcher tableau is not valid.")

        if not ExplicitRungeKutta.butcher_tableau_is_explicit(A=A):
            return False

        if index_of_solution >= b.T.shape[0]:
            msg = "The index of the solution (i-th row of `b.T`) is out of range."
            raise ValueError(msg)

        if c[0] != 0:
            return False, None

        # Find all the rows of A where c = 1 (therefore at t=tf). The Butcher tableau
        # supports FSAL if any of these rows (there might be more rows with c=1) matches
        # the rows of b.T corresponding to the next state (marked by `index_of_solution`).
        # This last condition means that the last kᵢ derivative is computed at (tf, xf),
        # that corresponds to the (t0, x0) pair of the next integration call.
        rows_of_A_with_fsal = (A == b.T[None, index_of_solution]).all(axis=1)
        rows_of_A_with_fsal = jnp.logical_and(rows_of_A_with_fsal, (c == 1))

        # If there is no match, it means that the Butcher tableau does not support FSAL.
        if not rows_of_A_with_fsal.any():
            return False, None

        # Return the index of the row of A providing the fsal derivative (that is the
        # possibly intermediate kᵢ derivative).
        # Note that if multiple rows match (it should not), we return the first match.
        return True, int(jnp.where(rows_of_A_with_fsal)[0].tolist()[0])


class ExplicitRungeKuttaSO3Mixin:
    """
    Mixin class to apply over explicit RK integrators defined on
    `PyTreeType = ODEState` to integrate the quaternion on SO(3).
    """

    @classmethod
    def post_process_state(
        cls, x0: js.ode_data.ODEState, t0: Time, xf: js.ode_data.ODEState, dt: TimeStep
    ) -> js.ode_data.ODEState:

        # Extract the initial base quaternion.
        W_Q_B_t0 = x0.physics_model.base_quaternion

        # We assume that the initial quaternion is already unary.
        exceptions.raise_runtime_error_if(
            condition=jnp.logical_not(jnp.allclose(W_Q_B_t0.dot(W_Q_B_t0), 1.0)),
            msg="The SO(3) integrator received a quaternion at t0 that is not unary.",
        )

        # Get the angular velocity ω to integrate the quaternion.
        # This velocity ω[t0] is computed in the previous timestep by averaging the kᵢ
        # corresponding to the active RK-based scheme. Therefore, by using the ω[t0],
        # we obtain an explicit RK scheme operating on the SO(3) manifold.
        # Note that the current integrator is not a semi-implicit scheme, therefore
        # using the final ω[tf] would be not correct.
        W_ω_WB_t0 = x0.physics_model.base_angular_velocity

        # Integrate the quaternion on SO(3).
        W_Q_B_tf = jaxsim.math.Quaternion.integration(
            quaternion=W_Q_B_t0,
            dt=dt,
            omega=W_ω_WB_t0,
            omega_in_body_fixed=False,
        )

        # Replace the quaternion in the final state.
        return xf.replace(
            physics_model=xf.physics_model.replace(base_quaternion=W_Q_B_tf),
            validate=True,
        )
