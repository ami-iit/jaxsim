import abc
import dataclasses
from typing import Any, ClassVar, Generic, Protocol, Self, Type, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass, Mutability

try:
    from typing import override
except ImportError:
    from typing_extensions import override


# =============
# Generic types
# =============

Time = jax.typing.ArrayLike
TimeStep = jax.typing.ArrayLike
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

    AuxDictDynamicsKey: ClassVar[str] = "aux_dict_dynamics"

    dynamics: Static[SystemDynamics[State, StateDerivative]] = dataclasses.field(
        repr=False, hash=False, compare=False, kw_only=True
    )

    params: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, hash=False, compare=False, kw_only=True
    )

    @classmethod
    def build(
        cls: Type[Self], *, dynamics: SystemDynamics[State, StateDerivative], **kwargs
    ) -> Self:
        """
        Build the integrator object.

        Args:
            dynamics: The system dynamics.
            **kwargs: Additional keyword arguments to build the integrator.

        Returns:
            The integrator object.
        """

        return cls(dynamics=dynamics, **kwargs)  # noqa

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
            xf = integrator(x0, t0, dt, **kwargs)

        assert Integrator.AuxDictDynamicsKey in integrator.params

        return xf, integrator.params

    @abc.abstractmethod
    def __call__(self, x0: State, t0: Time, dt: TimeStep, **kwargs) -> NextState:
        pass

    def init(
        self,
        x0: State,
        t0: Time,
        dt: TimeStep,
        *,
        key: jax.Array | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Initialize the integrator.

        Args:
            x0: The initial state of the system.
            t0: The initial time of the system.
            dt: The time step of the integration.
            key: An optional random key to initialize the integrator.

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

        _, aux_dict_dynamics = self.dynamics(x0, t0)

        with self.editable(validate=False) as integrator:
            _ = integrator(x0, t0, dt, **kwargs)
            aux_dict_step = integrator.params

        if Integrator.AuxDictDynamicsKey in aux_dict_dynamics:
            msg = "You cannot create a key '{}' in the __call__ method."
            raise KeyError(msg.format(Integrator.AuxDictDynamicsKey))

        return {Integrator.AuxDictDynamicsKey: aux_dict_dynamics} | aux_dict_step


@jax_dataclasses.pytree_dataclass
class ExplicitRungeKutta(Integrator[PyTreeType, PyTreeType], Generic[PyTreeType]):

    # The Runge-Kutta matrix.
    A: ClassVar[jax.typing.ArrayLike]

    # The weights coefficients.
    # Note that in practice we typically use its transpose `b.transpose()`.
    b: ClassVar[jax.typing.ArrayLike]

    # The nodes coefficients.
    c: ClassVar[jax.typing.ArrayLike]

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
        cls: Type[Self],
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

        # Adjust the shape of the tableau coefficients.
        c = jnp.atleast_1d(cls.c.squeeze())
        b = jnp.atleast_2d(jnp.vstack(cls.b.squeeze()))
        A = jnp.atleast_2d(cls.A.squeeze())

        # Check validity of the Butcher tableau.
        if not ExplicitRungeKutta.butcher_tableau_is_valid(A=A, b=b, c=c):
            raise ValueError("The Butcher tableau of this class is not valid.")

        # Store the adjusted shapes of the tableau coefficients.
        cls.c = c
        cls.b = b
        cls.A = A

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
        has_fsal, index_of_fsal = ExplicitRungeKutta.butcher_tableau_supports_fsal(
            A=cls.A, b=cls.b, c=cls.c, index_of_solution=cls.row_index_of_solution
        )

        # Build the integrator object.
        integrator = super().build(
            dynamics=dynamics,
            index_of_fsal=index_of_fsal,
            fsal_enabled_if_supported=bool(fsal_enabled_if_supported),
            **kwargs,
        )

        return integrator

    def __call__(self, x0: State, t0: Time, dt: TimeStep, **kwargs) -> NextState:

        # Here z is a batched state with as many batch elements as b.T rows.
        # Note that z has multiple batches only if b.T has more than one row,
        # e.g. in Butcher tableau of embedded schemes.
        z = self._compute_next_state(x0=x0, t0=t0, dt=dt, **kwargs)

        # The next state is the batch element located at the configured index of solution.
        return jax.tree_util.tree_map(lambda l: l[self.row_index_of_solution], z)

    def _compute_next_state(
        self, x0: State, t0: Time, dt: TimeStep, **kwargs
    ) -> NextState:
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
        carry0 = jax.tree_util.tree_map(
            lambda l: jnp.repeat(jnp.zeros_like(l)[jnp.newaxis, ...], c.size, axis=0),
            x0,
        )

        # Apply FSAL property by passing ẋ0 = f(x0, t0) from the previous iteration.
        get_ẋ0 = lambda: self.params.get("dxdt0", f(x0, t0)[0])

        # We use a `jax.lax.scan` to compile the `f` function only once.
        # Otherwise, if we compute e.g. for RK4 sequentially, the jit-compiled code
        # would include 4 repetitions of the `f` logic, making everything extremely slow.
        def scan_body(carry: jax.Array, i: int | jax.Array) -> tuple[jax.Array, None]:
            """"""

            # Unpack the carry, i.e. the stacked kᵢ vectors.
            K = carry

            # Define the computation of the Runge-Kutta stage.
            def compute_ki() -> jax.Array:
                ti = t0 + c[i] * Δt
                op = lambda x0, k: x0 + Δt * jnp.dot(A[i, :], k)
                xi = jax.tree_util.tree_map(op, x0, K)
                return f(xi, ti)[0]

            # This selector enables FSAL property in the first iteration (i=0).
            ki = jax.lax.cond(
                pred=jnp.logical_and(i == 0, self.has_fsal),
                true_fun=get_ẋ0,
                false_fun=compute_ki,
            )

            # Store the kᵢ derivative in K.
            op = lambda l_k, l_ki: l_k.at[i].set(l_ki)
            K = jax.tree_util.tree_map(op, K, ki)

            carry = K
            return carry, None

        # Compute the state derivatives kᵢ.
        K, _ = jax.lax.scan(
            f=scan_body,
            init=carry0,
            xs=jnp.arange(c.size),
        )

        # Update the FSAL property for the next iteration.
        if self.has_fsal:
            self.params["dxdt0"] = jax.tree_map(lambda l: l[self.index_of_fsal], K)

        # Compute the output state.
        # Note that z contains as many new states as the rows of `b.T`.
        op = lambda x0, ki: x0 + Δt * jnp.dot(b.T, ki)
        z = jax.tree_util.tree_map(op, x0, K)

        return z

    @staticmethod
    def butcher_tableau_is_valid(
        A: jax.typing.ArrayLike, b: jax.typing.ArrayLike, c: jax.typing.ArrayLike
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
    def butcher_tableau_is_explicit(A: jax.typing.ArrayLike) -> jtp.Bool:
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
        A: jax.typing.ArrayLike,
        b: jax.typing.ArrayLike,
        c: jax.typing.ArrayLike,
        index_of_solution: jtp.IntLike = 0,
    ) -> [bool, int | None]:
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
        return True, int(jnp.where(rows_of_A_with_fsal == True)[0].tolist()[0])
