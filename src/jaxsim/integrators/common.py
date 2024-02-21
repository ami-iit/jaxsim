import abc
import dataclasses
from typing import Any, Callable, ClassVar, Generic, Self, Type, TypeVar

import jax
import jax_dataclasses
from jax_dataclasses import Static

from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass, Mutability

# =============
# Generic types
# =============

Time = jax.typing.ArrayLike
TimeStep = jax.typing.ArrayLike
State = NextState = TypeVar("State")

StateDerivative = TypeVar("StateDerivative")
SystemDynamics = Callable[[State, Time], tuple[StateDerivative, dict[str, Any]]]

# =======================
# Base integrator classes
# =======================


@jax_dataclasses.pytree_dataclass
class Integrator(JaxsimDataclass, abc.ABC, Generic[State]):

    AuxDictDynamicsKey: ClassVar[str] = "aux_dict_dynamics"

    dynamics: Static[SystemDynamics] = dataclasses.field(
        repr=False, hash=False, compare=False, kw_only=True
    )

    params: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, hash=False, compare=False, kw_only=True
    )

    @classmethod
    def build(cls: Type[Self], *, dynamics: SystemDynamics, **kwargs) -> Self:
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
