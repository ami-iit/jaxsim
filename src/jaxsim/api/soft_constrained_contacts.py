from __future__ import annotations

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.terrain import FlatTerrain, Terrain

from .contact import ContactModel, ContactParams, ContactsState
from .model import JaxSimModel


@jax_dataclasses.pytree_dataclass
class ConstrainedContactsParams(ContactParams):
    """Parameters of the constrained contacts model."""

    timeconst: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    dampratio: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    dmin: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )
    dmax: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )
    width: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    mid: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    power: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )

    @staticmethod
    def build(
        timeconst: float = 0.1,
        dampratio: float = 0.5,
        dmin: float = 0.0,
        dmax: float = 1.0,
        width: float = 0.1,
        mid: float = 0.5,
        power: float = 2.0,
    ) -> ConstrainedContactsParams:
        """
        Create a ConstrainedContactsParams instance with specified parameters.

        Args:
            timeconst (float, optional): The time constant. Defaults to 0.1.
            dampratio (float, optional): The damping ratio. Defaults to 0.5.
            dmin (float, optional): The minimum damping. Defaults to 0.0.
            dmax (float, optional): The maximum damping. Defaults to 1.0.
            width (float, optional): The width of the damping function. Defaults to 0.1.
            mid (float, optional): The mid value of the damping function. Defaults to 0.5.
            power (float, optional): The power of the damping function. Defaults to 2.0.

        Returns:
            ConstrainedContactsParams: A ConstrainedContactsParams instance with the specified parameters.
        """

        return ConstrainedContactsParams(
            timeconst=jnp.array(timeconst, dtype=float),
            dampratio=jnp.array(dampratio, dtype=float),
            dmin=jnp.array(dmin, dtype=float),
            dmax=jnp.array(dmax, dtype=float),
            width=jnp.array(width, dtype=float),
            mid=jnp.array(mid, dtype=float),
            power=jnp.array(power, dtype=float),
        )

    def __iter__(self):
        return iter(
            [
                self.timeconst,
                self.dampratio,
                self.dmin,
                self.dmax,
                self.width,
                self.mid,
                self.power,
            ]
        )


@jax_dataclasses.pytree_dataclass
class ConstrainedContacts(ContactModel):
    """Constrained contacts model."""

    parameters: ConstrainedContactsParams = jax_dataclasses.field(
        default_factory=ConstrainedContactsParams
    )

    terrain: Terrain = jax_dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        model: JaxSimModel,
        data: jtp.Vector,
        tau: jtp.Vector | None = None,
        tangential_deformation: jtp.Vector | None = None,
    ) -> tuple[jtp.Vector, jtp.Vector]:

        return jnp.zeros_like(tangential_deformation), None


@jax_dataclasses.pytree_dataclass
class ConstrainedContactsState(ContactsState):
    """
    Class storing the state of the constrained contacts model.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, **kwargs) -> ConstrainedContactsState:
        return ConstrainedContactsState()

    @staticmethod
    def build_from_jaxsim_model(
        model: JaxSimModel,
        **kwargs,
    ) -> ConstrainedContactsState:
        """
        Create a ConstrainedContactsState instance.

        Args:
            model: The target model.

        Returns:
            A `ConstrainedContactsState` instance with the specified parameters.
        """
        return ConstrainedContactsState.build()

    @staticmethod
    def zero(model: JaxSimModel) -> ConstrainedContactsState:
        return ConstrainedContactsState.build_from_jaxsim_model(model=model)
