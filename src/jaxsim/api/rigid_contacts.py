from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.terrain import FlatTerrain, Terrain

from .contact import ContactModel, ContactParams, ContactsState
from .model import JaxSimModel


@jax_dataclasses.pytree_dataclass
class RigidContactsParams(ContactParams):
    """
    Create a RigidContactsParams instance with specified parameters.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, *args, **kwargs) -> RigidContactsParams:
        return RigidContactsParams()

    @staticmethod
    def build_default_from_jaxsim_model(
        model: JaxSimModel,
        *args,
        **kwargs,
    ) -> RigidContactsParams:
        """
        Create a RigidContactsParams instance with good default parameters.

        Args:
            model: The target model.

        Returns:
            A `RigidContactsParams` instance with the specified parameters.
        """
        return RigidContactsParams.build()


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    parameters: RigidContactsParams = dataclasses.field(
        default_factory=RigidContactsParams
    )

    terrain: Terrain = dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        jdot_nu: jtp.Vector,
        j: jtp.Vector,
        nu: jtp.Vector,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces and material deformation rate.

        Args:
            position: The position of the collidable point.
            velocity: The linear velocity of the collidable point.

        Returns:
            A tuple containing the contact force and material deformation rate.
        """

        # Compute the contact force.
        contact_force = jnp.zeros_like(j)


@jax_dataclasses.pytree_dataclass
class RigidContactsState(ContactsState):
    """
    Class storing the state of the rigid contacts model.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, **kwargs) -> RigidContactsState:
        return RigidContactsState()

    @staticmethod
    def build_from_jaxsim_model(
        model: JaxSimModel,
        **kwargs,
    ) -> RigidContactsState:
        """
        Create a RigidContactsState instance with good default parameters.

        Args:
            model: The target model.

        Returns:
            A `RigidContactsState` instance with the specified parameters.
        """
        return RigidContactsState.build()

    @staticmethod
    def zero(model: JaxSimModel) -> RigidContactsState:
        return RigidContactsState.build_from_jaxsim_model(model=model)
