from __future__ import annotations

import abc
import dataclasses

import jax_dataclasses

import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class ContactsState(JaxsimDataclass, abc.ABC):
    """
    Abstract class storing the state of the contacts model.
    """

    @classmethod
    def build(cls, **kwargs) -> ContactsState:
        """
        Build the contact state object.
        Returns:
            The contact state object.
        """

        return cls(**kwargs)

    @classmethod
    def zero(cls, **kwargs) -> ContactsState:
        """
        Build a zero contact state.
        Returns:
            The zero contact state.
        """

        return cls.build(**kwargs)

    def valid(self, **kwargs) -> bool:
        """
        Check if the contacts state is valid.
        """

        return True


@jax_dataclasses.pytree_dataclass
class ContactsParams(JaxsimDataclass, abc.ABC):
    """
    Abstract class representing the parameters of a contact model.
    """

    @abc.abstractmethod
    def build(self) -> ContactsParams:
        """
        Create a `ContactsParams` instance with specified parameters.
        Returns:
            The `ContactsParams` instance.
        """

        raise NotImplementedError

    def valid(self, *args, **kwargs) -> bool:
        """
        Check if the parameters are valid.
        Returns:
            True if the parameters are valid, False otherwise.
        """

        return True


@jax_dataclasses.pytree_dataclass
class ContactModel(abc.ABC):
    """
    Abstract class representing a contact model.
    Attributes:
        parameters: The parameters of the contact model.
        terrain: The terrain model.
    """

    parameters: ContactsParams = dataclasses.field(default_factory=ContactsParams)
    terrain: jaxsim.terrain.Terrain = dataclasses.field(
        default_factory=jaxsim.terrain.FlatTerrain
    )

    @abc.abstractmethod
    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        **kwargs,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces.
        Args:
            position: The position of the collidable point.
            velocity: The velocity of the collidable point.
        Returns:
            A tuple containing the contact force and additional information.
        """

        raise NotImplementedError
