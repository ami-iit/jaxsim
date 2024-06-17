from __future__ import annotations

import abc
from typing import Any

import jaxsim.terrain
import jaxsim.typing as jtp


class ContactsState(abc.ABC):
    """
    Abstract class storing the state of the contacts model.
    """

    @classmethod
    @abc.abstractmethod
    def build(cls, **kwargs) -> ContactsState:
        """
        Build the contact state object.
        Returns:
            The contact state object.
        """

        return cls(**kwargs)

    @classmethod
    @abc.abstractmethod
    def zero(cls, **kwargs) -> ContactsState:
        """
        Build a zero contact state.
        Returns:
            The zero contact state.
        """

        return cls.build(**kwargs)

    @abc.abstractmethod
    def valid(self, **kwargs) -> bool:
        """
        Check if the contacts state is valid.
        """

        return True


class ContactsParams(abc.ABC):
    """
    Abstract class representing the parameters of a contact model.
    """

    @classmethod
    @abc.abstractmethod
    def build(cls) -> ContactsParams:
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


class ContactModel(abc.ABC):
    """
    Abstract class representing a contact model.
    Attributes:
        parameters: The parameters of the contact model.
        terrain: The terrain model.
    """

    parameters: ContactsParams
    terrain: jaxsim.terrain.Terrain

    @abc.abstractmethod
    def compute_contact_forces(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        **kwargs,
    ) -> tuple[Any, ...]:
        """
        Compute the contact forces.
        Args:
            position: The position of the collidable point.
            velocity: The velocity of the collidable point.
        Returns:
            A tuple containing the contact force and additional information.
        """
