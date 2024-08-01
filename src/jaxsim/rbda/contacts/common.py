from __future__ import annotations

import abc
from typing import Any

import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


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
        pass

    @classmethod
    @abc.abstractmethod
    def zero(cls, **kwargs) -> ContactsState:
        """
        Build a zero contact state.

        Returns:
            The zero contact state.
        """
        pass

    @abc.abstractmethod
    def valid(self, **kwargs) -> bool:
        """
        Check if the contacts state is valid.
        """
        pass


class ContactsParams(JaxsimDataclass):
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
        pass

    @abc.abstractmethod
    def valid(self, *args, **kwargs) -> bool:
        """
        Check if the parameters are valid.
        Returns:
            True if the parameters are valid, False otherwise.
        """
        pass


class ContactModel(JaxsimDataclass):
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
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        """
        Compute the contact forces.

        Args:
            position: The position of the collidable point w.r.t. the world frame.
            velocity:
                The linear velocity of the collidable point (linear component of the mixed 6D velocity).

        Returns:
            A tuple containing as first element the computed 6D contact force applied to the contact point and expressed in the world frame,
            and as second element a tuple of optional additional information.
        """
        pass
