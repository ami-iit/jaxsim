from __future__ import annotations

import abc
from typing import Any

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ContactsState(JaxsimDataclass):
    """
    Abstract class storing the state of the contacts model.
    """

    @classmethod
    @abc.abstractmethod
    def build(cls: type[Self], **kwargs) -> Self:
        """
        Build the contact state object.

        Returns:
            The contact state object.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def zero(cls: type[Self], **kwargs) -> Self:
        """
        Build a zero contact state.

        Returns:
            The zero contact state.
        """
        pass

    @abc.abstractmethod
    def valid(self, **kwargs) -> jtp.BoolLike:
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
    def build(cls: type[Self], **kwargs) -> Self:
        """
        Create a `ContactsParams` instance with specified parameters.

        Returns:
            The `ContactsParams` instance.
        """
        pass

    @abc.abstractmethod
    def valid(self, **kwargs) -> jtp.BoolLike:
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
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        **kwargs,
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        """
        Compute the contact forces.

        Args:
            model: The model to consider.
            data: The data of the considered model.

        Returns:
            A tuple containing as first element the computed 6D contact force applied to the contact point and expressed in the world frame,
            and as second element a tuple of optional additional information.
        """

        pass

    def initialize_model_and_data(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        validate: bool = True,
    ) -> tuple[js.model.JaxSimModel, js.data.JaxSimModelData]:
        """
        Helper function to initialize the active model and data objects.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered robot model.
            validate:
                Whether to validate if the model and data objects have been
                initialized with the current contact model.

        Returns:
            The initialized model and data objects.
        """

        with model.editable(validate=validate) as model_out:
            model_out.contact_model = self

        with data.editable(validate=validate) as data_out:
            data_out.contacts_params = data.contacts_params

        return model_out, data_out
