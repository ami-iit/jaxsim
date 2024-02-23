import jax.flatten_util
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.physics.algos.soft_contacts import SoftContactsState
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.physics.model.physics_model_state import (
    PhysicsModelInput,
    PhysicsModelState,
)
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class ODEInput(JaxsimDataclass):
    """"""

    physics_model: PhysicsModelInput

    @staticmethod
    def build(
        physics_model_input: PhysicsModelInput | None = None,
        physics_model: PhysicsModel | None = None,
    ) -> "ODEInput":
        """"""

        physics_model_input = (
            physics_model_input
            if physics_model_input is not None
            else PhysicsModelInput.zero(physics_model=physics_model)
        )

        return ODEInput(physics_model=physics_model_input)

    @staticmethod
    def zero(physics_model: PhysicsModel) -> "ODEInput":
        return ODEInput(
            physics_model=PhysicsModelInput.zero(physics_model=physics_model)
        )

    def valid(self, physics_model: PhysicsModel) -> bool:
        return self.physics_model.valid(physics_model=physics_model)


@jax_dataclasses.pytree_dataclass
class ODEState(JaxsimDataclass):
    """"""

    physics_model: PhysicsModelState
    soft_contacts: SoftContactsState

    @staticmethod
    def build(
        physics_model_state: PhysicsModelState | None = None,
        soft_contacts_state: SoftContactsState | None = None,
        physics_model: PhysicsModel | None = None,
    ) -> "ODEState":
        """"""

        physics_model_state = (
            physics_model_state
            if physics_model_state is not None
            else PhysicsModelState.zero(physics_model=physics_model)
        )

        soft_contacts_state = (
            soft_contacts_state
            if soft_contacts_state is not None
            else SoftContactsState.zero(physics_model=physics_model)
        )

        return ODEState(
            physics_model=physics_model_state, soft_contacts=soft_contacts_state
        )

    @staticmethod
    def deserialize(data: jtp.VectorJax, physics_model: PhysicsModel) -> "ODEState":
        dummy_object = ODEState.zero(physics_model=physics_model)
        _, unflatten_data = jax.flatten_util.ravel_pytree(dummy_object)

        return unflatten_data(data)

    @staticmethod
    def zero(physics_model: PhysicsModel) -> "ODEState":
        model_state = ODEState(
            physics_model=PhysicsModelState.zero(physics_model=physics_model),
            soft_contacts=SoftContactsState.zero(physics_model=physics_model),
        )

        assert model_state.valid(physics_model)
        return model_state

    def valid(self, physics_model: PhysicsModel) -> bool:
        return self.physics_model.valid(
            physics_model=physics_model
        ) and self.soft_contacts.valid(physics_model=physics_model)
