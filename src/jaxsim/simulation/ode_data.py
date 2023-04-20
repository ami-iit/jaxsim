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
    physics_model: PhysicsModelInput

    @staticmethod
    def zero(physics_model: PhysicsModel) -> "ODEInput":
        return ODEInput(
            physics_model=PhysicsModelInput.zero(physics_model=physics_model)
        )

    def valid(self, physics_model: PhysicsModel) -> bool:
        return self.physics_model.valid(physics_model=physics_model)


@jax_dataclasses.pytree_dataclass
class ODEState(JaxsimDataclass):
    physics_model: PhysicsModelState
    soft_contacts: SoftContactsState

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
