import jax.flatten_util
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.physics.algos.soft_contacts import SoftContactsState
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.physics.model.physics_model_state import (
    PhysicsModelInput,
    PhysicsModelState,
)


@jax_dataclasses.pytree_dataclass
class ODEInput:

    physics_model: PhysicsModelInput

    @staticmethod
    def zero(physics_model: PhysicsModel) -> "ODEInput":

        return ODEInput(
            physics_model=PhysicsModelInput.zero(physics_model=physics_model)
        )

    def valid(self, physics_model: PhysicsModel) -> bool:

        return self.physics_model.valid(physics_model=physics_model)

    def replace(self, validate: bool = True, **kwargs) -> "ODEInput":

        with jax_dataclasses.copy_and_mutate(self, validate=validate) as updated_input:

            _ = [updated_input.__setattr__(k, v) for k, v in kwargs.items()]

        return updated_input


@jax_dataclasses.pytree_dataclass
class ODEState:

    physics_model: PhysicsModelState
    soft_contacts: SoftContactsState

    def serialize(self) -> jtp.VectorJax:

        serialized_object, _ = jax.flatten_util.ravel_pytree(self)
        return serialized_object

    @staticmethod
    def deserialize(data: jtp.VectorJax, physics_model: PhysicsModel) -> "ODEState":

        dummy_object = ODEState.zero(physics_model=physics_model)
        _, unflatten_data = jax.flatten_util.ravel_pytree(dummy_object)

        return unflatten_data(data)

    def replace(self, validate: bool = True, **kwargs) -> "ODEState":

        with jax_dataclasses.copy_and_mutate(self, validate=validate) as updated_state:

            _ = [updated_state.__setattr__(k, v) for k, v in kwargs.items()]

        return updated_state

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
