import copy
from typing import Dict, List

import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import numpy.typing as npt

import jaxsim.high_level
import jaxsim.parsers.descriptions as descriptions
import jaxsim.physics
from jaxsim.physics.model.physics_model import PhysicsModel

from .common import VelRepr


class World:
    def __init__(
        self,
        default_velocity_representation: VelRepr = VelRepr.Mixed,
    ):

        self.default_vel_repr = default_velocity_representation

        self._gravity = jaxsim.physics.default_gravity()
        self._models: Dict[str, jaxsim.high_level.model.Model] = dict()

    def valid(self) -> bool:

        raise NotImplementedError

    def time(self) -> float:

        return self.t

    def gravity(self) -> npt.NDArray:

        return self._gravity

    def model_names(self) -> List[str]:

        return list(self._models.keys())

    def get_model(self, model_name: str) -> "jaxsim.high_level.model.Model":

        if model_name not in self._models.keys():
            raise ValueError(f"Failed to find model '{model_name}' in the world")

        return self._models[model_name]

    def models(
        self, model_names: List[str] = None
    ) -> List["jaxsim.high_level.model.Model"]:

        model_names = model_names if model_names is not None else self.model_names()
        return [self._models[name] for name in model_names]

    def set_gravity(self, gravity: npt.NDArray) -> None:

        if gravity.size != 3:
            raise ValueError(gravity)

        self._gravity = gravity.squeeze()

        # Make a deep copy so that the object is not affected if the logic below fails
        models_with_new_gravity = copy.deepcopy(self._models)

        for model in models_with_new_gravity:

            model: jaxsim.high_level.model.Model

            # Change the gravity vector
            with jax_dataclasses.copy_and_mutate(model.physics_model) as changed_model:
                changed_model.gravity = np.vstack(np.hstack([np.zeros(3), gravity]))

            # Update the model description
            model.physics_model = changed_model

        # Store the new models
        self._models = models_with_new_gravity

    def insert_model(
        self, model_description: descriptions.ModelDescription, model_name: str = None
    ) -> str:

        model_name = model_name if model_name is not None else model_description.name

        if model_name in self.model_names():
            msg = f"Model '{model_name}' is already part of the world"
            raise ValueError(msg)

        physics_model = PhysicsModel.build_from(
            model_description=model_description, gravity=self.gravity()
        )

        model = jaxsim.high_level.model.Model.build(
            physics_model=physics_model, vel_repr=self.default_vel_repr
        )

        self._models[model.name()] = model
        return self._models[model.name()].name()

    def remove_model(self, model_name: str) -> None:

        if model_name not in self.model_names():
            msg = f"Model '{model_name}' is not part of the world"
            raise ValueError(msg)

        self._models.pop(model_name)
