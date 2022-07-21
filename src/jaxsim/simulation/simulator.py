import dataclasses
import pathlib
from typing import Dict, List, Union

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.high_level
import jaxsim.parsers.descriptions as descriptions
import jaxsim.physics
import jaxsim.typing as jtp
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.simulation import ode_integration
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class SimulatorData:

    time_ns: jtp.Int = jnp.array(0, dtype=int)

    terrain: Terrain = jax_dataclasses.field(default_factory=lambda: FlatTerrain())
    contact_parameters: SoftContactsParams = jax_dataclasses.field(
        default_factory=lambda: SoftContactsParams()
    )

    models: Dict[str, Model] = jax_dataclasses.field(default_factory=dict)

    gravity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jaxsim.physics.default_gravity()
    )


@jax_dataclasses.pytree_dataclass
class JaxSim(JaxsimDataclass):

    step_size_ns: jtp.Int = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.001, dtype=int)
    )
    steps_per_run: jtp.Int = jax_dataclasses.static_field(default=1)

    velocity_representation: VelRepr = jax_dataclasses.field(default=VelRepr.Mixed)

    integrator_type: ode_integration.IntegratorType = jax_dataclasses.static_field(
        default=ode_integration.IntegratorType.EulerForward
    )

    data: SimulatorData = dataclasses.field(default_factory=lambda: SimulatorData())

    @staticmethod
    def build(
        step_size: jtp.Float,
        steps_per_run: jtp.Int = 1,
        velocity_representation: VelRepr = VelRepr.Mixed,
        integrator_type: ode_integration.IntegratorType = ode_integration.IntegratorType.EulerForward,
        simulator_data: SimulatorData = None,
    ) -> "JaxSim":

        return JaxSim(
            step_size_ns=jnp.array(step_size * 1e9, dtype=int),
            steps_per_run=int(steps_per_run),
            velocity_representation=velocity_representation,
            integrator_type=integrator_type,
            data=simulator_data if simulator_data is not None else SimulatorData(),
        )

    def reset(self, remove_models: bool = True) -> None:

        self.data.time_ns *= 0

        if remove_models:
            self.data.models = dict()
        else:
            _ = [m.zero() for m in self.models()]

    def set_step_size(self, step_size: float) -> None:

        self.step_size_ns = jnp.array(step_size * 1e9, dtype=int)

    def dt(self) -> jtp.Float:

        return (self.step_size_ns * self.steps_per_run) / 1e9

    def time(self) -> jtp.Float:

        return self.data.time_ns / 1e9

    def gravity(self) -> jtp.Vector:

        return self.data.gravity

    def model_names(self) -> List[str]:

        return list(self.data.models.keys())

    def get_model(self, model_name: str) -> Model:

        if model_name not in self.data.models.keys():
            raise ValueError(f"Failed to find model '{model_name}'")

        return self.data.models[model_name]

    def models(self, model_names: List[str] = None) -> List[Model]:

        model_names = model_names if model_names is not None else self.model_names()
        return [self.data.models[name] for name in model_names]

    def set_gravity(self, gravity: jtp.Vector):

        gravity = jnp.array(gravity)

        if gravity.size != 3:
            raise ValueError(gravity)

        self.data.gravity = gravity

        for model_name, model in self.data.models.items():
            model.physics_model.set_gravity(gravity=gravity)

        self._set_mutability(self._mutability())

    def insert_model_from_sdf(
        self,
        sdf: Union[pathlib.Path, str],
        model_name: str = None,
        considered_joints: List[str] = None,
    ) -> Model:

        model = jaxsim.high_level.model.Model.build_from_sdf(
            sdf=sdf,
            model_name=model_name,
            vel_repr=self.velocity_representation,
            considered_joints=considered_joints,
        )

        if model_name in self.model_names():
            msg = f"Model '{model_name}' is already part of the simulation"
            raise ValueError(msg)

        self.data.models[model.name()] = model
        self._set_mutability(self._mutability())

        return self.data.models[model.name()]

    def insert_model(
        self, model_description: descriptions.ModelDescription, model_name: str = None
    ) -> Model:

        model_name = model_name if model_name is not None else model_description.name

        if model_name in self.model_names():
            msg = f"Model '{model_name}' is already part of the simulation"
            raise ValueError(msg)

        physics_model = PhysicsModel.build_from(
            model_description=model_description, gravity=self.gravity()
        )

        model = jaxsim.high_level.model.Model.build(
            model_name=model_name,
            physics_model=physics_model,
            vel_repr=self.velocity_representation,
        )

        self.data.models[model.name()] = model
        self._set_mutability(self._mutability())

        return self.data.models[model.name()]

    def remove_model(self, model_name: str) -> None:

        if model_name not in self.model_names():
            msg = f"Model '{model_name}' is not part of the simulation"
            raise ValueError(msg)

        self.data.models.pop(model_name)
        self._set_mutability(self._mutability())

    def step(self) -> None:

        t0_ns = jnp.array(self.data.time_ns, dtype=int)
        dt_ns = jnp.array(self.step_size_ns * self.steps_per_run, dtype=int)

        tf_ns = t0_ns + dt_ns

        for model in self.models():

            with model.editable(validate=True) as integrated_model:

                integrated_model.integrate(
                    t0=jnp.array(t0_ns, dtype=float) / 1e9,
                    tf=jnp.array(tf_ns, dtype=float) / 1e9,
                    sub_steps=self.steps_per_run,
                    integrator_type=self.integrator_type,
                    terrain=self.data.terrain,
                    contact_parameters=self.data.contact_parameters,
                )

            self.data.models[model.name()].data = integrated_model.data

        self.data.time_ns += dt_ns

        self._set_mutability(self._mutability())
