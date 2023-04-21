import dataclasses
import functools
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.high_level
import jaxsim.parsers.descriptions as descriptions
import jaxsim.physics
import jaxsim.simulation.simulator_callbacks as scb
import jaxsim.typing as jtp
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model, StepData
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.simulation import ode_integration
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class SimulatorData(JaxsimDataclass):
    """
    Data used by the simulator.

    It can be used as JaxSim state in a functional programming style.
    """

    # Simulation time stored in ns in order to prevent floats approximation
    time_ns: jtp.Int = jnp.array(0, dtype=jnp.int64)

    # Terrain and contact parameters
    terrain: Terrain = jax_dataclasses.field(default_factory=lambda: FlatTerrain())
    contact_parameters: SoftContactsParams = jax_dataclasses.field(
        default_factory=lambda: SoftContactsParams()
    )

    # Dictionary containing all handled models
    models: Dict[str, Model] = jax_dataclasses.field(default_factory=dict)

    # Default gravity vector (could be overridden for individual models)
    gravity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jaxsim.physics.default_gravity()
    )


@jax_dataclasses.pytree_dataclass
class JaxSim(JaxsimDataclass):
    """The JaxSim simulator."""

    # Step size stored in ns in order to prevent floats approximation
    step_size_ns: jtp.Int = jax_dataclasses.field(
        default_factory=lambda: jnp.array(1_000_000, dtype=jnp.int64)
    )

    # Number of sub-steps performed at each integration step.
    # Note: there is no collision detection performed in sub-steps.
    steps_per_run: jtp.Int = jax_dataclasses.static_field(default=1)

    # Default velocity representation (could be overridden for individual models)
    velocity_representation: VelRepr = jax_dataclasses.field(default=VelRepr.Inertial)

    # Integrator type
    integrator_type: ode_integration.IntegratorType = jax_dataclasses.static_field(
        default=ode_integration.IntegratorType.EulerForward
    )

    # Simulator data
    data: SimulatorData = dataclasses.field(default_factory=lambda: SimulatorData())

    @staticmethod
    def build(
        step_size: jtp.Float,
        steps_per_run: jtp.Int = 1,
        velocity_representation: VelRepr = VelRepr.Inertial,
        integrator_type: ode_integration.IntegratorType = ode_integration.IntegratorType.EulerSemiImplicit,
        simulator_data: SimulatorData = None,
    ) -> "JaxSim":
        """
        Build a JaxSim simulator object.

        Args:
            step_size: The integration step size in seconds.
            steps_per_run: Number of sub-steps performed at each integration step.
            velocity_representation: Default velocity representation of simulated models.
            integrator_type: Type of integrator used for integrating the equations of motion.
            simulator_data: Optional simulator data to initialize the simulator state.

        Returns:
            The JaxSim simulator object.
        """

        return JaxSim(
            step_size_ns=jnp.array(step_size * 1e9, dtype=jnp.int64),
            steps_per_run=int(steps_per_run),
            velocity_representation=velocity_representation,
            integrator_type=integrator_type,
            data=simulator_data if simulator_data is not None else SimulatorData(),
        )

    def reset(self, remove_models: bool = True) -> None:
        """
        Reset the simulator.

        Args:
            remove_models: Flag indicating whether to remove all models from the simulator.
                           If False, the models are kept but their state is reset.
        """

        self.data.time_ns = jnp.zeros_like(self.data.time_ns)

        if remove_models:
            self.data.models = dict()
        else:
            _ = [m.zero() for m in self.models()]

    def set_step_size(self, step_size: float) -> None:
        """
        Set the integration step size.

        Args:
            step_size: The integration step size in seconds.
        """

        self.step_size_ns = jnp.array(step_size * 1e9, dtype=jnp.int64)

    def dt(self) -> jtp.Float:
        """
        Return the integration step size in seconds.

        Returns:
            The integration step size in seconds.
        """

        return (self.step_size_ns * self.steps_per_run) / 1e9

    def time(self) -> jtp.Float:
        """
        Return the current simulation time in seconds.

        Returns:
            The current simulation time in seconds.
        """

        return self.data.time_ns / 1e9

    def gravity(self) -> jtp.Vector:
        """
        Return the 3D gravity vector.

        Returns:
            The 3D gravity vector.
        """

        return self.data.gravity

    def model_names(self) -> List[str]:
        """
        Return the list of model names.

        Returns:
            The list of model names.
        """

        return list(self.data.models.keys())

    def get_model(self, model_name: str) -> Model:
        """
        Return the model with the given name.

        Args:
            model_name: The name of the model to return.

        Returns:
            The model with the given name.
        """

        if model_name not in self.data.models.keys():
            raise ValueError(f"Failed to find model '{model_name}'")

        return self.data.models[model_name]

    def models(self, model_names: List[str] = None) -> List[Model]:
        """
        Return the simulated models.

        Args:
            model_names: Optional list of model names to return.
                         If None, all models are returned.

        Returns:
            The list of simulated models.
        """

        model_names = model_names if model_names is not None else self.model_names()
        return [self.data.models[name] for name in model_names]

    def set_gravity(self, gravity: jtp.Vector) -> None:
        """
        Set the gravity vector to all the simulated models.

        Args:
            gravity: The 3D gravity vector.
        """

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
        """
        Insert a model from an SDF resource.

        Args:
            sdf: The SDF resource.
            model_name: Optional name of the model to insert.
                        It defaults to the name of the model found in the SDF resource.
            considered_joints: Optional list of joints to consider.
                               It is also useful to specify the joint serialization.

        Returns:
            The newly inserted model.
        """

        # Build the model from the input SDF resource
        model = jaxsim.high_level.model.Model.build_from_sdf(
            sdf=sdf,
            model_name=model_name,
            vel_repr=self.velocity_representation,
            considered_joints=considered_joints,
        )

        # Make sure the model is not already part of the simulation
        if model.name() in self.model_names():
            msg = f"Model '{model.name()}' is already part of the simulation"
            raise ValueError(msg)

        # Insert the model
        self.data.models[model.name()] = model

        # Propagate the current mutability property to make sure that also the
        # newly inserted model matches the mutability of the simulator
        self._set_mutability(self._mutability())

        # Return the newly inserted model
        return self.data.models[model.name()]

    def insert_model(
        self, model_description: descriptions.ModelDescription, model_name: str = None
    ) -> Model:
        """
        Insert a model from a model description object.

        Args:
            model_description: The model description object.
            model_name: Optional name of the model to insert.

        Returns:
            The newly inserted model.
        """

        model_name = model_name if model_name is not None else model_description.name

        if model_name in self.model_names():
            msg = f"Model '{model_name}' is already part of the simulation"
            raise ValueError(msg)

        # Build the physics model the model description
        physics_model = PhysicsModel.build_from(
            model_description=model_description, gravity=self.gravity()
        )

        # Build the high-level model from the physics model
        model = jaxsim.high_level.model.Model.build(
            model_name=model_name,
            physics_model=physics_model,
            vel_repr=self.velocity_representation,
        )

        # Insert the model into the simulators
        self.data.models[model.name()] = model
        self._set_mutability(self._mutability())

        # Return the newly inserted model
        return self.data.models[model.name()]

    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the simulator.

        Args:
            model_name: The name of the model to remove.
        """

        if model_name not in self.model_names():
            msg = f"Model '{model_name}' is not part of the simulation"
            raise ValueError(msg)

        _ = self.data.models.pop(model_name)
        self._set_mutability(self._mutability())

    def step(self, clear_inputs: bool = False) -> Dict[str, StepData]:
        """
        Advance the simulation by one step.

        Args:
            clear_inputs: Zero the inputs of the models after the integration.

        Returns:
            A dictionary containing the StepData of all models.
        """

        # Compute the initial and final time of the integration as integers
        t0_ns = jnp.array(self.data.time_ns, dtype=jnp.int64)
        dt_ns = jnp.array(self.step_size_ns * self.steps_per_run, dtype=jnp.int64)

        # Compute the final time using integer arithmetics
        tf_ns = t0_ns + dt_ns

        # We collect the StepData of all models
        step_data = dict()

        for model in self.models():
            # Integrate individually all models and collect their StepData.
            # We use the context manager to make sure that the PyTree of the models
            # never changes, so that it never triggers JIT recompilations.
            with model.editable(validate=True) as integrated_model:
                step_data[model.name()] = integrated_model.integrate(
                    t0=jnp.array(t0_ns, dtype=float) / 1e9,
                    tf=jnp.array(tf_ns, dtype=float) / 1e9,
                    sub_steps=self.steps_per_run,
                    integrator_type=self.integrator_type,
                    terrain=self.data.terrain,
                    contact_parameters=self.data.contact_parameters,
                    clear_inputs=clear_inputs,
                )

            self.data.models[model.name()].data = integrated_model.data

        self.data.time_ns += dt_ns

        self._set_mutability(self._mutability())
        return step_data

    @functools.partial(jax.jit, static_argnames=["horizon_steps"])
    def step_over_horizon(
        self,
        horizon_steps: jtp.Int,
        callback_handler: Union[
            "scb.SimulatorCallback",
            "scb.CallbackHandler",
        ] = None,
        clear_inputs: jtp.Bool = False,
    ) -> Union["JaxSim", Tuple["JaxSim", Tuple["scb.SimulatorCallback", jtp.PyTree]]]:
        """
        Advance the simulation by a given number of steps.

        Args:
            horizon_steps: The number of steps to advance the simulation.
            callback_handler: A callback handler to inject custom login in the simulation loop.
            clear_inputs: Zero the inputs of the models after the integration.

        Returns:
            The updated simulator if no callback handler is provided, otherwise a tuple
            containing the updated simulator and a tuple with the updated callback object
            and the optional output it produced.
        """

        # Process a mutable copy of the simulator
        original_mutability = self._mutability()
        sim = self.copy().mutable(validate=True)

        # Helper to get callbacks from the handler
        get_cb = (
            lambda h, cb_name: getattr(h, cb_name)
            if h is not None and hasattr(h, cb_name)
            else None
        )

        # Get the callbacks
        configure_cb: Optional[scb.ConfigureCallbackSignature] = get_cb(
            h=callback_handler, cb_name="configure_cb"
        )
        pre_step_cb: Optional[scb.PreStepCallbackSignature] = get_cb(
            h=callback_handler, cb_name="pre_step_cb"
        )
        post_step_cb: Optional[scb.PostStepCallbackSignature] = get_cb(
            h=callback_handler, cb_name="post_step_cb"
        )

        # Callback: configuration
        sim = configure_cb(sim) if configure_cb is not None else sim

        # Initialize the carry
        Carry = Tuple[JaxSim, scb.CallbackHandler]
        carry_init: Carry = (sim, callback_handler)

        def body_fun(carry: Carry, xs: None) -> Tuple[Carry, jtp.PyTree]:
            sim, callback_handler = carry

            # Make sure to pass a mutable version of the simulator to the callbacks
            sim = sim.mutable(validate=True)

            # Callback: pre-step
            # TODO: should we allow also producing a pre-step output?
            sim = pre_step_cb(sim) if pre_step_cb is not None else sim

            # Integrate all models
            step_data = sim.step(clear_inputs=clear_inputs)

            # Callback: post-step
            sim, out_post_step = (
                post_step_cb(sim, step_data)
                if post_step_cb is not None
                else (sim, None)
            )

            # Pack the carry
            carry = (sim, callback_handler)

            return carry, out_post_step

        # Integrate over the given horizon
        (sim, callback_handler), out_cb_horizon = jax.lax.scan(
            f=body_fun, init=carry_init, xs=None, length=horizon_steps
        )

        # Enforce original mutability of the entire simulator
        sim._set_mutability(original_mutability)

        return (
            sim
            if callback_handler is None
            else (sim, (callback_handler, out_cb_horizon))
        )
