import dataclasses
import functools
import pathlib
from typing import Dict, List, Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax_dataclasses
import rod
from jax_dataclasses import Static

import jaxsim.high_level
import jaxsim.physics
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model, StepData
from jaxsim.parsers import descriptions
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.utils import Mutability, Vmappable, oop

from . import simulator_callbacks as scb
from .ode_integration import IntegratorType


@jax_dataclasses.pytree_dataclass
class SimulatorData(Vmappable):
    """
    Data used by the simulator.

    It can be used as JaxSim state in a functional programming style.
    """

    # Simulation time stored in ns in order to prevent floats approximation
    time_ns: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.uint64)
    )

    # Terrain and contact parameters
    terrain: Terrain = dataclasses.field(default_factory=lambda: FlatTerrain())
    contact_parameters: SoftContactsParams = dataclasses.field(
        default_factory=lambda: SoftContactsParams()
    )

    # Dictionary containing all handled models
    models: Dict[str, Model] = dataclasses.field(default_factory=dict)

    # Default gravity vector (could be overridden for individual models)
    gravity: jtp.Vector = dataclasses.field(
        default_factory=lambda: jaxsim.physics.default_gravity()
    )


@jax_dataclasses.pytree_dataclass
class JaxSim(Vmappable):
    """The JaxSim simulator."""

    # Step size stored in ns in order to prevent floats approximation
    step_size_ns: Static[jtp.Int] = dataclasses.field(
        default_factory=lambda: jnp.array(1_000_000, dtype=jnp.uint64)
    )

    # Number of sub-steps performed at each integration step.
    # Note: there is no collision detection performed in sub-steps.
    steps_per_run: Static[jtp.Int] = dataclasses.field(default=1)

    # Default velocity representation (could be overridden for individual models)
    velocity_representation: Static[VelRepr] = dataclasses.field(
        default=VelRepr.Inertial
    )

    # Integrator type
    integrator_type: Static[IntegratorType] = dataclasses.field(
        default=IntegratorType.EulerForward
    )

    # Simulator data
    data: SimulatorData = dataclasses.field(default_factory=lambda: SimulatorData())

    @staticmethod
    def build(
        step_size: jtp.Float,
        steps_per_run: jtp.Int = 1,
        velocity_representation: VelRepr = VelRepr.Inertial,
        integrator_type: IntegratorType = IntegratorType.EulerSemiImplicit,
        simulator_data: SimulatorData | None = None,
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
            step_size_ns=jnp.array(step_size * 1e9, dtype=jnp.uint64),
            steps_per_run=int(steps_per_run),
            velocity_representation=velocity_representation,
            integrator_type=integrator_type,
            data=simulator_data if simulator_data is not None else SimulatorData(),
        )

    @functools.partial(
        oop.jax_tf.method_rw, static_argnames=["remove_models"], validate=False
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
            self.data.models = {}
        else:
            _ = [m.zero() for m in self.models()]

    @functools.partial(oop.jax_tf.method_rw, jit=False)
    def set_step_size(self, step_size: float) -> None:
        """
        Set the integration step size.

        Args:
            step_size: The integration step size in seconds.
        """

        self.step_size_ns = jnp.array(step_size * 1e9, dtype=jnp.uint64)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def step_size(self) -> jtp.Float:
        """
        Get the integration step size.

        Returns:
            The integration step size in seconds.
        """

        return jnp.array(self.step_size_ns / 1e9, dtype=float)

    @functools.partial(oop.jax_tf.method_ro)
    def dt(self) -> jtp.Float:
        """
        Return the integration step size in seconds.

        Returns:
            The integration step size in seconds.
        """

        return jnp.array((self.step_size_ns * self.steps_per_run) / 1e9, dtype=float)

    @functools.partial(oop.jax_tf.method_ro)
    def time(self) -> jtp.Float:
        """
        Return the current simulation time in seconds.

        Returns:
            The current simulation time in seconds.
        """

        return jnp.array(self.data.time_ns / 1e9, dtype=float)

    @functools.partial(oop.jax_tf.method_ro)
    def gravity(self) -> jtp.Vector:
        """
        Return the 3D gravity vector.

        Returns:
            The 3D gravity vector.
        """

        return jnp.array(self.data.gravity, dtype=float)

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def model_names(self) -> tuple[str, ...]:
        """
        Return the list of model names.

        Returns:
            The list of model names.
        """

        return tuple(self.data.models.keys())

    @functools.partial(
        oop.jax_tf.method_ro, static_argnames=["model_name"], jit=False, vmap=False
    )
    def get_model(self, model_name: str) -> Model:
        """
        Return the model with the given name.

        Args:
            model_name: The name of the model to return.

        Returns:
            The model with the given name.
        """

        if model_name not in self.data.models:
            raise ValueError(f"Failed to find model '{model_name}'")

        return self.data.models[model_name]

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def models(self, model_names: tuple[str, ...] | None = None) -> tuple[Model, ...]:
        """
        Return the simulated models.

        Args:
            model_names: Optional list of model names to return.
                         If None, all models are returned.

        Returns:
            The list of simulated models.
        """

        model_names = model_names if model_names is not None else self.model_names()
        return tuple(self.data.models[name] for name in model_names)

    @functools.partial(oop.jax_tf.method_rw)
    def set_gravity(self, gravity: jtp.Vector) -> None:
        """
        Set the gravity vector to all the simulated models.

        Args:
            gravity: The 3D gravity vector.
        """

        gravity = jnp.array(gravity, dtype=float)

        if gravity.size != 3:
            raise ValueError(gravity)

        self.data.gravity = gravity

        for model in self.data.models.values():
            model.physics_model.set_gravity(gravity=gravity)

    @functools.partial(oop.jax_tf.method_rw, jit=False, vmap=False, validate=False)
    def insert_model_from_description(
        self,
        model_description: Union[pathlib.Path, str, rod.Model],
        model_name: str | None = None,
        considered_joints: List[str] | None = None,
    ) -> Model:
        """
        Insert a model from a model description.

        Args:
            model_description: A path to an SDF/URDF file, a string containing its content, or a pre-parsed/pre-built rod model.
            model_name: The optional name of the model that overrides the one in the description.
            considered_joints: Optional list of joints to consider. It is also useful to specify the joint serialization.

        Returns:
            The newly inserted model.
        """

        if self.vectorized:
            raise RuntimeError("Cannot insert a model in a vectorized simulation")

        # Build the model from the given model description
        model = jaxsim.high_level.model.Model.build_from_model_description(
            model_description=model_description,
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

        # Return the newly inserted model
        return self.data.models[model.name()]

    @functools.partial(oop.jax_tf.method_rw, jit=False, vmap=False, validate=False)
    def insert_model_from_sdf(
        self,
        sdf: Union[pathlib.Path, str],
        model_name: str | None = None,
        considered_joints: List[str] | None = None,
    ) -> Model:
        """
        Insert a model from an SDF resource.
        """

        msg = "JaxSim.{} is deprecated, use JaxSim.{} instead."
        logging.warning(
            msg=msg.format("insert_model_from_sdf", "insert_model_from_description")
        )

        return self.insert_model_from_description(
            model_description=sdf,
            model_name=model_name,
            considered_joints=considered_joints,
        )

    @functools.partial(oop.jax_tf.method_rw, jit=False, vmap=False, validate=False)
    def insert_model(
        self,
        model_description: descriptions.ModelDescription,
        model_name: str | None = None,
    ) -> Model:
        """
        Insert a model from a model description object.

        Args:
            model_description: The model description object.
            model_name: Optional name of the model to insert.

        Returns:
            The newly inserted model.
        """

        if self.vectorized:
            raise RuntimeError("Cannot insert a model in a vectorized simulation")

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

        # Return the newly inserted model
        return self.data.models[model.name()]

    @functools.partial(
        oop.jax_tf.method_rw,
        jit=False,
        validate=False,
        static_argnames=["model_name"],
    )
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

    @functools.partial(oop.jax_tf.method_rw, vmap_in_axes=(0, None))
    def step(self, clear_inputs: bool = False) -> Dict[str, StepData]:
        """
        Advance the simulation by one step.

        Args:
            clear_inputs: Zero the inputs of the models after the integration.

        Returns:
            A dictionary containing the StepData of all models.
        """

        # Compute the initial and final time of the integration as integers
        t0_ns = jnp.array(self.data.time_ns, dtype=jnp.uint64)
        dt_ns = jnp.array(self.step_size_ns * self.steps_per_run, dtype=jnp.uint64)

        # Compute the final time using integer arithmetics
        tf_ns = t0_ns + dt_ns

        # We collect the StepData of all models
        step_data = {}

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

        # Store the final time
        self.data.time_ns += dt_ns

        return step_data

    @functools.partial(
        oop.jax_tf.method_ro,
        static_argnames=["horizon_steps"],
        vmap_in_axes=(0, None, 0, None),
    )
    def step_over_horizon(
        self,
        horizon_steps: jtp.Int,
        callback_handler: (
            Union["scb.SimulatorCallback", "scb.CallbackHandler"] | None
        ) = None,
        clear_inputs: jtp.Bool = False,
    ) -> Union[
        "JaxSim",
        tuple["JaxSim", tuple["scb.SimulatorCallback", tuple[jtp.PyTree, jtp.PyTree]]],
    ]:
        """
        Advance the simulation by a given number of steps.

        Args:
            horizon_steps: The number of steps to advance the simulation.
            callback_handler: A callback handler to inject custom login in the simulation loop.
            clear_inputs: Zero the inputs of the models after the integration.

        Returns:
            The updated simulator if no callback handler is provided, otherwise a tuple
            containing the updated simulator and a tuple containing callback data.
            The optional callback data is a tuple containing the updated callback object,
            the produced pre-step output, and the produced post-step output.
        """

        # Process a mutable copy of the simulator
        original_mutability = self._mutability()
        sim = self.copy().mutable(validate=True)

        # Helper to get callbacks from the handler
        get_cb = lambda h, cb_name: (
            getattr(h, cb_name) if h is not None and hasattr(h, cb_name) else None
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
        Carry = tuple[JaxSim, scb.CallbackHandler]
        carry_init: Carry = (sim, callback_handler)

        def body_fun(
            carry: Carry, xs: None
        ) -> tuple[Carry, tuple[jtp.PyTree, jtp.PyTree]]:
            sim, callback_handler = carry

            # Make sure to pass a mutable version of the simulator to the callbacks
            sim = sim.mutable(validate=True)

            # Callback: pre-step
            sim, out_pre_step = (
                pre_step_cb(sim) if pre_step_cb is not None else (sim, None)
            )

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

            return carry, (out_pre_step, out_post_step)

        # Integrate over the given horizon
        (sim, callback_handler), (
            out_pre_step_horizon,
            out_post_step_horizon,
        ) = jax.lax.scan(f=body_fun, init=carry_init, xs=None, length=horizon_steps)

        # Enforce original mutability of the entire simulator
        sim._set_mutability(original_mutability)

        return (
            sim
            if callback_handler is None
            else (
                sim,
                (callback_handler, (out_pre_step_horizon, out_post_step_horizon)),
            )
        )

    def vectorize(self: Self, batch_size: int) -> Self:
        """
        Inherit docs.
        """

        jaxsim_vec: JaxSim = super().vectorize(batch_size=batch_size)  # noqa

        # We need to manually specify the batch size of the handled models
        with jaxsim_vec.mutable_context(mutability=Mutability.MUTABLE):
            for model in jaxsim_vec.models():
                model.batch_size = batch_size

        return jaxsim_vec
