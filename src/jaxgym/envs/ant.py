import dataclasses
import pathlib
from typing import Any, ClassVar, Optional

import jax.numpy as jnp
import jax.random
import jax_dataclasses
import numpy as np
import numpy.typing as npt
import rod

import jaxgym.jax.pytree_space as spaces
import jaxsim.typing as jtp
from jaxgym.jax import JaxDataclassEnv, JaxEnv
from jaxgym.vector.jax import JaxVectorEnv
from jaxsim import JaxSim
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim.simulation.simulator import SimulatorData, VelRepr
from jaxsim.utils import JaxsimDataclass, Mutability


@jax_dataclasses.pytree_dataclass
class AntObservation(JaxsimDataclass):
    """Observation of the Ant environment."""

    base_height: jtp.Float
    gravity_projection: jtp.Array

    joint_positions: jtp.Array
    joint_velocities: jtp.Array

    base_linear_velocity: jtp.Array
    base_angular_velocity: jtp.Array

    contact_state: jtp.Array

    @staticmethod
    def build(
        base_height: jtp.Float,
        gravity_projection: jtp.Array,
        joint_positions: jtp.Array,
        joint_velocities: jtp.Array,
        base_linear_velocity: jtp.Array,
        base_angular_velocity: jtp.Array,
        contact_state: jtp.Array,
    ) -> "AntObservation":
        """Build an AntObservation object."""

        return AntObservation(
            base_height=jnp.array(base_height, dtype=float),
            gravity_projection=jnp.array(gravity_projection, dtype=float),
            joint_positions=jnp.array(joint_positions, dtype=float),
            joint_velocities=jnp.array(joint_velocities, dtype=float),
            base_linear_velocity=jnp.array(base_linear_velocity, dtype=float),
            base_angular_velocity=jnp.array(base_angular_velocity, dtype=float),
            contact_state=jnp.array(contact_state, dtype=bool),
        )


import multiprocessing

from meshcat_viz import MeshcatWorld


@dataclasses.dataclass
class MeshcatVizRenderState:
    """Render state of a meshcat-viz visualizer."""

    world: MeshcatWorld = dataclasses.dataclass(init=False)

    _gui_process: Optional[multiprocessing.Process] = dataclasses.field(
        default=None, init=False, repr=False, hash=False, compare=False
    )

    _jaxsim_to_meshcat_viz_name: dict[str, str] = dataclasses.field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        """"""

        self.world = MeshcatWorld()
        self.world.open()

    @staticmethod
    def open_window(web_url: str) -> None:
        """Open a new window with the given web url."""

        import webview

        print(web_url)
        webview.create_window("meshcat", web_url)
        webview.start(gui="qt")

    def open_window_in_process(self) -> None:
        """"""

        if self._gui_process is not None:
            self._gui_process.terminate()
            self._gui_process.close()

        self._gui_process = multiprocessing.Process(
            target=MeshcatVizRenderState.open_window, args=(self.world.web_url,)
        )
        self._gui_process.start()


StateType = SimulatorData
ActType = jnp.ndarray
ObsType = AntObservation
RewardType = float | jnp.ndarray
TerminalType = bool | jnp.ndarray
# RenderStateType = None
RenderStateType = MeshcatVizRenderState


@jax_dataclasses.pytree_dataclass
class AntReachTargetFuncEnvV0(
    JaxDataclassEnv[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    ]
):
    """Ant environment implementing a target reaching task."""

    name: ClassVar = jax_dataclasses.static_field(default="AntReachTargetFuncEnvV0")

    # Store an instance of the JaxSim simulator.
    # It gets initialized with SimulatorData with a functional approach.
    _simulator: JaxSim = jax_dataclasses.field(default=None)

    def __post_init__(self) -> None:
        """Environment initialization."""

        # Dummy initialization (not needed here)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            _ = self.jaxsim

        # simulator_data = self.initial(rng=jax.random.PRNGKey(seed=0))
        # dofs = simulator_data.models["ant"].dofs()
        dofs = self.jaxsim.get_model(model_name="ant").dofs()

        # Create the action space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            high = jnp.array([50.0] * dofs, dtype=float)
            self._action_space = spaces.PyTree(low=-high, high=high)

        # Get joint limits
        s_min, s_max = self.jaxsim.get_model(model_name="ant").joint_limits()
        s_range = s_max - s_min

        low = AntObservation.build(
            base_height=0.25,
            gravity_projection=-jnp.ones(3),
            joint_positions=s_min - 0.05 * s_range,
            joint_velocities=-4.0 * jnp.ones_like(s_min),
            base_linear_velocity=-5.0 * jnp.ones(3),
            base_angular_velocity=-10.0 * jnp.ones(3),
            contact_state=jnp.array([False] * 4),
        )

        high = AntObservation.build(
            base_height=1.0,
            gravity_projection=jnp.ones(3),
            joint_positions=s_max + 0.05 * s_range,
            joint_velocities=4.0 * jnp.ones_like(s_max),
            base_linear_velocity=5.0 * jnp.ones(3),
            base_angular_velocity=10.0 * jnp.ones(3),
            contact_state=jnp.array([True] * 4),
        )

        # Create the observation space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._observation_space = spaces.PyTree(low=low, high=high)

    @property
    def jaxsim(self) -> JaxSim:
        """"""

        if self._simulator is not None:
            return self._simulator

        # Create the jaxsim simulator.
        # We use a small integration step so that contact detection is more accurate,
        # and perform multiple integration steps when we apply the action.
        simulator = JaxSim.build(
            # Note: any change of either 'step_size' or 'steps_per_run' requires
            # updating the number of integration steps in the 'transition' method.
            step_size=0.000_500,
            steps_per_run=1,
            # velocity_representation=VelRepr.Inertial,  # TODO
            velocity_representation=VelRepr.Body,
            integrator_type=IntegratorType.EulerSemiImplicit,
            simulator_data=SimulatorData(
                gravity=jnp.array([0, 0, -10.0]),
                # contact_parameters=SoftContactsParams.build(K=5_000, D=10),
                contact_parameters=SoftContactsParams.build(K=10_000, D=20),
            ),
        ).mutable(mutable=True, validate=False)

        # Get the SDF path
        model_sdf_path = (
            pathlib.Path.home()
            / "git"
            / "jaxsim"
            / "examples"
            / "resources"
            / "ant.sdf"
        )

        # TODO: load with rod and change the pos limits spring & friction params

        # Insert the model
        _ = simulator.insert_model_from_description(
            model_description=model_sdf_path, model_name="ant"
        )

        # Fix the pytree structure of the model data so that its corresponding shape
        # does not change. This is important to keep enabled the shape validation
        # checks for JIT compilation.
        simulator.data.models = {
            model_name: jax.tree_util.tree_map(lambda leaf: jnp.array(leaf), model_data)
            for model_name, model_data in simulator.data.models.items()
        }

        # Store the simulator object and configure it as immutable with enabled
        # pytree structure validation.
        # This is done to ensure that the corresponding pytree structure remains constant,
        # preventing unwanted JIT recompilations due to mistakes when setting its data.
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._simulator = simulator.mutable(mutable=True, validate=True)

        return self._simulator

    def initial(self, rng: Any = None) -> StateType:
        """"""

        # Sample an initial observation
        initial_observation: AntObservation = self.observation_space.sample_with_key(
            key=rng
        )

        with self.jaxsim.editable(validate=False) as simulator:
            # Reset the simulator and get the model
            simulator.reset(remove_models=False)
            model = simulator.get_model(model_name="ant")

            # Reset the joint positions
            model.reset_joint_positions(
                positions=initial_observation.joint_positions,
                joint_names=model.joint_names(),
            )

            # Reset the joint velocities
            # model.reset_joint_velocities(
            #     velocities=0.1 * initial_observation.joint_velocities,
            #     joint_names=model.joint_names(),
            # )

            # TODO: inizializzare s.t. non ci siano penetrazioni leg/terrain
            # Reset the base position
            model.reset_base_position(
                # position=jnp.array([0, 0, initial_observation.base_height])
                position=jnp.array([0, 0, 0.5])
            )

            # Reset the base velocity
            model.reset_base_velocity(
                base_velocity=jnp.hstack(
                    [
                        0.1 * initial_observation.base_linear_velocity,
                        0.1 * initial_observation.base_angular_velocity,
                    ]
                )
            )

            # Simulate for 1s so that the model starts from a
            # resting pose on the ground
            simulator = simulator.step_over_horizon(
                horizon_steps=2 * 1000, clear_inputs=True
            )

        # Return the simulation state
        return simulator.data

    def transition(
        self, state: StateType, action: ActType, rng: Any = None
    ) -> StateType:
        """"""

        # Get the JaxSim simulator
        simulator = self.jaxsim

        # Initialize the simulator with the environment state (containing SimulatorData)
        with simulator.editable(validate=True) as simulator:
            simulator.data = state

        @jax_dataclasses.pytree_dataclass
        class SetTorquesOverHorizon(simulator_callbacks.PreStepCallback):
            def pre_step(self, sim: JaxSim) -> JaxSim:
                """"""

                model = sim.get_model(model_name="ant")
                model.zero_input()
                model.set_joint_generalized_force_targets(
                    forces=jnp.atleast_1d(action), joint_names=model.joint_names()
                )

                return sim

        # Compute the number of integration steps to perform
        # transition_step_duration = 0.050
        # number_of_integration_steps = int(transition_step_duration / simulator.dt())
        # number_of_integration_steps = jnp.array(
        #     transition_step_duration / simulator.dt(), dtype=jnp.uint32
        # )

        # number_of_integration_steps = 100  # 0.050
        number_of_integration_steps = 10  # 0.010

        # Stepping logic
        with simulator.editable(validate=True) as simulator:
            # simulator, _ = simulator.step_over_horizon_plain(
            simulator, _ = simulator.step_over_horizon(
                horizon_steps=number_of_integration_steps,
                clear_inputs=False,
                callback_handler=SetTorquesOverHorizon(),
            )

        # Return the new environment state (updated SimulatorData)
        return simulator.data

    def observation(self, state: StateType) -> ObsType:
        """"""

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=True) as simulator:
            simulator.data = state
            model = simulator.get_model("ant")

        # Compute the normalized gravity projection in the body frame
        W_R_B = model.base_orientation(dcm=True)
        # W_gravity = state.simulator.gravity()
        W_gravity = self.jaxsim.gravity()
        B_gravity = W_R_B.T @ (W_gravity / jnp.linalg.norm(W_gravity))

        # Build the observation from the state
        return AntObservation.build(
            base_height=model.base_position()[2],
            gravity_projection=B_gravity,
            joint_positions=model.joint_positions(),
            joint_velocities=model.joint_velocities(),
            base_linear_velocity=model.base_velocity()[0:3],
            base_angular_velocity=model.base_velocity()[3:6],
            contact_state=model.in_contact(
                link_names=[
                    name
                    for name in model.link_names()
                    if name.startswith("leg_") and name.endswith("_lower")
                ]
            )
            # contact_state=jnp.array(
            #     [
            #         model.get_link(name).in_contact()
            #         for name in model.link_names()
            #         if name.startswith("leg_") and name.endswith("_lower")
            #     ],
            #     dtype=bool,
            # ),
        )

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """"""

        with self.jaxsim.editable(validate=True) as simulator_pre:
            simulator_pre.data = state
            model_pre = simulator_pre.get_model("ant")

        with self.jaxsim.editable(validate=True) as simulator_next:
            simulator_next.data = next_state
            model_next = simulator_next.get_model("ant")

        W_p_B_pre = model_pre.base_position()
        W_p_B_next = model_next.base_position()

        v_WB = (W_p_B_next - W_p_B_pre) / simulator_pre.dt()
        terminal = self.terminal(state=state)

        reward = 0.0
        reward += 1.0 * (1.0 - jnp.array(terminal, dtype=float))  # alive
        reward += 100.0 * v_WB[0]  # forward velocity
        # reward += 1.0 * model_next.in_contact(
        #     link_names=[
        #         name
        #         for name in model_next.link_names()
        #         if name.startswith("leg_") and name.endswith("_lower")
        #     ]
        # ).any().astype(
        #     float
        # )  # contact status
        reward -= 0.1 * jnp.linalg.norm(action) / action.size  # control cost

        return reward

    def terminal(self, state: StateType) -> TerminalType:
        """"""

        # Get the current observation
        observation = self.observation(state=state)

        # base_too_high = (
        #     observation.base_height >= self.observation_space.high.base_height
        # )

        no_feet_in_contact = jnp.where(observation.contact_state.any(), False, True)

        # The state is terminal if the observation is outside is space
        # return jax.lax.select(
        #     pred=self.observation_space.contains(x=observation),
        #     on_true=False,
        #     on_false=True,
        # )
        # return jnp.array([base_too_high, no_feet_in_contact]).any()
        return no_feet_in_contact

    # =========
    # Rendering
    # =========

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """Show the state."""

        model_name = "ant"

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=False) as simulator:
            simulator.data = state
            model = simulator.get_model(model_name=model_name)

        # Insert the model lazily in the visualizer if it is not already there
        if model_name not in render_state.world._meshcat_models.keys():
            from rod.urdf.exporter import UrdfExporter

            urdf_string = UrdfExporter.sdf_to_urdf_string(
                sdf=rod.Sdf(
                    version="1.7",
                    model=model.physics_model.description.extra_info["sdf_model"],
                ),
                pretty=True,
                gazebo_preserve_fixed_joints=False,
            )

            meshcat_viz_name = render_state.world.insert_model(
                model_description=urdf_string, is_urdf=True, model_name=None
            )

            render_state._jaxsim_to_meshcat_viz_name[model_name] = meshcat_viz_name

        # Check that the model is in the visualizer
        if (
            not render_state._jaxsim_to_meshcat_viz_name[model_name]
            in render_state.world._meshcat_models.keys()
        ):
            raise ValueError(f"The '{model_name}' model is not in the meshcat world")

        # Update the model in the visualizer
        render_state.world.update_model(
            model_name=render_state._jaxsim_to_meshcat_viz_name[model_name],
            joint_names=model.joint_names(),
            joint_positions=model.joint_positions(),
            base_position=model.base_position(),
            base_quaternion=model.base_orientation(dcm=False),
        )

        return render_state, np.empty(0)

    def render_init(self, open_gui: bool = False, **kwargs) -> RenderStateType:
        """Initialize the render state."""

        meshcat_viz_state = MeshcatVizRenderState()

        if open_gui:
            meshcat_viz_state.open_window_in_process()

        return meshcat_viz_state

    def render_close(self, render_state: RenderStateType) -> None:
        """Close the render state."""

        render_state.world.close()

        if render_state._gui_process is not None:
            render_state._gui_process.terminate()
            render_state._gui_process.close()

    # TODO: fare classe generica che dato un JaxSim visualizza tutti i modelli
    # -> vettorizzato?? metterlo dentro JaxSIm? Fare classe nuova in jaxsim?
    # def update_meshcat_world(
    #     self, world: "MeshcatWorld", state: StateType  # TODO: come gestire lo stato??
    # ) -> "MeshcatWorld":
    #     """"""
    #
    #     # Initialize the simulator with the environment state (containing SimulatorData)
    #     # and get the simulated model
    #     with self.jaxsim.editable(validate=False) as simulator:
    #         simulator.data = state
    #         model = simulator.get_model("ant")
    #
    #     # Add the model to the world if not already present
    #     if "ant" not in world._meshcat_models.keys():
    #         _ = world.insert_model(
    #             model_description=(
    #                 pathlib.Path.home()
    #                 / "git"
    #                 / "jaxsim"
    #                 / "examples"
    #                 / "resources"
    #                 / "ant.sdf"
    #             ),
    #             is_urdf=False,
    #             model_name="ant",
    #         )
    #
    #     # Update the model
    #     world.update_model(
    #         model_name="ant",
    #         joint_names=model.joint_names(),
    #         joint_positions=model.joint_positions(),
    #         base_position=model.base_position(),
    #         base_quaternion=model.base_orientation(dcm=False),
    #     )
    #
    #     return world


class AntReachTargetEnvV0(JaxEnv):
    """"""

    def __init__(self, render_mode: str | None = None, **kwargs: Any) -> None:
        """"""

        from jaxgym.wrappers.jax import (
            ClipActionWrapper,
            FlattenSpacesWrapper,
            JaxTransformWrapper,
            TimeLimit,
        )

        func_env = AntReachTargetFuncEnvV0()

        func_env_wrapped = func_env
        func_env_wrapped = TimeLimit(
            env=func_env_wrapped, max_episode_steps=5_000
        )  # TODO
        func_env_wrapped = ClipActionWrapper(env=func_env_wrapped)
        func_env_wrapped = FlattenSpacesWrapper(env=func_env_wrapped)
        func_env_wrapped = JaxTransformWrapper(env=func_env_wrapped, function=jax.jit)

        super().__init__(
            func_env=func_env_wrapped,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class AntReachTargetVectorEnvV0(JaxVectorEnv):
    """"""

    metadata = dict()

    def __init__(
        self,
        # func_env: JaxDataclassEnv[
        #     StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
        # ],
        num_envs: int,
        render_mode: str | None = None,
        # max_episode_steps: int = 5_000,
        jit_compile: bool = True,
        **kwargs,
    ) -> None:
        """"""

        print("+++", kwargs)

        env = AntReachTargetFuncEnvV0()

        # Vectorize the environment.
        # Note: it automatically wraps the environment in a TimeLimit wrapper.
        super().__init__(
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=5_000,  # TODO
            jit_compile=jit_compile,
        )

        # from jaxgym.vector.jax import FlattenSpacesVecWrapper
        #
        # vec_env_wrapped = FlattenSpacesVecWrapper(env=vec_env)


if __name__ == "__main__":
    """Stable Baselines"""

    from typing import Optional

    from jaxgym.wrappers.jax import (
        FlattenSpacesWrapper,
        JaxTransformWrapper,
        TimeLimit,
        ToNumPyWrapper,
    )

    def make_jax_env(
        max_episode_steps: Optional[int] = 500, jit: bool = True
    ) -> JaxEnv:
        """"""

        # TODO: single env -> time limit with stable_baselines?

        if max_episode_steps in {None, 0}:
            env = AntReachTargetFuncEnvV0()
        else:
            env = TimeLimit(
                env=AntReachTargetFuncEnvV0(), max_episode_steps=max_episode_steps
            )

        return JaxEnv(
            func_env=ToNumPyWrapper(
                env=FlattenSpacesWrapper(env=env)
                if not jit
                else JaxTransformWrapper(
                    function=jax.jit,
                    env=FlattenSpacesWrapper(env=env),
                ),
            ),
            render_mode="meshcat_viz",
        )

    env = make_jax_env(max_episode_steps=5, jit=False)

    obs, state_info = env.reset(seed=0)
    _ = env.render()
    raise
    for _ in range(5):
        action = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(action=action)
        obs, reward, terminated, truncated, info = env.step(
            action=jnp.zeros_like(action)
        )

    # =========
    # Visualize
    # =========

    visualize = False

    if visualize:
        rollout_visualizer = visualizer(env=lambda: make_jax_env(1_000), policy=model)

        import time

        time.sleep(3)
        rollout_visualizer(None)
