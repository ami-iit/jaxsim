import pathlib
from typing import Any, ClassVar

import jax.numpy as jnp
import jax.random
import jax_dataclasses
import numpy as np
import numpy.typing as npt

import jaxgym.jax.pytree_space as spaces
import jaxsim.typing as jtp
from jaxgym.envs.ant import MeshcatVizRenderState
from jaxgym.jax import JaxDataclassEnv, JaxEnv
from jaxgym.vector.jax import JaxVectorEnv
from jaxsim import JaxSim, logging
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim.simulation.simulator import SimulatorData, VelRepr
from jaxsim.utils import JaxsimDataclass, Mutability


@jax_dataclasses.pytree_dataclass
class CartpoleObservation(JaxsimDataclass):
    """Observation of the CartPole environment."""

    linear_pos: jtp.Float
    linear_vel: jtp.Float

    pivot_pos: jtp.Float
    pivot_vel: jtp.Float

    @staticmethod
    def build(
        linear_pos: jtp.Float,
        linear_vel: jtp.Float,
        pivot_pos: jtp.Float,
        pivot_vel: jtp.Float,
    ) -> "CartpoleObservation":
        """"""

        return CartpoleObservation(
            linear_pos=jnp.array(linear_pos, dtype=float),
            linear_vel=jnp.array(linear_vel, dtype=float),
            pivot_pos=jnp.array(pivot_pos, dtype=float),
            pivot_vel=jnp.array(pivot_vel, dtype=float),
        )


StateType = SimulatorData
ActType = jnp.ndarray
ObsType = CartpoleObservation
RewardType = float | jnp.ndarray
TerminalType = bool | jnp.ndarray
RenderStateType = MeshcatVizRenderState


@jax_dataclasses.pytree_dataclass
class CartpoleSwingUpFuncEnvV0(
    JaxDataclassEnv[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    ]
):
    """CartPole environment implementing a swing-up task."""

    name: ClassVar = jax_dataclasses.static_field(
        default="CartpoleSwingUpFunctionalEnvV0"
    )

    # Store an instance of the JaxSim simulator.
    # It gets initialized with SimulatorData with a functional approach.
    _simulator: JaxSim = jax_dataclasses.field(default=None)

    def __post_init__(self) -> None:
        """Environment initialization."""

        # Dummy initialization (not needed here)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            _ = self.jaxsim
        # _ = self.initial(rng=jax.random.PRNGKey(seed=0))

        # Create the action space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._action_space = spaces.PyTree(
                low=jnp.array(-50.0, dtype=float), high=jnp.array(50.0, dtype=float)
            )

        low = CartpoleObservation.build(
            linear_pos=-2.4,
            linear_vel=-10.0,
            pivot_pos=-jnp.pi,
            pivot_vel=-4 * jnp.pi,
        )

        high = CartpoleObservation.build(
            linear_pos=2.4,
            linear_vel=10.0,
            pivot_pos=jnp.pi,
            pivot_vel=4 * jnp.pi,
        )

        # Create the observation space (static attribute)
        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            self._observation_space = spaces.PyTree(low=low, high=high)

    @property
    def jaxsim(self) -> JaxSim:
        """"""

        if self._simulator is not None:
            return self._simulator

        # T = 0.010
        # dt = 0.001
        T = 0.050
        dt = 0.000_500

        # Create the jaxsim simulator
        simulator = JaxSim.build(
            # step_size=0.001,
            # steps_per_run=10,
            step_size=dt,
            steps_per_run=int(T / dt),
            velocity_representation=VelRepr.Inertial,
            integrator_type=IntegratorType.EulerSemiImplicit,
            simulator_data=SimulatorData(gravity=jnp.array([0, 0, -10.0])),
        ).mutable(mutable=True, validate=False)

        # Get the SDF path
        model_urdf_path = (
            pathlib.Path.home()
            / "git"
            / "jaxsim"
            / "examples"
            / "resources"
            / "cartpole.urdf"
        )

        # Insert the model
        _ = simulator.insert_model_from_description(
            model_description=model_urdf_path, model_name="cartpole"
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
        initial_observation: CartpoleObservation = (
            self.observation_space.sample_with_key(key=rng)
        )

        with self.jaxsim.editable(validate=False) as simulator:
            # Reset the simulator and get the model
            simulator.reset(remove_models=False)
            model = simulator.get_model(model_name="cartpole")

            # Reset the joint positions
            model.reset_joint_positions(
                positions=0.9
                * jnp.array(
                    [initial_observation.linear_pos, initial_observation.pivot_pos]
                ),
                joint_names=["linear", "pivot"],
            )

            # Reset the joint velocities
            model.reset_joint_velocities(
                velocities=0.9
                * jnp.array(
                    [initial_observation.linear_vel, initial_observation.pivot_vel]
                ),
                joint_names=["linear", "pivot"],
            )

            # TODO: reset the joint velocities
            # logging.error("ZEROOO")
            # model.reset_joint_positions(positions=jnp.array([0, jnp.deg2rad(180.0)]))
            # model.reset_joint_velocities(velocities=jnp.array([0, 0.0]))

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

        # Stepping logic
        with simulator.editable(validate=True) as simulator:
            # Get the simulated model
            model = simulator.get_model(model_name="cartpole")

            # Zero all the inputs
            model.zero_input()

            # print(action)
            # action = action.squeeze()

            # Apply a linear force to the cart
            model.set_joint_generalized_force_targets(
                forces=jnp.atleast_1d(action), joint_names=["linear"]
            )

            # TODO: in multi-step -> reset action?
            # Or always one step and handle multi-steps with callbacks e.g. controllers?
            simulator.step(clear_inputs=False)

        # Return the new environment state (updated SimulatorData)
        return simulator.data

    def observation(self, state: StateType) -> ObsType:
        """"""

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=False) as simulator:
            simulator.data = state
            model = simulator.get_model("cartpole")

        # Extract the positions and velocities of the joints
        linear_pos, pivot_pos = model.joint_positions()
        linear_vel, pivot_vel = model.joint_velocities()

        # Build the observation from the state
        return CartpoleObservation.build(
            linear_pos=linear_pos,
            linear_vel=linear_vel,
            # Make sure that the pivot position is always in [-π, π]
            pivot_pos=jnp.arctan2(jnp.sin(pivot_pos), jnp.cos(pivot_pos)),
            pivot_vel=pivot_vel,
        )

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """"""

        # Get the current observation
        observation = self.observation(state=next_state)
        # observation = type(self).observation(self=self, state=next_state)

        # Compute the reward terms
        reward_alive = 1.0 - jnp.array(self.terminal(state=next_state), dtype=float)
        # type(self).terminal(self=self, state=next_state), dtype=float
        reward_pivot = jnp.cos(observation.pivot_pos)
        cost_action = jnp.sqrt(action.dot(action))
        cost_pivot_vel = jnp.sqrt(observation.pivot_vel**2)
        cost_linear_pos = jnp.abs(observation.linear_pos)

        reward = 0
        reward += reward_alive
        reward += reward_pivot
        reward -= 0.001 * cost_action
        reward -= 0.100 * cost_pivot_vel
        reward -= 0.500 * cost_linear_pos

        return reward

    def terminal(self, state: StateType) -> TerminalType:
        """"""

        # Get the current observation
        observation = self.observation(state=state)
        # observation = type(self).observation(self=self, state=state)

        # The state is terminal if the observation is outside is space
        return jax.lax.select(
            pred=self.observation_space.contains(x=observation),
            on_true=False,
            on_false=True,
        )

    # =========
    # Rendering
    # =========

    # =========
    # Rendering
    # =========

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """Show the state."""

        model_name = "cartpole"

        # Initialize the simulator with the environment state (containing SimulatorData)
        # and get the simulated model
        with self.jaxsim.editable(validate=False) as simulator:
            simulator.data = state
            model = simulator.get_model(model_name=model_name)

        # Insert the model lazily in the visualizer if it is not already there
        if model_name not in render_state.world._meshcat_models.keys():
            import rod
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

        # Initialize the render state
        meshcat_viz_state = MeshcatVizRenderState()

        if open_gui:
            meshcat_viz_state.open_window_in_process()

        return meshcat_viz_state

    def render_close(self, render_state: RenderStateType) -> None:
        """Close the render state."""

        render_state.close()


class CartpoleSwingUpEnvV0(JaxEnv):
    """"""

    def __init__(self, render_mode: str | None = None, **kwargs: Any) -> None:
        """"""

        from jaxgym.wrappers.jax import (
            ClipActionWrapper,
            FlattenSpacesWrapper,
            JaxTransformWrapper,
            TimeLimit,
        )

        func_env = CartpoleSwingUpFuncEnvV0()

        func_env_wrapped = func_env
        func_env_wrapped = TimeLimit(env=func_env_wrapped, max_episode_steps=5_000)
        func_env_wrapped = ClipActionWrapper(env=func_env_wrapped)
        func_env_wrapped = FlattenSpacesWrapper(env=func_env_wrapped)
        func_env_wrapped = JaxTransformWrapper(env=func_env_wrapped, function=jax.jit)

        super().__init__(
            func_env=func_env_wrapped,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class CartpoleSwingUpVectorEnvV0(JaxVectorEnv):
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

        # print("+++", kwargs)

        env = CartpoleSwingUpFuncEnvV0()

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


if __name__ == "__main__REGISTER":
    """"""

    import gymnasium as gym

    import jaxgym.envs

    gym.envs.registry.keys()

    #
    #
    #

    env = gym.make("CartpoleSwingUpEnv-V0")
    env.spec.pprint(print_all=True)

    #
    #
    #

    vec_env = gym.make_vec(
        "CartpoleSwingUpEnv-V0", num_envs=2, vectorization_mode="custom"
    )
    vec_env.spec.pprint(print_all=True)

    from jaxgym.vector.jax.wrappers import FlattenSpacesVecWrapper

    vec_env_wrapped = FlattenSpacesVecWrapper(env=vec_env)


if __name__ == "__main__+":
    """"""

    # env = CartpoleFunctionalEnvV0()
    # state = env.initial(rng=jax.random.PRNGKey(0))
    # action = env.action_space.sample(key=jax.random.PRNGKey(1))

    key = jax.random.PRNGKey(0)
    num = 1000

    # TODO next week:
    # - this is ok
    # - write Env wrapper for autoreset / check gymnasium -> lambda for get_action from pytorch?
    # - figure out what the info dicts of gymnasium are
    # - decide how to perform training loop -> rl algos from where (jax-based or pytorch)?
    #
    # Pytorch is ok if we sample in parallel only a single step (e.g. on thousands of envs)

    # from jaxgym.functional.wrappers.transform import TransformWrapper
    # from jaxgym.functional.wrappers.jax.time_limit import TimeLimit
    #
    # env = CartpoleSwingUpFunctionalEnvV0()
    # env = TimeLimit(env=env, max_episode_steps=100)
    # # vec_env.transform(func=jax.vmap)
    # # vec_env.transform(func=jax.jit)
    # vec_env = TransformWrapper(env=env, function=jax.vmap)
    # vec_env = TransformWrapper(env=vec_env, function=jax.jit)
    # states = vec_env.initial(rng=jax.random.split(key, num=num))
    # _ = vec_env.observation(state=states)
    # action = vec_env.action_space.sample(key=jax.random.split(key, num=1).squeeze())
    # actions = jnp.repeat(action, repeats=num, axis=0)
    # next_states = vec_env.transition(state=states, action=actions)
    # reward = vec_env.reward(state=states, action=actions, next_state=next_states)
    # infos = vec_env.step_info(state=states, action=actions, next_state=next_states)

    # from jaxgym.functional.jax.vector import JaxVectorEnv
    # from jaxgym.functional.jax.time_limit import TimeLimit
    # from jaxgym.functional.wrappers.transform import TransformWrapper
    # from jaxgym.functional.core import FuncWrapper
    #
    # env = CartpoleSwingUpFunctionalEnvV0()
    #
    # env_wrapped = TimeLimit(env=env, max_episode_steps=100)
    # env_wrapped = TransformWrapper(env=env_wrapped, function=jax.jit)
    # # CartpoleSwingUpFunctionalEnvV0.transform(self=env_wrapped, func=jax.jit)
    # state = env_wrapped.initial(rng=key)
    # _ = env_wrapped.observation(state=state)
    # action = env_wrapped.action_space.sample(key=key)
    # next_state = env_wrapped.transition(state=state, action=action)
    # reward = env_wrapped.reward(state=state, action=action, next_state=next_state)
    # info = env_wrapped.step_info(state=state, action=action, next_state=next_state)
    # next_state = env_wrapped.transition(state=next_state, action=action)

    from jaxgym.functional.jax.time_limit import TimeLimit
    from jaxgym.functional.jax.transform import JaxTransformWrapper
    from jaxgym.functional.jax.vector import JaxVectorEnv
    from jaxgym.functional.wrappers.transform import TransformWrapper

    env = CartpoleSwingUpFunctionalEnvV0()
    # env = TimeLimit(env=env, max_episode_steps=3)
    num_envs = 2
    vec_env = JaxVectorEnv(
        func_env=env, num_envs=num_envs, max_episode_steps=4, jit_compile=True
    )
    observations, state_infos = vec_env.reset()

    actions = vec_env.action_space.sample(key=key)
    # actions = jnp.repeat(jnp.atleast_2d(action).T, repeats=num_envs, axis=1).T

    # TODO: the output dict misses the final observation when truncated
    # final_observation | final_info
    _ = vec_env.step(action=actions)

    # self = vec_env
    # env = self.func_env
    # states = self.states
    # keys_1 = self.subkey(num=self.num_envs)
    # keys_2 = self.subkey(num=self.num_envs)
    # states, _ = jax.jit(JaxVectorEnv.step_autoreset_func)(
    #     env, states, actions, keys_1, keys_2
    # )

    # @jax.jit
    # def split(key: jax.random.PRNGKeyArray, num: int) -> jax.random.PRNGKeyArray:
    #     return jax.random.split(key=key, num=num)
    #
    # _ = split(key, 5)

    # _ = vec_env.func_env.transition(state=vec_env.states, action=actions)
    # _ = vec_env.step(action=actions)

    # observation = env.observation(state=state)
    # terminal = env.terminal(state=state)
    # reward = env.reward(state=state, action=action)
    # next_state = env.transition(state=state, action=action, rng=jax.random.PRNGKey(2))

    # jax.tree_util.tree_structure(state)
    # jax.tree_util.tree_structure(next_state)
    #
    # jax.tree_util.tree_leaves(state)
    # jax.tree_util.tree_leaves(next_state)

    # with env.jaxsim.editable(validate=True) as simulator:
    #     simulator.data = next_state
