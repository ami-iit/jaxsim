import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union

import gymnasium as gym
import jax.random
import numpy as np
import stable_baselines3
import numpy.typing as npt
from gymnasium.experimental.vector.vector_env import VectorWrapper
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env as vec_env_sb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import jaxsim.typing as jtp
from jaxgym.envs.ant import AntReachTargetFuncEnvV0
from jaxgym.envs.cartpole import CartpoleSwingUpFuncEnvV0
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper, JaxEnv, PyTree
from jaxgym.vector.jax import FlattenSpacesVecWrapper, JaxVectorEnv
from jaxgym.wrappers.jax import (  # TimeLimitStableBaselines,
    ClipActionWrapper,
    FlattenSpacesWrapper,
    JaxTransformWrapper,
    NaNHandlerWrapper,
    SquashActionWrapper,
    TimeLimit,
    ToNumPyWrapper,
)

# Full cartpole example with collection loop
#
# -> validate with visualization
# -> move to pytorch later
# -> study TensorDict -> create wrapper -> use PPO
# -> Check JaxToTorch wrapper and adapt it to work for JaxVectorEnv
# -> alternatively, wait for stable_baselines3 + gymnasium (open issue)

# For TensorDict, check:
#
# -> create docker image with torch 2.0
# - EnvBase https://github.com/pytorch/rl/blob/main/torchrl/envs/common.py#L120 (has batch_size -> maybe vectorized?)
# - torchrl.collectors
# - torchrl.envs.libs.gym.GymWrapper
# - torchrl.envs.libs.brax.BraxWrapper

# TODO: JaxSimEnv with render support?


class CustomVecEnvSB(vec_env_sb.VecEnv):
    """"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        jax_vector_env: JaxVectorEnv | VectorWrapper,
        # num_envs: int,
        # observation_space: spaces.Space,
        # action_space: spaces.Space,
        # render_mode: Optional[str] = None,
    ) -> None:
        """"""

        if not isinstance(jax_vector_env.unwrapped, JaxVectorEnv):
            raise TypeError(type(jax_vector_env))

        self.jax_vector_env = jax_vector_env

        single_env_action_space: PyTree = jax_vector_env.unwrapped.single_action_space

        single_env_observation_space: PyTree = (
            jax_vector_env.unwrapped.single_observation_space
        )

        super().__init__(
            num_envs=self.jax_vector_env.num_envs,
            action_space=single_env_action_space.to_box(),
            observation_space=single_env_observation_space.to_box(),
            render_mode=None,
        )

        self.actions = np.zeros_like(self.jax_vector_env.action_space.sample())

        # Initialize the RNG seed
        self._seed = None
        self.seed()

    def reset(self) -> vec_env_sb.base_vec_env.VecEnvObs:
        """"""

        observations, state_infos = self.jax_vector_env.reset(seed=self._seed)
        return np.array(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def tree_inverse_transpose(pytree: jtp.PyTree, batch_size: int) -> List[jtp.PyTree]:
        """"""

        return [
            jax.tree_util.tree_map(lambda leaf: leaf[i], pytree)
            for i in range(batch_size)
        ]

    def step_wait(self) -> vec_env_sb.base_vec_env.VecEnvStepReturn:
        """"""

        (
            observations,
            rewards,
            terminals,
            truncated,
            step_infos,
        ) = self.jax_vector_env.step(actions=self.actions)

        done = np.logical_or(terminals, truncated)

        # list_of_step_infos = [
        #     jax.tree_util.tree_map(lambda l: l[i], step_infos)
        #     for i in range(self.jax_vector_env.num_envs)
        # ]

        list_of_step_infos = self.tree_inverse_transpose(
            pytree=step_infos, batch_size=self.jax_vector_env.num_envs
        )

        # def pytree_to_numpy(pytree: jtp.PyTree) -> jtp.PyTree:
        #     return jax.tree_util.tree_map(lambda leaf: np.array(leaf), pytree)
        #
        # list_of_step_infos_numpy = [pytree_to_numpy(pt) for pt in list_of_step_infos]

        list_of_step_infos_numpy = [
            ToNumPyWrapper.pytree_to_numpy(pytree=pt) for pt in list_of_step_infos
        ]

        return (
            np.array(observations),
            np.array(rewards),
            np.array(done),
            list_of_step_infos_numpy,
        )

    def close(self) -> None:
        return self.jax_vector_env.close()

    def get_attr(
        self, attr_name: str, indices: vec_env_sb.base_vec_env.VecEnvIndices = None
    ) -> List[Any]:
        raise NotImplementedError

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
    ) -> None:
        raise NotImplementedError

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        raise NotImplementedError

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: vec_env_sb.base_vec_env.VecEnvIndices = None,
    ) -> List[bool]:
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """"""

        if seed is None:
            seed = np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")

        if np.array(seed, dtype="uint32") != np.array(seed):
            raise ValueError(f"seed must be compatible with 'uint32' casting")

        self._seed = seed
        return [seed]

        # _ = self.jax_vector_env.reset(seed=seed)
        # return [None]


def make_vec_env_stable_baselines(
    jax_dataclass_env: JaxDataclassEnv | JaxDataclassWrapper,
    n_envs: int = 1,
    seed: Optional[int] = None,
    # monitor_dir: Optional[str] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    #
    # env_id: Union[str, Callable[..., gym.Env]],
    # # n_envs: int = 1,
    # # seed: Optional[int] = None,
    # start_index: int = 0,
    # monitor_dir: Optional[str] = None,
    # wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    # env_kwargs: Optional[Dict[str, Any]] = None,
    # vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    # vec_env_kwargs: Optional[Dict[str, Any]] = None,
    # monitor_kwargs: Optional[Dict[str, Any]] = None,
    # wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> vec_env_sb.VecEnv:
    """"""

    env = jax_dataclass_env

    vec_env_kwargs = vec_env_kwargs if vec_env_kwargs is not None else dict()

    # Vectorize the environment.
    # Note: it automatically wraps the environment in a TimeLimit wrapper.
    # Note: the space must be PyTree.
    vec_env = JaxVectorEnv(
        func_env=env,
        num_envs=n_envs,
        **vec_env_kwargs,
    )

    # Flatten the PyTree spaces to regular Box spaces
    vec_env = FlattenSpacesVecWrapper(env=vec_env)

    # if seed is not None:
    # _ = vec_env.reset(seed=seed)

    vec_env_sb = CustomVecEnvSB(jax_vector_env=vec_env)

    if seed is not None:
        _ = vec_env_sb.seed(seed=seed)

    return vec_env_sb


def visualizer(
    env: JaxEnv | Callable[[None], JaxEnv], policy: BaseAlgorithm
) -> Callable[[Optional[int]], None]:
    """"""

    import numpy as np
    import rod
    from loop_rate_limiters import RateLimiter
    from meshcat_viz import MeshcatWorld

    from jaxsim import JaxSim

    # Open the visualizer
    world = MeshcatWorld()
    world.open()

    # Create the JaxSim environment and get the simulator
    env = env() if isinstance(env, Callable) else env
    sim: JaxSim = env.unwrapped.func_env.unwrapped.jaxsim

    # Extract the SDF string from the simulated model
    jaxsim_model = sim.get_model(model_name="cartpole")
    rod_model = jaxsim_model.physics_model.description.extra_info["sdf_model"]
    rod_sdf = rod.Sdf(model=rod_model, version="1.7")
    sdf_string = rod_sdf.serialize(pretty=True)

    # Insert the model from a URDF/SDF resource
    model_name = world.insert_model(model_description=sdf_string, is_urdf=False)

    # Create the visualization function
    def rollout(seed: Optional[int] = None) -> None:
        """"""

        # Reset the environment
        observation, state_info = env.reset(seed=seed)

        # Initialize the model state with the initial observation
        world.update_model(
            model_name=model_name,
            joint_names=["linear", "pivot"],
            joint_positions=np.array([observation[0], observation[2]]),
        )

        rtf = 1.0
        down_sampling = 1
        rate = RateLimiter(frequency=float(rtf / (sim.dt() * down_sampling)))

        done = False

        # Visualization loop
        while not done:
            action, _ = policy.predict(observation=observation, deterministic=True)
            print(action)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            world.update_model(
                model_name=model_name,
                joint_names=["linear", "pivot"],
                joint_positions=np.array([observation[0], observation[2]]),
            )

            print(done)
            rate.sleep()

        print("done")

    return rollout


# ============
# ENVIRONMENTS
# ============

# TODO:
#
# - Initialize ANT already in contact -> otherwise it jumps around (idle after falling?)
# - Tune spring for joint limits and joint friction
# - wrapper to squash action space to [-1, 1]


def make_jax_env_cartpole(
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = 500,
) -> JaxEnv:
    """"""

    # TODO: single env -> time limit with stable_baselines?

    import os

    import torch

    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""

    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    env = NaNHandlerWrapper(env=CartpoleSwingUpFuncEnvV0())

    if max_episode_steps is not None:
        env = TimeLimit(env=env, max_episode_steps=max_episode_steps)

    return JaxEnv(
        render_mode=render_mode,
        func_env=ToNumPyWrapper(
            env=JaxTransformWrapper(
                function=jax.jit,
                env=FlattenSpacesWrapper(
                    env=ClipActionWrapper(
                        env=SquashActionWrapper(env=env),
                    )
                ),
            )
        ),
    )


def make_jax_env_ant(
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = 1_000,
) -> JaxEnv:
    """"""

    # TODO: single env -> time limit with stable_baselines?

    import os

    import torch

    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""

    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    env = NaNHandlerWrapper(env=AntReachTargetFuncEnvV0())

    if max_episode_steps is not None:
        env = TimeLimit(env=env, max_episode_steps=max_episode_steps)

    return JaxEnv(
        render_mode=render_mode,
        func_env=ToNumPyWrapper(
            env=JaxTransformWrapper(
                function=jax.jit,
                env=FlattenSpacesWrapper(
                    env=ClipActionWrapper(
                        env=SquashActionWrapper(env=env),
                    )
                ),
            )
        ),
    )


# =============================
# Test JaxVecEnv vs DummyVecEnv
# =============================

if __name__ == "__main__?":
    """"""

    max_episode_steps = 200
    func_env = NaNHandlerWrapper(env=CartpoleSwingUpFuncEnvV0())

    if max_episode_steps is not None:
        func_env = TimeLimit(env=func_env, max_episode_steps=max_episode_steps)

    func_env = (
        # ToNumPyWrapper(env=
        # env=JaxTransformWrapper(
        #     function=jax.jit,
        # env=FlattenSpacesWrapper(
        ClipActionWrapper(
            env=SquashActionWrapper(env=func_env),
        )
        # ),
        # ),
        # )
    )

    vec_env = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        n_envs=10,
        seed=42,
        vec_env_kwargs=dict(
            # max_episode_steps=5_000,
            jit_compile=True,
        ),
    )

    # Seed the environment
    # vec_env.seed(seed=42)

    # Reset the environment.
    # This has to be done only once since the vectorized environment supports autoreset.
    observations = vec_env.reset()

    # Initialize a random policy
    random_policy = lambda obs: vec_env.jax_vector_env.action_space.sample()

    for _ in range(1):
        # Sample random actions
        actions = random_policy(observations)

        # Step the environment
        observations, rewards, dones, step_infos = vec_env.step(actions=actions)

        print(observations, rewards, dones, step_infos)
        # print()
        # print(dones)


def evaluate(
    env: gym.Env | Callable[[...], gym.Env],
    num_episodes: int = 1,
    seed: int | None = None,
    render: bool = False,
    policy: Callable[[npt.NDArray], npt.NDArray] | None = None,
) -> None:
    """"""

    # Create the environment if a callable is passed
    env = env if isinstance(env, gym.Env) else env()

    # Initialize a random policy if none is passed
    policy = policy if policy is not None else lambda obs: env.action_space.sample()

    episodes_length = []
    cumulative_rewards = []

    for e in range(num_episodes):
        # Reset the environment
        observation, step_info = env.reset(seed=seed)

        # Initialize done flag
        done = False

        # Render the environment
        if render:
            env.render()

        episodes_length += [0]
        cumulative_rewards += [0]

        # Evaluation loop
        while not done:
            # Increase episode length counter
            episodes_length[-1] += 1

            # Predict the action
            action = policy(observation)

            # Step the environment
            observation, reward, terminal, truncated, step_info = env_eval.step(
                action=action
            )

            # Determine if the episode is done
            done = terminal or truncated

            # Store the cumulative reward
            cumulative_rewards[-1] += reward

            # Render the environment
            if render:
                _ = env_eval.render()

    print("ep_len_mean\t", np.array(episodes_length).mean())
    print("ep_rew_mean\t", np.array(cumulative_rewards).mean())


# Train with SB
if __name__ == "__main__cartpole_cpu_vec_env":
    """"""

    max_episode_steps = 200
    func_env = NaNHandlerWrapper(env=CartpoleSwingUpFuncEnvV0())

    if max_episode_steps is not None:
        func_env = TimeLimit(env=func_env, max_episode_steps=max_episode_steps)

    func_env = ClipActionWrapper(
        env=SquashActionWrapper(env=func_env),
    )

    vec_env_sb = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        n_envs=10,
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    import torch as th

    model = PPO(
        "MlpPolicy",
        env=vec_env_sb,
        # n_steps=2048,
        n_steps=256,  # in the vector env -> real ones are x10
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        normalize_advantage=True,
        # target_kl=0.010,
        target_kl=0.025,
        verbose=1,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 512], vf=[512, 512]),
            log_std_init=np.log(0.05),
            # squash_output=True,
        ),
    )

    print(model.policy)

    # Create the evaluation environment
    env_eval = make_jax_env_cartpole(
        render_mode="meshcat_viz",
        max_episode_steps=500,
    )

    for _ in range(1):
        # Train the model
        model = model.learn(total_timesteps=50_000, progress_bar=False)

        # Create the policy closure
        policy = lambda observation: model.policy.predict(
            observation=observation, deterministic=True
        )[0]

        # Evaluate the policy
        print("Evaluating...")
        evaluate(
            env=env_eval,
            num_episodes=10,
            seed=None,
            render=True,
            policy=policy,
        )

# Train with SB
if __name__ == "__main__cartpole_gpu_vec_env":
    """"""

    max_episode_steps = 200
    func_env = NaNHandlerWrapper(env=CartpoleSwingUpFuncEnvV0())

    if max_episode_steps is not None:
        func_env = TimeLimit(env=func_env, max_episode_steps=max_episode_steps)

    func_env = ClipActionWrapper(
        env=SquashActionWrapper(env=func_env),
    )

    vec_env_sb = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        # n_envs=10,
        n_envs=512,
        # n_envs=2048,  # TODO
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    import torch as th

    model = PPO(
        "MlpPolicy",
        env=vec_env_sb,
        # n_steps=2048,
        # n_steps=256,  # in the vector env -> real ones are x10
        n_steps=5,  # in the vector env -> real ones are x10
        batch_size=256,
        # batch_size=512,  # TODO
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        normalize_advantage=True,
        # target_kl=0.010,
        target_kl=0.025,
        verbose=1,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[512, 512], vf=[512, 512]),
            log_std_init=np.log(0.05),
            # squash_output=True,
        ),
    )

    print(model.policy)

    # Create the evaluation environment
    env_eval = make_jax_env_cartpole(
        render_mode="meshcat_viz",
        max_episode_steps=200,
    )

    for _ in range(10):
        # Train the model
        model = model.learn(total_timesteps=50_000, progress_bar=False)

        # Create the policy closure
        policy = lambda observation: model.policy.predict(
            observation=observation, deterministic=True
        )[0]

        # Evaluate the policy
        print("Evaluating...")
        evaluate(
            env=env_eval,
            num_episodes=10,
            seed=None,
            render=True,
            policy=policy,
        )

    # # Create the policy closure
    # policy = lambda observation: model.policy.predict(
    #     observation=observation, deterministic=True
    # )[0]
    #
    # evaluate(
    #     env=env_eval,
    #     num_episodes=10,
    #     seed=None,
    #     render=True,
    #     policy=policy,
    # )

    # =======================
    # Evaluation environments
    # =======================
    #
    # env_eval = make_jax_env_cartpole(render_mode="meshcat_viz", max_episode_steps=None)
    #
    # # observation, step_info = env_eval.reset(seed=42)
    # observation, step_info = env_eval.reset()
    #
    # # Initialize done flag
    # done = False
    #
    # env_eval.render()
    #
    # i = 0
    # cum_reward = 0.0
    #
    # while not done:
    #     i += 1
    #
    #     if i == 2000:
    #         done = True
    #
    #     # Sample a random action
    #     # action = 0.1* random_policy(env, observation)
    #     action, _ = model.policy.predict(observation=observation, deterministic=False)
    #
    #     # Step the environment
    #     observation, reward, terminal, truncated, step_info = env_eval.step(
    #         action=action
    #     )
    #
    #     print(reward)
    #     cum_reward += reward
    #
    #     # Render the environment
    #     _ = env_eval.render()
    #
    #     # print(observation, reward, terminal, truncated, step_info)
    #     # print(env.state)
    #
    # print("reward =", cum_reward)
    # print("episode length =", i)
    #
    # env_eval.close()


# Train with SB
if __name__ == "__main__ant_vec_gpu_env":
    """"""

    max_episode_steps = 1000
    func_env = NaNHandlerWrapper(env=AntReachTargetFuncEnvV0())

    if max_episode_steps is not None:
        func_env = TimeLimit(env=func_env, max_episode_steps=max_episode_steps)

    func_env = ClipActionWrapper(
        env=SquashActionWrapper(env=func_env),
    )

    vec_env_sb = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        # n_envs=10,
        # n_envs=2048,  # troppo -> JIT lungo
        # n_envs=100,
        # n_envs=1024,
        n_envs=2048,
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    %time _ = vec_env_sb.reset()
    %time _ = vec_env_sb.reset()
    actions = vec_env_sb.jax_vector_env.action_space.sample()
    %time _ = vec_env_sb.step(actions)
    %time _ = vec_env_sb.step(actions)

    import torch as th

    # TODO: se ogni reset c'e' 1 sec di sim -> mega lento perche' ci sara' sempre
    # un env che si sta resettando!

    model = PPO(
        "MlpPolicy",
        env=vec_env_sb,
        # n_steps=2048,
        # n_steps=512,  # in the vector env -> real ones are x10
        # n_steps=10,  # in the vector env -> real ones are x10
        n_steps=2,  # in the vector env -> real ones are x2048
        # batch_size=256,
        batch_size=1024,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        normalize_advantage=True,
        # target_kl=0.010,
        target_kl=0.025,
        verbose=1,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[2048, 1024], vf=[1024, 1024, 256]),
            # net_arch=dict(pi=[2048, 2048], vf=[2048, 1024, 512]),
            log_std_init=np.log(0.05),
            # squash_output=True,
        ),
    )

    print(model.policy)

    # Create the evaluation environment
    env_eval = make_jax_env_ant(
        render_mode="meshcat_viz",
        max_episode_steps=1000,
    )

    for _ in range(10):
        # Train the model
        model = model.learn(total_timesteps=500_000, progress_bar=False)

        # Create the policy closure
        policy = lambda observation: model.policy.predict(
            observation=observation, deterministic=True
        )[0]

        # Evaluate the policy
        print("Evaluating...")
        evaluate(
            env=env_eval,
            num_episodes=10,
            seed=None,
            render=True,
            policy=policy,
        )

    # evaluate(
    #     env=env_eval,
    #     num_episodes=10,
    #     seed=None,
    #     render=True,
    #     # policy=policy,
    # )

# =============
# RANDOM POLICY
# =============

# TODO: generate a JaxVecEnv and validate with DummyVecEnv from SB

if __name__ == "__main__+":
    """"""

    # Create the environment
    env = make_jax_env_ant(render_mode="meshcat_viz", max_episode_steps=None)
    # env = make_jax_env_cartpole(render_mode="meshcat_viz", max_episode_steps=10)

    # Reset the environment
    # observation, state_info = env.reset(seed=42)
    observation, state_info = env.reset()

    # Initialize a random policy
    random_policy = lambda env, obs: env.action_space.sample()

    # Initialize done flag
    done = False

    env.render()

    # s = env.state
    # env.func_env.env.transition(state=s, action=env.func_env.action_space.sample())
    #
    # with env.func_env.unwrapped.jaxsim.editable(validate=True) as sim:
    #     sim.data = env.state["env"]

    # import time
    #
    # time.sleep(2)

    i = 0
    cum_reward = 0.0

    while not done:
        i += 1

        if i == 2000:
            done = True

        # Sample a random action
        # action = 0.1* random_policy(env, observation)
        action, _ = model.policy.predict(observation=observation, deterministic=False)

        # Step the environment
        observation, reward, terminal, truncated, step_info = env.step(action=action)

        print(reward)
        cum_reward += reward

        # Render the environment
        _ = env.render()

        # print(observation, reward, terminal, truncated, step_info)
        # print(env.state)

    print(cum_reward)

    env.close()

# =================
# TRAINING PPO/TRPO
# =================

if __name__ == "__main__)":
    """Stable Baselines"""

    # Initialize properties
    # seed = 42

    # Create a single environment
    # func_env = CartpoleSwingUpFuncEnvV0()

    # env = JaxEnv(
    #     func_env=ToNumPyWrapper(
    #         env=JaxTransformWrapper(
    #             function=jax.jit,
    #             env=FlattenSpacesWrapper(env=CartpoleSwingUpFuncEnvV0()),
    #         )
    #     )
    # )

    # TODO: try with single env first?

    # env = JaxEnv(
    #     func_env=ToNumPyWrapper(
    #         env=JaxTransformWrapper(
    #             function=jax.jit,
    #             env=FlattenSpacesWrapper(
    #                 env=TimeLimit(env=func_env, max_episode_steps=1_000)
    #             ),
    #         )
    #     )
    # )

    # def make_env() -> gym.Env:
    # def make_jax_env(max_episode_steps: Optional[int] = 500) -> JaxEnv:
    #     """"""
    #
    #     # TODO: single env -> time limit with stable_baselines?
    #
    #     if max_episode_steps is None:
    #         env = CartpoleSwingUpFuncEnvV0()
    #     else:
    #         env = TimeLimit(
    #             env=CartpoleSwingUpFuncEnvV0(), max_episode_steps=max_episode_steps
    #         )
    #
    #     return JaxEnv(
    #         func_env=ToNumPyWrapper(
    #             env=JaxTransformWrapper(
    #                 function=jax.jit,
    #                 env=FlattenSpacesWrapper(env=env),
    #             )
    #         )
    #     )

    # check_env(env=env, warn=True, skip_render_check=True)

    # observation = env.reset()
    # action = env.action_space.sample()
    # observation, reward, terminated, truncate, info = env.step(action)
    # print(observation, reward, terminated, truncate, info)

    #
    #
    #

    # vec_env = make_vec_env_stable_baselines(
    #     jax_dataclass_env=func_env,
    #     n_envs=10,
    #     seed=42,
    #     vec_env_kwargs=dict(
    #         max_episode_steps=5_000,
    #         jit_compile=True,
    #     ),
    # )

    # vec_env.reset()

    vec_env = make_vec_env(
        # env_id=lambda: make_jax_env_cartpole(max_episode_steps=500),
        env_id=lambda: make_jax_env_ant(max_episode_steps=2_000),
        n_envs=10,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    import torch as th

    # ANT
    model = PPO(
        "MlpPolicy",
        env=vec_env,
        # n_steps=2048,
        n_steps=512,  # in the vector env -> real ones are x10
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.98,
        clip_range=0.1,
        normalize_advantage=True,
        # target_kl=0.010,
        target_kl=0.025,
        verbose=1,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[1024, 512], vf=[1024, 512]),
            # log_std_init=0.10,
            # log_std_init=2.5,
            log_std_init=np.log(0.25),
            # squash_output=True,
        ),
    )

    print(model.policy)
    model = model.learn(total_timesteps=200_000, progress_bar=False)
    # model = model.learn(total_timesteps=500_000, progress_bar=False)

    #
    # CARTPOLE
    #

    # TODO: with squash_output -> do I need to apply tanh to the action?
    # model = PPO(
    #     "MlpPolicy",
    #     env=vec_env,
    #     # n_steps=2048,
    #     n_steps=256,  # in the vector env -> real ones are x10
    #     batch_size=256,
    #     n_epochs=10,
    #     gamma=0.95,
    #     gae_lambda=0.9,
    #     clip_range=0.1,
    #     normalize_advantage=True,
    #     # target_kl=0.010,
    #     target_kl=0.025,
    #     verbose=1,
    #     learning_rate=0.000_300,
    #     policy_kwargs=dict(
    #         activation_fn=th.nn.ReLU,
    #         net_arch=dict(pi=[256, 256], vf=[256, 256]),
    #         log_std_init=0.05,
    #         # squash_output=True,
    #     ),
    # )

    model = TRPO(
        "MlpPolicy",
        env=vec_env,
        n_steps=256,  # in the vector env -> real ones are x10
        batch_size=1024,
        gamma=0.95,
        gae_lambda=0.95,
        normalize_advantage=True,
        target_kl=0.025,
        verbose=1,
        learning_rate=0.000_300,
        policy_kwargs=dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=0.05,  # TODO np.log()..., maybe 0.05 too large?
            # squash_output=True,
        ),
    )

    print(model.policy)
    model = model.learn(total_timesteps=500_000, progress_bar=False)
    # model.learn(total_timesteps=25_000)
    # model.save("ppo_cartpole")

    # # Vectorize the environment.
    # # Note: it automatically wraps the environment in a TimeLimit wrapper.
    # vec_env = JaxVectorEnv(
    #     func_env=env,
    #     num_envs=10,
    #     max_episode_steps=5_000,
    #     jit_compile=True,
    # )
    #
    # from jaxgym.vector.jax import FlattenSpacesVecWrapper
    #
    # vec_env = FlattenSpacesVecWrapper(env=vec_env)
    #
    # # test.reset(seed=0)
    # # test.step(actions=test.action_space.sample())

    # =========
    # Visualize
    # =========

    visualize = False

    if visualize:
        rollout_visualizer = visualizer(env=lambda: make_jax_env(1_000), policy=model)

        import time

        time.sleep(3)
        rollout_visualizer(None)


if __name__ == "__main___":
    """"""

    # Initialize properties
    seed = 42
    # num_envs = 3

    # Create a single environment
    func_env = CartpoleSwingUpFuncEnvV0()

    from jaxgym.jax.env import JaxEnv
    from jaxgym.wrappers.jax import (
        ClipActionWrapper,
        FlattenSpacesWrapper,
        JaxTransformWrapper,
    )

    func_env = ClipActionWrapper(env=func_env)
    func_env = FlattenSpacesWrapper(env=func_env)
    func_env = JaxTransformWrapper(env=func_env, function=jax.jit)

    # state = func_env.initial(rng=jax.random.PRNGKey(seed=seed))
    # action = func_env.action_space.sample()
    # func_env.transition(state, action)

    env = JaxEnv(func_env=func_env)

    observation, state_info = env.reset(seed=seed)


# TODO: specs kind of ok -> see bottom of cartpole
if __name__ == "__main___":
    """"""

    # Initialize properties
    seed = 42
    # num_envs = 3

    # Create a single environment
    env = CartpoleSwingUpFuncEnvV0()

    # Vectorize the environment.
    # Note: it automatically wraps the environment in a TimeLimit wrapper.
    vec_env = JaxVectorEnv(
        func_env=env,
        num_envs=1_000,
        max_episode_steps=10,
        jit_compile=True,
    )

    # FRIDAY:
    # from jaxgym.functional.jax.flatten_spaces import FlattenSpacesWrapper
    # test = FlattenSpacesWrapper(env=env)
    # test.transition(
    #     state=test.initial(rng=jax.random.PRNGKey(0)), action=test.action_space.sample()
    # )

    from jaxgym.vector.jax import FlattenSpacesVecWrapper

    test = FlattenSpacesVecWrapper(env=vec_env)
    test.reset(seed=0)
    test.step(actions=test.action_space.sample())

    # import cProfile
    # from pstats import SortKey
    #
    # with cProfile.Profile() as pr:
    #
    #     for _ in range(1000):
    #         _ = test.step(actions=test.action_space.sample())
    #
    #     pr.print_stats(sort=SortKey.CUMULATIVE)
    #
    # exit(0)

    # o = test.env.observation_space.sample()
    # test.env.observation_space.flatten_sample(o)

    # exit(0)
    # raise  # all good!
    # TODO: benchmark non-jit sections
    # TODO: benchmark from the code instead cmdline after jit compilation

    # Reset the environment.
    # This has to be done only once since the vectorized environment supports autoreset.
    observations, state_infos = vec_env.reset(seed=seed)

    # Initialize a random policy
    random_policy = lambda obs: vec_env.action_space.sample()

    for _ in range(1):
        # Sample random actions
        actions = random_policy(observations)

        # Step the environment
        observations, rewards, terminals, truncated, step_infos = vec_env.step(
            action=actions
        )

        print(observations, rewards, terminals, truncated, step_infos)
