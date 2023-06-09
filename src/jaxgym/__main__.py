import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import functools
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import jax.random
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import numpy.typing as npt
import stable_baselines3
from gymnasium.experimental.vector.vector_env import VectorWrapper
from sb3_contrib import TRPO
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env as vec_env_sb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

import jaxsim.typing as jtp
from jaxgym.envs.ant import AntReachTargetFuncEnvV0
from jaxgym.envs.cartpole import CartpoleSwingUpFuncEnvV0
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper, JaxEnv, PyTree
from jaxgym.vector.jax import FlattenSpacesVecWrapper, JaxVectorEnv
from jaxgym.wrappers.jax import (  # TimeLimitStableBaselines,
    ActionNoiseWrapper,
    ClipActionWrapper,
    FlattenSpacesWrapper,
    JaxTransformWrapper,
    NaNHandlerWrapper,
    SquashActionWrapper,
    TimeLimit,
    ToNumPyWrapper,
)

#
#
#


class MujocoModel:
    """"""

    def __init__(self, xml_path: pathlib.Path) -> None:
        """"""

        if not xml_path.exists():
            raise FileNotFoundError(f"Could not find file '{xml_path}'")

        self.model = mujoco.MjModel.from_xml_path(filename=str(xml_path), assets=None)

        self.data = mujoco.MjData(self.model)

        # Populate data
        mujoco.mj_forward(self.model, self.data)

        # print(self.model.opt)

    def time(self) -> float:
        """"""

        return self.data.time

    def gravity(self) -> npt.NDArray:
        """"""

        return self.model.opt.gravity

    def number_of_joints(self) -> int:
        """"""

        return self.model.njnt

    def number_of_geometries(self) -> int:
        """"""

        return self.model.ngeom

    def number_of_bodies(self) -> int:
        """"""

        return self.model.nbody

    def joint_names(self) -> List[str]:
        """"""

        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(self.number_of_joints())
        ]

    def joint_dofs(self, joint_name: str) -> int:
        """"""

        if joint_name not in self.joint_names():
            raise ValueError(f"Joint '{joint_name}' not found")

        return self.data.joint(joint_name).qpos.size

    def joint_position(self, joint_name: str) -> npt.NDArray:
        """"""

        if joint_name not in self.joint_names():
            raise ValueError(f"Joint '{joint_name}' not found")

        return self.data.joint(joint_name).qpos

    def joint_velocity(self, joint_name: str) -> npt.NDArray:
        """"""

        if joint_name not in self.joint_names():
            raise ValueError(f"Joint '{joint_name}' not found")

        return self.data.joint(joint_name).qvel

    def body_names(self) -> List[str]:
        """"""

        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, idx)
            for idx in range(self.number_of_bodies())
        ]

    def body_position(self, body_name: str) -> npt.NDArray:
        """"""

        if body_name not in self.body_names():
            raise ValueError(f"Body '{body_name}' not found")

        return self.data.body(body_name).xpos

    def body_orientation(self, body_name: str, dcm: bool = False) -> npt.NDArray:
        """"""

        if body_name not in self.body_names():
            raise ValueError(f"Body '{body_name}' not found")

        return (
            self.data.body(body_name).xmat if dcm else self.data.body(body_name).xquat
        )

    def geometry_names(self) -> List[str]:
        """"""

        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, idx)
            for idx in range(self.number_of_geometries())
        ]

    def geometry_position(self, geometry_name: str) -> npt.NDArray:
        """"""

        if geometry_name not in self.geometry_names():
            raise ValueError(f"Geometry '{geometry_name}' not found")

        return self.data.geom(geometry_name).xpos

    def geometry_orientation(
        self, geometry_name: str, dcm: bool = False
    ) -> npt.NDArray:
        """"""

        if geometry_name not in self.geometry_names():
            raise ValueError(f"Geometry '{geometry_name}' not found")

        R = np.reshape(self.data.geom(geometry_name).xmat, (3, 3))

        if dcm:
            return R

        q_xyzw = Rotation.from_matrix(R).as_quat()
        return q_xyzw[[3, 0, 1, 2]]

    def to_string(self) -> Tuple[str, str]:
        """Convert a mujoco model to a string."""

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+") as f:
            mujoco.mj_saveLastXML(f.name, self.model)
            mjcf_string = pathlib.Path(f.name).read_text()

        with tempfile.NamedTemporaryFile(mode="w+") as f:
            mujoco.mj_printModel(self.model, f.name)
            compiled_model_string = pathlib.Path(f.name).read_text()

        return mjcf_string, compiled_model_string


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
        log_rewards: bool = False,
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

        # Initialize the rewards logger
        self.logger_rewards = [] if log_rewards else None

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

        if self.logger_rewards is not None:
            self.logger_rewards.append(np.array(rewards).mean())

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
        return [False] * self.num_envs
        # raise NotImplementedError

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

    vec_env_sb = CustomVecEnvSB(jax_vector_env=vec_env, log_rewards=True)

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
    # vec_env_norm: Optional[VecNormalize] = None,
) -> None:
    """"""

    # Create the environment if a callable is passed
    env = env if isinstance(env, gym.Env) else env()

    # Initialize a random policy if none is passed
    policy = policy if policy is not None else lambda obs: env.action_space.sample()

    # if vec_env_norm is not None and not isinstance(vec_env_norm, VecNormalize):
    #     raise TypeError(vec_env_norm, VecNormalize)

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

    vec_env = make_vec_env_stable_baselines(
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
        env=vec_env,
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
        env=SquashActionWrapper(
            # env=func_env
            env=ActionNoiseWrapper(env=func_env)
        ),
    )

    vec_env = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        # n_envs=10,
        n_envs=512,
        # n_envs=2048,  # TODO
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    vec_env = VecMonitor(
        venv=VecNormalize(
            venv=vec_env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.95,
            epsilon=1e-8,
        )
    )

    %time _ = vec_env.reset()
    %time _ = vec_env.reset()
    actions = vec_env.jax_vector_env.action_space.sample()
    %time _ = vec_env.step(actions)
    %time _ = vec_env.step(actions)

    # 0: ok
    # 1: ok
    # 2: ok
    # 3: ok
    # 4: ok
    # 5: ok
    # 6: ok
    # 7: -> now
    # 8:
    # 9:
    vec_env.venv.venv.logger_rewards = []
    seed = vec_env.seed(seed=7)[0]
    _ = vec_env.reset()

    import torch as th

    model = PPO(
        "MlpPolicy",
        env=vec_env,
        n_steps=5,  # in the vector env -> real ones are x512
        batch_size=256,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.1,
        normalize_advantage=True,
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

    # from stable_baselines3.common.

    # rewards = np.zeros((10, 982))  DO NOT EXEC
    # rewards_7 = np.array(vec_env.venv.venv.logger_rewards)   CHANGE _X
    # rewards[seed, :] = np.array(vec_env.venv.venv.logger_rewards)

    # rewards = np.vstack([rewards, np.atleast_2d(vec_env.logger_rewards)])

    # plt.plot(
    #     # np.arange(start=1, stop=len(vec_env.venv.venv.logger_rewards) + 1) * 512,
    #     # vec_env.venv.venv.logger_rewards,
    #     np.arange(start=1, stop=len(vec_env.venv.venv.logger_rewards) + 3) * 512,
    #     rewards.T,
    #     label=r"$\hat{r}$"
    # )
    # # plt.plot(step_data[model_js.name()].tf, joint_positions_mj, label=["d", "theta"])
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("Time steps")
    # plt.ylabel("Average reward over 512 environments")
    # # plt.title("Trajectory of the model's base")
    # plt.show()

    # import pickle
    # with open(file=pathlib.Path.home()
    #     / "git"
    #     / "jaxsim"
    #     / "scripts"
    #     / f"ppo_cartpole_swingup_rewards.pickle", mode="w+b") as f:
    #     pickle.dump(rewards, f)

    # model.save(
    #     path=pathlib.Path.home()
    #     / "git"
    #     / "jaxsim"
    #     / "scripts"
    #     / f"ppo_cartpole_swing_up_seed={seed}.zip"
    # )

    for _ in range(5):
        # Train the model
        model = model.learn(total_timesteps=50_000, progress_bar=False)
        # %time model = model.learn(total_timesteps=500_000, progress_bar=False)

        # Create the policy closure
        policy = lambda observation: model.policy.predict(
            # observation=observation, deterministic=True
            observation=vec_env.normalize_obs(observation), deterministic=True
        )[0]

        # Evaluate the policy
        print("Evaluating...")
        evaluate(
            env=env_eval,
            num_episodes=10,
            seed=None,
            render=True,
            policy=policy,
            # vec_env_norm=vec_env,
        )

    # for _ in range(n_steps):
    #     observation = mj_observation(mujoco_model=m)
    #     action = model.policy.predict(observation=observation, deterministic=True)[0]
    #     m.data.ctrl = np.atleast_1d(action)
    #     mujoco.mj_step(m.model, m.data)
    #     # mujoco.mj_forward(m.model, m.data)

    # import palettable
    # # https://jiffyclub.github.io/palettable/cartocolors/diverging/
    # colors = palettable.cartocolors.diverging.Geyser_5.mpl_colors
    #
    # r = rewards.copy()
    # mean = r.mean(axis=0)
    # std = r.std(axis=0)
    # std_up = mean + std/2
    # std_down = mean - std/2
    #
    # fig, ax1 = plt.subplots(1, 1)
    # ax1.fill_between(
    #     np.arange(start=1, stop=mean.size + 1) * 512,
    #     std_down,
    #     std_up,
    #     label=r"$\pm \sigma$",
    #     color=colors[1],
    # )
    # ax1.plot(
    #     np.arange(start=1, stop=mean.size + 1) * 512,
    #     mean,
    #     color=colors[0],
    # )
    # ax1.grid()
    # # ax1.legend(loc="lower right")
    # ax1.set_title(r"\textbf{Average reward}")
    # ax1.set_xlabel("Samples")
    #
    # # plt.show()
    #
    # import tikzplotlib
    # tikzplotlib.clean_figure()
    # print(tikzplotlib.get_tikz_code())

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


# Comparison with mujoco
if __name__ == "__main_comparison_mujoco":
    """"""

    model = PPO.load(
        path=pathlib.Path.home()
        / "git"
        / "jaxsim"
        / "scripts"
        / "ppo_cartpole_swing_up_seed=7.zip"
    )

    # Create the evaluation environment
    env_eval = make_jax_env_cartpole(
        render_mode="meshcat_viz",
        max_episode_steps=250,
    )

    # =================
    # Mujoco evaluation
    # =================

    def mj_observation(mujoco_model: MujocoModel) -> npt.NDArray:
        """"""

        mujoco.mj_forward(mujoco_model.model, mujoco_model.data)

        pivot_pos = mujoco_model.joint_position(joint_name="pivot")
        # θ = np.arctan2(np.sin(pivot_pos), np.cos(pivot_pos))
        θ = pivot_pos

        return np.array(
            [
                mujoco_model.joint_position(joint_name="linear"),
                mujoco_model.joint_velocity(joint_name="linear"),
                θ,
                mujoco_model.joint_velocity(joint_name="pivot"),
            ]
        ).squeeze().copy()

    def mj_reset(
        mujoco_model: MujocoModel, observation: Optional[npt.NDArray] = None
    ) -> npt.NDArray:
        """"""

        observation = (
            observation
            if observation is not None
            else np.array([0.0, 0.0, np.deg2rad(180), 0.0])
        )

        linear_pos = observation[0]
        linear_vel = observation[1]
        pivot_pos = observation[2]
        pivot_vel = observation[3]

        mujoco_model.data.qpos = np.array([linear_pos, pivot_pos])
        mujoco_model.data.qvel = np.array([linear_vel, pivot_vel])
        mujoco.mj_forward(mujoco_model.model, mujoco_model.data)

        return mj_observation(mujoco_model=mujoco_model)

    def mj_step(action: npt.NDArray, mujoco_model: MujocoModel) -> None:
        """"""

        n_steps = int(0.050 / mujoco_model.model.opt.timestep)
        mujoco_model.data.ctrl = np.atleast_1d(action.squeeze()).copy()
        mujoco.mj_step(mujoco_model.model, mujoco_model.data, n_steps)

    # m.data.qpos = np.array([0.0, np.deg2rad(180)])
    # m.data.qvel = np.array([0.0, 0.0])
    # mujoco.mj_forward(m.model, m.data)

    # # Create the policy closure
    # policy = lambda observation: model.policy.predict(
    #     # observation=observation, deterministic=True
    #     observation=vec_env.normalize_obs(observation), deterministic=True
    # )[0]

    # ==============
    # Mujoco regular
    # ==============

    model_xml_path = (
        pathlib.Path.home()
        / "git"
        / "jaxsim"
        / "examples"
        / "resources"
        / "cartpole_mj.xml"
    )

    self = m = MujocoModel(xml_path=model_xml_path)

    mj_action = []
    mj_pos_cart = []
    mj_pos_pole = []

    done = False
    iterations = 0
    mj_reset(mujoco_model=m)

    while not done:
        iterations += 1
        observation = mj_observation(mujoco_model=m)
        # action = model.policy.predict(observation=observation, deterministic=True)[0]
        obs_policy = observation.copy()
        obs_policy[2] = np.arctan2(np.sin(obs_policy[2]), np.cos(obs_policy[2]))
        action = policy(obs_policy)

        mj_step(action=action, mujoco_model=m)

        mj_action.append(action * 50)
        mj_pos_cart.append(observation[0])
        mj_pos_pole.append(observation[2])

        import time

        time.sleep(0.050)

        print(observation, "\t", action)

        if iterations >= 201:
            break

    # ======
    # Jaxsim
    # ======

    # js_action = []
    # js_pos_cart = []
    # js_pos_pole = []
    #
    # done = False
    # iterations = 0
    # observation, _ = env_eval.reset()
    #
    # while not done:
    #     iterations += 1
    #     action = policy(observation)
    #     observation, _, _, _, _, = env_eval.step(action)
    #
    #     js_action.append(action * 50)
    #     js_pos_cart.append(observation[0])
    #     js_pos_pole.append(observation[2])
    #
    #     # import time
    #     #
    #     # time.sleep(0.050)
    #
    #     print(observation, "\t", action)
    #
    #     if iterations >= 201:
    #         break

    # ============
    # Mujoco alt 1
    # ============

    model_xml_path = (
            pathlib.Path.home()
            / "git"
            / "jaxsim"
            / "examples"
            / "resources"
            / "cartpole_mj.xml"
    )

    m = MujocoModel(xml_path=model_xml_path)

    mj_action_alt1 = []
    mj_pos_cart_alt1 = []
    mj_pos_pole_alt1 = []

    done = False
    iterations = 0
    mj_reset(mujoco_model=m)

    while not done:
        iterations += 1
        observation = mj_observation(mujoco_model=m)
        # action = policy(observation)
        obs_policy = observation.copy()
        obs_policy[2] = np.arctan2(np.sin(obs_policy[2]), np.cos(obs_policy[2]))
        action = policy(obs_policy)
        mj_step(action=action, mujoco_model=m)

        mj_action_alt1.append(action * 50)
        mj_pos_cart_alt1.append(observation[0])
        mj_pos_pole_alt1.append(observation[2])

        import time

        time.sleep(0.050)

        print(observation, "\t", action)

        if iterations >= 201:
            break

    mj_action_alt1 = np.array(mj_action_alt1)
    mj_pos_cart_alt1 = np.array(mj_pos_cart_alt1)
    mj_pos_pole_alt1 = np.array(mj_pos_pole_alt1)

    # ============
    # Mujoco alt 2
    # ============

    model_xml_path = (
            pathlib.Path.home()
            / "git"
            / "jaxsim"
            / "examples"
            / "resources"
            / "cartpole_mj.xml"
    )

    m = MujocoModel(xml_path=model_xml_path)

    mj_action_alt2 = []
    mj_pos_cart_alt2 = []
    mj_pos_pole_alt2 = []

    done = False
    iterations = 0
    mj_reset(mujoco_model=m)

    while not done:
        iterations += 1
        observation = mj_observation(mujoco_model=m)
        # action = policy(observation)
        obs_policy = observation.copy()
        obs_policy[2] = np.arctan2(np.sin(obs_policy[2]), np.cos(obs_policy[2]))
        action = policy(obs_policy)
        mj_step(action=action, mujoco_model=m)

        mj_action_alt2.append(action * 50)
        mj_pos_cart_alt2.append(observation[0])
        mj_pos_pole_alt2.append(observation[2])

        import time

        time.sleep(0.050)

        print(observation, "\t", action)

        if iterations >= 201:
            break

    mj_action_alt2 = np.array(mj_action_alt2)
    mj_pos_cart_alt2 = np.array(mj_pos_cart_alt2)
    mj_pos_pole_alt2 = np.array(mj_pos_pole_alt2)

    # ====
    # Plot
    # ====

    import palettable
    # https://jiffyclub.github.io/palettable/cartocolors/diverging/
    # colors = palettable.cartocolors.diverging.Geyser_5.mpl_colors
    colors = palettable.cartocolors.qualitative.Prism_8.mpl_colors

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    time = np.arange(start=0, stop=len(mj_action)) * 0.050

    # ax1.plot(time, js_pos_pole, label=r"Jaxsim", color=colors[1], linewidth=1)
    # ax1.plot(time, mj_pos_pole, label=r"Mujoco", color=colors[7], linewidth=1)
    # ax2.plot(time, js_pos_cart, label=r"Jaxsim", color=colors[1], linewidth=1)
    # ax2.plot(time, mj_pos_cart, label=r"Mujoco", color=colors[7], linewidth=1)
    # ax3.plot(time, js_action, label=r"Jaxsim", color=colors[1], linewidth=1)
    # ax3.plot(time, mj_action, label=r"Mujoco", color=colors[7], linewidth=1)

    ax1.plot(time, mj_pos_pole, label=r"nominal", color=colors[1], linewidth=1)
    ax1.plot(time, mj_pos_pole_alt1, label=r"mass", color=colors[3], linewidth=1)
    ax1.plot(time, mj_pos_pole_alt2, label=r"mass+friction", color=colors[7], linewidth=1)
    ax2.plot(time, mj_pos_cart, label=r"nominal", color=colors[1], linewidth=1)
    ax2.plot(time, mj_pos_cart_alt1, label=r"mass", color=colors[3], linewidth=1)
    ax2.plot(time, mj_pos_cart_alt2, label=r"mass+friction", color=colors[7], linewidth=1)
    ax3.plot(time, mj_action, label=r"nominal", color=colors[1], linewidth=1)
    ax3.plot(time, mj_action_alt1, label=r"mass", color=colors[3], linewidth=1)
    ax3.plot(time, mj_action_alt2, label=r"mass+friction", color=colors[7], linewidth=1)

    ax1.grid()
    ax1.set_ylabel(r"Pole angle $\theta$ [rad]")
    ax2.grid()
    ax2.set_ylabel(r"Cart position $d$ [m]")
    ax3.grid()
    ax3.set_ylabel(r"Force applied to cart $f$ [N]")

    # ax1.set_title(r"Pole angle $\theta$")
    # ax2.set_title(r"Cart position $d$")
    # ax3.set_title(r"Force applied to cart $f$")

    # ax1.legend()
    # ax2.legend()
    # ax3.legend()

    # plt.legend()
    # plt.title(r"\textbf{Comparison of cartpole swing-up performance}")
    # fig.suptitle(r"\textbf{Comparison of cartpole swing-up performance}")
    fig.supxlabel("Time [s]")
    # plt.show()

    import tikzplotlib
    tikzplotlib.clean_figure()
    print(tikzplotlib.get_tikz_code())

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

    # TODO: rename _sb to prevent collision with module
    vec_env_sb = make_vec_env_stable_baselines(
        jax_dataclass_env=func_env,
        # n_envs=10,
        # n_envs=2048,  # troppo -> JIT lungo
        # n_envs=100,
        # n_envs=1024,
        # n_envs=2048,
        n_envs=512,
        seed=42,
        vec_env_kwargs=dict(
            jit_compile=True,
        ),
    )

    # %time _ = vec_env_sb.reset()
    # %time _ = vec_env_sb.reset()
    # actions = vec_env_sb.jax_vector_env.action_space.sample()
    # %time _ = vec_env_sb.step(actions)
    # %time _ = vec_env_sb.step(actions)

    import torch as th

    # TODO: se ogni reset c'e' 1 sec di sim -> mega lento perche' ci sara' sempre
    # un env che si sta resettando!

    model = PPO(
        "MlpPolicy",
        env=vec_env_sb,
        # n_steps=2048,
        # n_steps=512,  # in the vector env -> real ones are x10
        # n_steps=10,  # in the vector env -> real ones are x10
        n_steps=4,  # in the vector env -> real ones are x2048
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
            # policy=policy,
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
