import functools
from typing import Any, Dict, List, Optional, Type, Union

import gymnasium as gym
import jax.random
import numpy as np
from gymnasium.experimental.vector.vector_env import VectorWrapper
from stable_baselines3.common import vec_env as vec_env_sb

import jaxsim.typing as jtp
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper, PyTree
from jaxgym.vector.jax import FlattenSpacesVecWrapper, JaxVectorEnv
from jaxgym.wrappers.jax import ToNumPyWrapper


class CustomVecEnvSB(vec_env_sb.VecEnv):
    """Custom vectorized environment for SB3."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        jax_vector_env: JaxVectorEnv | VectorWrapper,
    ) -> None:
        """
        Create a custom vectorized environment for SB3 from a JaxVectorEnv.

        Args:
            jax_vector_env: The JaxVectorEnv to wrap.
        """

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
        """Reset all the environments."""

        observations, state_infos = self.jax_vector_env.reset(seed=self._seed)
        return np.array(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def tree_inverse_transpose(pytree: jtp.PyTree, batch_size: int) -> List[jtp.PyTree]:
        """
        Utility function to perform the inverse of a pytree transpose operation.

        It converts a pytree having the batch size in the first dimension of its leaves
        to a list of pytrees having a single batch sample in their leaves.

        Note: Check the direct transpose operation in the following link:
        https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#transposing-trees

        Args:
            pytree: The batched pytree.
            batch_size: The batch size.

        Returns:
            A list of pytrees having a single batch sample in their leaves.
        """

        return [
            jax.tree_util.tree_map(lambda leaf: leaf[i], pytree)
            for i in range(batch_size)
        ]

    def step_wait(self) -> vec_env_sb.base_vec_env.VecEnvStepReturn:
        """Wait for the step taken with step_async()."""

        (
            observations,
            rewards,
            terminals,
            truncated,
            step_infos,
        ) = self.jax_vector_env.step(actions=self.actions)

        done = np.logical_or(terminals, truncated)

        # Convert the infos from a batched dictionary to a list of dictionaries
        list_of_step_infos = self.tree_inverse_transpose(
            pytree=step_infos, batch_size=self.jax_vector_env.num_envs
        )

        # Convert all info data to numpy
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
        """Clean up the environment's resources."""

        return self.jax_vector_env.close()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """Sets the random seeds for all environments."""

        if seed is None:
            seed = np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")

        if np.array(seed, dtype="uint32") != np.array(seed):
            raise ValueError(f"seed must be compatible with 'uint32' casting")

        self._seed = seed
        return [seed]

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


def make_vec_env_stable_baselines(
    jax_dataclass_env: JaxDataclassEnv | JaxDataclassWrapper,
    n_envs: int = 1,
    seed: Optional[int] = None,
    # monitor_dir: Optional[str] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> vec_env_sb.VecEnv:
    """
    Create a SB3 vectorized environment from an individual `JaxDataclassEnv`.

    Args:
        jax_dataclass_env: The individual `JaxDataclassEnv`.
        n_envs: Number of parallel environments.
        seed: The seed for the vectorized environment.
        vec_env_kwargs: Additional arguments to pass upon environment creation.

    Returns:
        The SB3 vectorized environment.
    """

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

    # Convert the vectorized environment to a SB3 vectorized environment
    vec_env_sb = CustomVecEnvSB(jax_vector_env=vec_env)

    # Set the seed
    if seed is not None:
        _ = vec_env_sb.seed(seed=seed)

    return vec_env_sb
