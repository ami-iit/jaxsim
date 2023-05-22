import copy
from jaxsim import logging
from typing import Any, Sequence

import jax.flatten_util
import jax.numpy as jnp
import jax.random
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)
from gymnasium.experimental.vector.vector_env import ArrayType, VectorEnv
from gymnasium.utils import seeding
from gymnasium.vector.utils import batch_space

import jaxgym.jax.pytree_space as spaces
import jaxsim.typing as jtp
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper
from jaxgym.wrappers.jax import JaxTransformWrapper, TimeLimit
from jaxsim.utils import not_tracing


# https://github.com/Farama-Foundation/Gymnasium/blob/e0cd42f77504060e770ab52932bf7eba45ff1976/gymnasium/experimental/functional_jax_env.py#L116
# TODO: allow num_envs = 1 so we have automatically autoreset?
# class JaxVectorEnv(VectorEnv[VectorObsType, VectorActType, ArrayType]):
# Note no dataclass here on all stuff related to VectorEnv
class JaxVectorEnv(VectorEnv[ObsType, ActType, ArrayType]):
    """
    A vectorized version of JAX-based functional environments exposing `VectorEnv` APIs.
    """

    observation_space: spaces.PyTree
    action_space: spaces.PyTree

    def __init__(
        self,
        func_env: JaxDataclassEnv[
            StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
        ]
        | JaxDataclassWrapper,
        num_envs: int,
        max_episode_steps: int = 0,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        reward_range: tuple[float, float] = (-float("inf"), float("inf")),
        spec: EnvSpec | None = None,
        jit_compile: bool = True,
    ) -> None:
        """"""

        if not isinstance(func_env.unwrapped, JaxDataclassEnv):
            raise TypeError(type(func_env.unwrapped), JaxDataclassEnv)

        metadata = metadata if metadata is not None else dict(render_mode=list())

        self.num_envs = num_envs
        self.func_env_single = func_env
        self.single_observation_space = func_env.observation_space
        self.single_action_space = func_env.action_space

        # TODO: convert other spaces to their PyTree equivalent
        assert isinstance(func_env.action_space, spaces.PyTree)
        assert isinstance(func_env.observation_space, spaces.PyTree)

        self.action_space = batch_space(self.single_action_space, n=num_envs)
        self.observation_space = batch_space(self.single_observation_space, n=num_envs)

        # TODO: attributes below
        self.metadata = metadata
        self.render_mode = render_mode
        self.reward_range = reward_range
        self.spec = spec
        # self.time_limit = max_episode_steps

        # Store the original functional environment
        self.func_env_single = func_env

        def has_wrapper(
            func_env: JaxDataclassEnv | JaxDataclassWrapper,
            wrapper_cls: type,
        ) -> bool:
            """"""

            while not isinstance(func_env, JaxDataclassEnv):
                if isinstance(func_env, wrapper_cls):
                    return True

                func_env = func_env.env

            return False

        # Always wrap the environment in a TimeLimit wrapper, that automatically counts
        # the number of steps and issues a "truncated" flag.
        # Note: the TimeLimit wrapper is a no-op if max_episode_steps is 0.
        # Note: the state of the wrapped environment now is different. The state of
        #       the original environment is now encapsulated in a dictionary.
        # TODO: make this optional? Check if it is already wrapped?
        # if max_episode_steps is not None:
        if not has_wrapper(func_env=self.func_env_single, wrapper_cls=TimeLimit):
            logging.debug(
                "[JaxVectorEnv] Wrapping the environment in a 'TimeLimit' wrapper"
            )
            self.func_env_single = TimeLimit(
                env=self.func_env_single, max_episode_steps=max_episode_steps
            )

        # Initialize the attribute that will store the environments state
        self.states = None

        # Initialize the step counter
        # TODO: handled by TimeLimit
        # self.steps = jnp.zeros(self.num_envs, dtype=jnp.uint32)

        # TODO: in our case, assume pytree? -> batch easy and generic?
        # --> singledispatch from gymnasium.space to pytree? And add a wrapper to_numpy|to_pytorch later?
        # Doing like this, Obs|Action|Reward are always pytree.
        # self._is_box_action_space = isinstance(self.action_space, gym.spaces.Box)

        # if self.render_mode == "rgb_array":
        #     self.render_state = self.func_env.render_init()
        # else:
        #     self.render_state = None

        # Initialize the RNGs with a random seed
        seed = np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")
        self._np_random, _ = seeding.np_random(seed=int(seed))
        self._key = jax.random.PRNGKey(seed=seed)

        # self.func_env = TransformWrapper(env=self.func_env, function=jax.vmap)
        self.func_env = JaxTransformWrapper(env=self.func_env_single, function=jax.vmap)

        # Compile resources in JIT if requested.
        # Note: this wrapper will override any other JIT wrapper already present.
        if jit_compile:
            self.step_autoreset_func = jax.jit(self.step_autoreset_func)
            self.func_env = JaxTransformWrapper(env=self.func_env, function=jax.jit)

    def subkey(self, num: int = 1) -> jax.random.PRNGKeyArray:
        """
        Generate one or multiple sub-keys from the internal key.

        Note:
            The internal key is automatically updated, there's no need to handle
            the environment key externally.

        Args:
            num: Number of keys to generate.

        Returns:
            The generated sub-keys.
        """

        self._key, *sub_keys = jax.random.split(self._key, num=num + 1)
        return jnp.stack(sub_keys).squeeze()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environments.

        Note:
            This method should be called just once after the creation of the vectorized
            environment. This class implements autoreset, therefore environments that
            either terminated or have been truncated get automatically reset.

        Args:
            seed:
            options:

        Returns:
            A tuple containing the initial observations and the initial states' info.
        """

        super().reset(seed=seed)
        self._key = jax.random.PRNGKey(seed) if seed is not None else self._key

        # Generate initial states
        self.states = self.func_env.initial(rng=self.subkey(num=self.num_envs))

        # Sample initial observations and infos
        observations = self.func_env.observation(self.states)
        infos = self.func_env.state_info(self.states)

        return observations, infos

    @staticmethod
    def binary_mask_pytree(
        pytree_a: jtp.PyTree, pytree_b: jtp.PyTree, mask: Sequence[bool]
    ) -> jtp.PyTree:
        """
        Compute a new vectorized PyTree selecting elements from either of the
        two input PyTrees according to the boolean mask.

        Note:
            The shapes of pytree_a and pytree_b must match, and they must be vectorized,
            meaning that all their leafs have share the dimension of the first axis.
            The mask should have as many elements as this shared dimension.

        Args:
            pytree_a: the first vectorized PyTree object.
            pytree_b: the second vectorized PyTree object.
            mask: the boolean mask to select elements either from pytree_a (when True)
                  or pytree_b (when False).

        Returns:
            A new PyTree having elements taken either from pytree_a or pytree_a
            according to mask.
        """

        def check():
            first_dim_a = jax.tree_util.tree_map(lambda l: l.shape[0], pytree_a)
            first_dim_b = jax.tree_util.tree_map(lambda l: l.shape[0], pytree_b)

            # Check that the input PyTrees have the same first dimension of their leaves
            if first_dim_a != first_dim_b:
                raise ValueError()

            in_axis_a = jnp.unique(
                jax.flatten_util.ravel_pytree(first_dim_a)[0]
            ).squeeze()
            in_axis_b = jnp.unique(
                jax.flatten_util.ravel_pytree(first_dim_a)[0]
            ).squeeze()

            # Check that all leaves have the same first dimension and it matches with
            # the length of the mask
            if in_axis_a != in_axis_b != len(mask):
                raise ValueError()

        if not_tracing(var=pytree_a):
            check()

        # Convert the boolean mask to a PyTree having boolean leaves.
        # True elements of the leaves are taken from pytree_a, False ones from pytree_b.
        mask_pytree = jax.tree_util.tree_map(
            lambda l: jnp.ones_like(l, dtype=bool)
            * mask[(...,) + (jnp.newaxis,) * (l.ndim - 1)],
            pytree_a,
        )

        # Create the output pytree taking elements from either pytree_a or pytree_b
        # according to the boolean PyTree built from the mask
        tree_out = jax.tree_util.tree_map(
            lambda a, b, m: jnp.where(m, a, b), pytree_a, pytree_b, mask_pytree
        )

        return tree_out

    @staticmethod
    def step_autoreset_func(
        env: JaxDataclassEnv | JaxDataclassWrapper,
        states: StateType,
        actions: ActType,
        key1: jax.random.PRNGKeyArray,
        key2: jax.random.PRNGKeyArray,
    ) -> tuple[StateType, tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]]:
        """"""

        # Duplicate the keys
        # TODO: with new jax version maybe split can be jitted -> pass just one key and split inside
        # key, *subkey_1 = jax.random.split(key, num=num_envs + 1)
        # key, *subkey_2 = jax.random.split(key, num=num_envs + 1)

        # Compute data by stepping the environments
        next_states = env.transition(state=states, action=actions, rng=key1)
        rewards = env.reward(state=states, action=actions, next_state=next_states)
        terminals = env.terminal(state=next_states)
        step_infos = env.step_info(state=states, action=actions, next_state=next_states)
        truncated = step_infos["truncated"]

        # Check if any environment is done
        dones = jnp.logical_or(terminals, truncated)

        # Add into step_infos the information about the final state even if the
        # environments are not done.
        # This is necessary for having a constant structure of the output pytree.
        # The _final_observation|_final_info masks can be used to filter out the
        # actual final data from the final_observation|final_info dictionaries.
        #
        # Note: the step_info dictionary of done environments that have been
        #       automatically reset shouldn't be consumed. It refers to the environment
        #       before being reset. Users in this case should read final_info.
        step_infos |= (
            dict(
                final_observation=env.observation(state=next_states),
                terminal_observation=env.observation(state=next_states),  # sb3
                final_info=copy.deepcopy(step_infos),
                _final_observation=dones,
                _final_info=dones,
                is_done=dones,
            )
            # Backward compatibility (?) -> SB3 (TODO: done in TimeLimit)
            # | {
            #     "TimeLimit.truncated": truncated,
            #     "terminal_observation": env.observation(state=next_states),
            # }
        )

        # Compute the new state and new state_infos for all environments.
        # We return this data only for those that are done.
        # new_states = env.initial(rng=key2)
        # new_state_infos = env.state_info(state=new_states)

        # Compute the new states for all environments.
        # We return this data only for those that are done.
        new_states = env.initial(rng=key2)

        # Merge the environment states
        new_env_states = JaxVectorEnv.binary_mask_pytree(
            mask=dones,
            # If done, return the new initial states
            pytree_a=new_states,
            # If not done, return the next states
            pytree_b=next_states,
        )

        # Compute the new observations.
        # This is a normal observation for environments that are not done.
        # This is the observation of the initial state for environments that were done.
        new_observations = env.observation(state=new_env_states)

        return new_env_states, (
            new_observations,
            rewards,
            terminals,
            truncated,
            step_infos,
        )

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Steps through the environment using the action."""

        # TODO: clip as method of Space.clip(x)? use here with hasattr(space, clip)?
        assert isinstance(self.action_space, spaces.PyTree)
        actions = self.action_space.clip(x=actions)

        # TODO: move these inside autoreset as soon as jax.random.split
        #       supports jit compilation
        keys_1 = self.subkey(num=self.num_envs)
        keys_2 = self.subkey(num=self.num_envs)

        self.states, out = JaxVectorEnv.step_autoreset_func(
            env=self.func_env,
            states=self.states,
            actions=actions,
            key1=keys_1,
            key2=keys_2,
        )

        return out
