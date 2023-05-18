import multiprocessing
from typing import Any, Generic, SupportsFloat

import gymnasium as gym
import jax.numpy as jnp
import jax.random
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding
from meshcat_viz import MeshcatWorld

import jaxgym.jax.pytree_space as spaces
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper
from jaxsim import logging


class JaxEnv(gym.Env[ObsType, ActType], Generic[ObsType, ActType]):
    """"""

    action_space: spaces.PyTree
    observation_space: spaces.PyTree

    metadata: dict[str, Any] = {"render_modes": ["meshcat_viz", "meshcat_viz_gui"]}

    def __init__(
        self,
        # func_env: FuncEnv | FuncWrapper,
        func_env: JaxDataclassEnv | JaxDataclassWrapper,
        metadata: dict[str, Any] | None = None,
        render_mode: str | None = None,
        reward_range: tuple[float, float] = (-float("inf"), float("inf")),
        spec: EnvSpec | None = None,
    ) -> None:
        """"""

        if not isinstance(func_env.unwrapped, JaxDataclassEnv):
            raise TypeError(type(func_env.unwrapped), JaxDataclassEnv)

        metadata = metadata if metadata is not None else dict(render_mode=list())

        # Store the jax environment
        self.func_env = func_env

        # Initialize the state of the environment
        self.state = None

        # Expose the same spaces
        self.action_space = func_env.action_space
        self.observation_space = func_env.observation_space
        # assert isinstance(self.action_space, spaces.PyTree)
        # assert isinstance(self.observation_space, spaces.PyTree)

        # Store the other mandatory attributes that gym.Env expects
        self.metadata = metadata
        self.render_mode = render_mode
        self.reward_range = reward_range

        # Store the environment specs
        self.spec = spec

        # self._is_box_action_space = isinstance(self.action_space, gym.spaces.Box)
        #
        # if self.render_mode == "rgb_array":
        #     self.render_state = self.func_env.render_init()
        # else:
        #     self.render_state = None
        self.render_state = None
        self._meshcat_world = None  # old
        self._meshcat_window = None  # old

        # Initialize the RNGs with a random seed
        seed = np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")
        self._np_random, _ = seeding.np_random(seed=int(seed))
        self.rng = jax.random.PRNGKey(seed=seed)

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

        self.rng, *sub_keys = jax.random.split(self.rng, num=num + 1)
        return jnp.stack(sub_keys).squeeze()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """"""

        # TODO: clip action with wrapper
        # assert isinstance(self.action_space, spaces.PyTree)
        # action = self.action_space.clip(x=action)

        # if self._is_box_action_space:
        #     assert isinstance(self.action_space, gym.spaces.Box)  # For typing
        #     action = np.clip(action, self.action_space.low, self.action_space.high)
        # else:  # Discrete
        #     # For now we assume jax envs don't use complex spaces
        #     err_msg = f"{action!r} ({type(action)}) invalid"
        #     assert self.action_space.contains(action), err_msg

        # rng, self.rng = jrng.split(self.rng)

        # Advance the functional environment
        next_state = self.func_env.transition(
            state=self.state, action=action, rng=self.subkey(num=1)
        )

        # Extract updated data from the advanced environment
        observation = self.func_env.observation(state=next_state)
        reward = self.func_env.reward(
            state=self.state, action=action, next_state=next_state
        )
        info = self.func_env.step_info(
            state=self.state, action=action, next_state=next_state
        )

        # Detect if the environment reached a terminal state
        terminated = self.func_env.terminal(state=next_state)
        truncated = (
            # False if "truncated" not in info else type(terminated)(info["truncated"])
            type(terminated)(False)
            if "truncated" not in info
            else type(terminated)(info["truncated"])
        )

        # Remove the redundant "truncated" entry from info if present
        _ = info.pop("truncated", None)

        # Store the updated state
        self.state = next_state

        # observation = jax_to_numpy(observation)

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment using the seed."""

        super().reset(seed=seed)
        self.rng = jax.random.PRNGKey(seed) if seed is not None else self.rng

        # Seed the spaces
        self.action_space.seed(seed=seed)
        self.observation_space.seed(seed=seed)

        # Generate initial state
        self.state = self.func_env.initial(rng=self.subkey(num=1))

        # Sample the initial observation and info
        obs = self.func_env.observation(state=self.state)
        info = self.func_env.state_info(state=self.state)

        # obs = jax_to_numpy(obs)

        # assert self.observation_space.contains(obs), obs
        if obs not in self.observation_space:
            logging.warning(f"Initial observation not in observation space")
            logging.debug(obs)

        return obs, info

    # @property
    # def visualizer(self) -> MeshcatWorld:
    #     """Returns the visualizer if `render_mode` is 'meshcat_viz'."""
    #
    #     if self._meshcat_world is not None:
    #         return self._meshcat_world
    #
    #     world = MeshcatWorld()
    #     world.open()
    #
    #     def open_window(web_url: str) -> None:
    #         import tkinter
    #
    #         import webview
    #
    #         # Create an instance of tkinter frame or window
    #         win = tkinter.Tk()
    #         win.geometry("700x350")
    #
    #         webview.create_window("meshcat", web_url)
    #         webview.start(gui="qt")
    #
    #     # TODO: non si apre niente in subprocess!
    #     p = multiprocessing.Process(target=open_window, args=(world.web_url,))
    #
    #     self._meshcat_window = p
    #     self._meshcat_world = world
    #     return self._meshcat_world

    # def update_meshcat_world(self) -> None:
    #     """"""
    #
    #     return None

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Returns the render state if `render_mode` is 'rgb_array'."""

        if self.render_mode not in {None, "meshcat_viz", "meshcat_viz_gui"}:
            raise NotImplementedError(self.render_mode)

        if self.render_mode is None:
            return None

        if self.render_state is None:
            if self.render_mode == "meshcat_viz":
                self.render_state = self.func_env.render_init(open_gui=False)
            elif self.render_mode == "meshcat_viz_gui":
                self.render_state = self.func_env.render_init(open_gui=True)
            else:
                raise ValueError(self.render_mode)

        self.render_state, image = self.func_env.render_image(
            self.state, self.render_state
        )

        return image

        # # TODO: how to create proper interfaces?
        # self._meshcat_world = self.func_env.unwrapped.update_meshcat_world(
        #     world=self.visualizer, state=self.state["env"]
        # )
        #
        # return None

        # if self.render_mode == "rgb_array":
        #     self.render_state, image = self.func_env.render_image(
        #         self.state, self.render_state
        #     )
        #     return image
        # else:
        #     raise NotImplementedError

    def close(self) -> None:
        """"""

        # import meshcat_viz.meshcat
        #
        # if self._meshcat_world is not None:
        #     # self._meshcat_world.close()
        #     self._meshcat_world = None
        #     self._meshcat_window.kill()
        #     self._meshcat_window = None

        if self.render_state is not None:
            self.func_env.render_close(self.render_state)
            self.render_state = None

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.func_env}>"
