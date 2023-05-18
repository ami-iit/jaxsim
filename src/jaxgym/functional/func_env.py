import abc
from typing import Any, Generic

import gymnasium as gym
import numpy.typing as npt
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)


# Similar to https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/experimental/functional.py
class FuncEnv(
    Generic[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType],
    abc.ABC,
):
    """
    Base class for functional environments.

    Note:
        This class is meant to be kept stateless.
        The state of the environment (possibly encapsulated with the states of wrappers)
        should be stored in the `state` argument of the methods.

    Note:
        This functional approach mainly targets JAX-based environments, but has been
        formulated in a generic way so that it can be implemented in other frameworks.
    """

    # These spaces have to be populated in the __post_init__ method.
    # If necessary, the __post_init__ method can access the simulator.
    # _action_space: spaces.Space | None = jax_dataclasses.static_field(init=False)
    # _observation_space: spaces.Space | None = jax_dataclasses.static_field(init=False)
    _action_space: gym.Space | None = None
    _observation_space: gym.Space | None = None

    @property
    def unwrapped(
        self,
    ) -> "FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType]":
        """Return the innermost environment."""

        return self

    @property
    def action_space(self) -> gym.Space[ActType]:
        """Return the action space."""

        return self._action_space

    @property
    def observation_space(self) -> gym.Space[ObsType]:
        """Return the observation space."""

        return self._observation_space

    # ================
    # Abstract methods
    # ================

    @abc.abstractmethod
    def initial(self, rng: Any = None) -> StateType:
        """
        Initialize the environment returning its initial state.s.

        Args:
            rng: A resource to initialize the RNG of the functional environment.

        Returns:
            The initial state of the environment.
        """
        pass

    @abc.abstractmethod
    def transition(
        self, state: StateType, action: ActType, rng: Any = None
    ) -> StateType:
        """
        Compute the next state by applying the given action to the functional environment
        in the given state.

        Args:
            state:
            action:
            rng:

        Returns:
            The next state of the environment.
        """
        pass

    @abc.abstractmethod
    def observation(self, state: StateType) -> ObsType:
        """
        Compute the observation of the environment in the given state.

        Args:
            state:

        Returns:
            The observation computed from the given state.
        """
        pass

    @abc.abstractmethod
    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """"""
        pass

    @abc.abstractmethod
    def terminal(self, state: StateType) -> TerminalType:
        """"""
        pass

    # =========
    # Rendering
    # =========

    # @abc.abstractmethod
    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """Show the state."""
        raise NotImplementedError

    # @abc.abstractmethod
    def render_init(self, **kwargs) -> RenderStateType:
        """Initialize the render state."""
        raise NotImplementedError

    # @abc.abstractmethod
    def render_close(self, render_state: RenderStateType) -> None:
        """Close the render state."""
        raise NotImplementedError

    # =============
    # Other methods
    # =============

    def state_info(self, state: StateType) -> dict[str, Any]:
        """Info dict about a single state."""

        return {}

    def step_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict[str, Any]:
        """Info dict about a full transition."""

        return {
            # TODO: this is to keep the autoreset structure constant and return
            #       always a dict with the same keys. Check if this is necessary.
            # "state_info": self.state_info(state=state),
            # "next_state_info": self.state_info(state=next_state),
        }

    # TODO: remove
    # def transform(self, func: Callable[[Callable], Callable]) -> None:
    #     """Functional transformations."""
    #     raise NotImplementedError
    #     # with self.mutable_context(mutability=Mutability.MUTABLE):
    #     self.initial = func(self.initial)
    #     self.transition = func(self.transition)
    #     self.observation = func(self.observation)
    #     self.reward = func(self.reward)
    #     self.terminal = func(self.terminal)
    #     self.state_info = func(self.state_info)
    #     self.step_info = func(self.step_info)

    def __str__(self) -> str:
        """"""

        return f"<{type(self).__name__}>"

    def __repr__(self) -> str:
        """"""

        return str(self)
