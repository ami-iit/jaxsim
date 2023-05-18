from typing import Any, Generic, TypeVar

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

from jaxgym.functional import FuncEnv

WrapperStateType = TypeVar("WrapperStateType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperRewardType = TypeVar("WrapperRewardType")


class FuncWrapper(
    FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType],
    Generic[
        # FuncEnv types
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
        # FuncWrapper types
        WrapperStateType,
        WrapperObsType,
        WrapperActType,
        WrapperRewardType,
    ],
):
    """"""

    env: FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType]

    _action_space: gym.Space | None = None
    _observation_space: gym.Space | None = None

    @property
    def unwrapped(
        self,
    ) -> FuncEnv[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    ]:
        """"""

        return self.env.unwrapped

    @property
    def action_space(self) -> gym.Space[ActType]:
        """"""

        return (
            self._action_space
            if self._action_space is not None
            else self.env.action_space
        )

    @property
    def observation_space(self) -> gym.Space[ObsType]:
        """"""

        return (
            self._observation_space
            if self._observation_space is not None
            else self.env.observation_space
        )

    @action_space.setter
    def action_space(self, space: gym.Space[ActType]) -> None:
        """"""

        self._action_space = space

    @observation_space.setter
    def observation_space(self, space: gym.Space[ObsType]) -> None:
        """"""

        self._observation_space = space

    def __str__(self):
        """"""

        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """"""

        return str(self)

    # ==================================
    # Implementation of FunEnv interface
    # ==================================

    def initial(self, rng: Any = None) -> WrapperStateType:
        """"""

        return self.env.initial(rng=rng)

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        return self.env.transition(state=state, action=action, rng=rng)

    def observation(self, state: WrapperStateType) -> WrapperObsType:
        """"""

        return self.env.observation(state=state)

    def reward(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> WrapperRewardType:
        """"""

        return self.env.reward(state=state, action=action, next_state=next_state)

    def terminal(self, state: WrapperStateType) -> TerminalType:
        """"""

        return self.env.terminal(state=state)

    def state_info(self, state: WrapperStateType) -> dict[str, Any]:
        """"""

        return self.env.state_info(state=state)

    def step_info(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> dict[str, Any]:
        """"""

        return self.env.step_info(state=state, action=action, next_state=next_state)

    # =========
    # Rendering
    # =========

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """"""
        return self.env.render_image(state=state, render_state=render_state)

    def render_init(self, **kwargs) -> RenderStateType:
        """"""
        return self.env.render_init(**kwargs)

    def render_close(self, render_state: RenderStateType) -> None:
        """"""
        return self.env.render_close(render_state=render_state)


# class ActionFuncWrapper(
#     FuncWrapper[
#         # FuncEnv types
#         StateType,
#         ObsType,
#         ActType,
#         RewardType,
#         TerminalType,
#         RenderStateType,
#         # FuncWrapper types
#         StateType,
#         ObsType,
#         WrapperActType,
#         RewardType,
#     ],
#     Generic[WrapperActType],
#     abc.ABC,
# ):
#     """"""
#
#     @abc.abstractmethod
#     def action(self, action: WrapperActType) -> ActType:
#         """"""
#
#         pass
#
#     def reward(
#         self,
#         state: WrapperStateType,
#         action: WrapperActType,
#         next_state: WrapperStateType,
#     ) -> WrapperRewardType:
#         """"""
#
#         return self.env.reward(
#             state=state, action=self.action(action=action), next_state=next_state
#         )
#
#     def transition(
#         self, state: WrapperStateType, action: WrapperActType, rng: Any = None
#     ) -> WrapperStateType:
#         """"""
#
#         return self.env.transition(
#             state=state, action=self.action(action=action), rng=rng
#         )
#
#     def step_info(
#         self,
#         state: WrapperStateType,
#         action: WrapperActType,
#         next_state: WrapperStateType,
#     ) -> dict[str, Any]:
#         """"""
#
#         return self.env.step_info(
#             state=state, action=self.action(action=action), next_state=next_state
#         )
