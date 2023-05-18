from typing import Callable, Generic

from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

from jaxgym.functional import FuncEnv, FuncWrapper

WrapperStateType = StateType
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


class TransformWrapper(
    FuncWrapper[
        #
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
        #
        WrapperStateType,
        WrapperObsType,
        WrapperActType,
        WrapperRewardType,
    ],
    Generic[
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
    ],
):
    """"""

    def __init__(
        self,
        env: FuncEnv[
            StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
        ],
        function: Callable[[Callable], Callable] = lambda f: f,
        transform_initial: bool = True,
        transform_transition: bool = True,
        transform_observation: bool = True,
        transform_reward: bool = True,
        transform_terminal: bool = True,
        transform_state_info: bool = True,
        transform_step_info: bool = True,
    ):
        """"""

        self.env = env

        # Here to show up in repr
        self.function = function

        if transform_initial:
            self.initial = function(self.initial)

        if transform_transition:
            self.transition = function(self.transition)

        if transform_observation:
            self.observation = function(self.observation)

        if transform_reward:
            self.reward = function(self.reward)

        if transform_terminal:
            self.terminal = function(self.terminal)

        if transform_state_info:
            self.state_info = function(self.state_info)

        if transform_step_info:
            self.step_info = function(self.step_info)

    # @staticmethod
    # def transform_env(
    #     env: FuncEnv[
    #         StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    #     ],
    #     function: Callable[[Callable], Callable],
    #     transform_initial: bool = True,
    #     transform_transition: bool = True,
    #     transform_observation: bool = True,
    #     transform_reward: bool = True,
    #     transform_terminal: bool = True,
    #     transform_state_info: bool = True,
    #     transform_step_info: bool = True,
    # ) -> FuncWrapper[
    #     StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    # ]:
    #     """"""
    #
    #     if transform_initial:
    #         self.initial = function(self.initial)
    #
    #     if transform_transition:
    #         self.transition = function(self.transition)
    #
    #     if transform_observation:
    #         self.observation = function(self.observation)
    #
    #     if transform_reward:
    #         self.reward = function(self.reward)
    #
    #     if transform_terminal:
    #         self.terminal = function(self.terminal)
    #
    #     if transform_state_info:
    #         self.state_info = function(self.state_info)
    #
    #     if transform_step_info:
    #         self.step_info = function(self.step_info)
