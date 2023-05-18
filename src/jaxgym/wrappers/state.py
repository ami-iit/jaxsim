import abc
from typing import Any, Generic, TypeVar

import numpy.typing as npt
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

from jaxgym.functional import FuncWrapper

WrapperStateType = TypeVar("WrapperStateType")
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


class StateWrapper(
    FuncWrapper[
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
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
        WrapperStateType,
    ],
    abc.ABC,
):
    """"""

    @abc.abstractmethod
    def wrapper_state_to_environment_state(
        self, wrapper_state: WrapperStateType
    ) -> StateType:
        """"""

        pass

    @abc.abstractmethod
    def initial(self, rng: Any = None) -> WrapperStateType:
        """"""

        pass

    @abc.abstractmethod
    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        pass

    #
    #
    #

    def observation(self, state: WrapperStateType) -> WrapperObsType:
        """"""

        return self.env.observation(
            state=self.wrapper_state_to_environment_state(wrapper_state=state)
        )

    def reward(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> WrapperRewardType:
        """"""

        return self.env.reward(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            action=action,
            next_state=self.wrapper_state_to_environment_state(
                wrapper_state=next_state
            ),
        )

    def terminal(self, state: WrapperStateType) -> TerminalType:
        """"""

        return self.env.terminal(
            state=self.wrapper_state_to_environment_state(wrapper_state=state)
        )

    def state_info(self, state: WrapperStateType) -> dict[str, Any]:
        """Info dict about a single state."""

        return self.env.state_info(
            state=self.wrapper_state_to_environment_state(wrapper_state=state)
        )

    def step_info(
        self, state: WrapperStateType, action: ActType, next_state: WrapperStateType
    ) -> dict[str, Any]:
        """Info dict about a full transition."""

        return self.env.step_info(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            action=action,
            next_state=self.wrapper_state_to_environment_state(
                wrapper_state=next_state
            ),
        )

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, npt.NDArray]:
        """Render the state."""

        return self.env.render_image(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            render_state=render_state,
        )
