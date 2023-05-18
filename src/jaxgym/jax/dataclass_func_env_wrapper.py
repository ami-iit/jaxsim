import abc
from typing import Any, Generic

import jax_dataclasses
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

import jaxgym.jax.pytree_space as spaces
from jaxgym.functional.func_wrapper import (
    FuncWrapper,
    WrapperActType,
    WrapperObsType,
    WrapperRewardType,
    WrapperStateType,
)
from jaxgym.jax import JaxDataclassEnv
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class JaxDataclassWrapper(
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
    # TODO
    Generic[
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
    JaxsimDataclass,
):
    """"""

    env: JaxDataclassEnv[
        StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    ]

    _action_space: spaces.PyTree | None = jax_dataclasses.static_field(init=False)
    _observation_space: spaces.PyTree | None = jax_dataclasses.static_field(init=False)

    def __post_init__(self) -> None:
        """"""

        if not isinstance(self.env.unwrapped, JaxDataclassEnv):
            raise TypeError(type(self.env.unwrapped), JaxDataclassEnv)

    @property
    def action_space(self) -> spaces.PyTree:
        """"""

        return (
            self._action_space
            if self._action_space is not None
            else self.env.action_space
        )

    @property
    def observation_space(self) -> spaces.PyTree:
        """"""

        return (
            self._observation_space
            if self._observation_space is not None
            else self.env.observation_space
        )

    @action_space.setter
    def action_space(self, space: spaces.PyTree) -> None:
        """"""

        self._action_space = space

    @observation_space.setter
    def observation_space(self, space: spaces.PyTree) -> None:
        """"""

        self._observation_space = space


@jax_dataclasses.pytree_dataclass
class JaxDataclassActionWrapper(
    JaxDataclassWrapper[
        # FuncEnv types
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
        # FuncWrapper types
        StateType,
        ObsType,
        ActType,
        RewardType,
    ],
    # TODO
    Generic[
        #
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
    ],
    abc.ABC,
):
    """"""

    @abc.abstractmethod
    def action(self, action: WrapperActType) -> ActType:
        """"""

        pass

    def reward(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> WrapperRewardType:
        """"""

        return self.env.reward(
            state=state, action=self.action(action=action), next_state=next_state
        )

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        return self.env.transition(
            state=state, action=self.action(action=action), rng=rng
        )

    def step_info(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> dict[str, Any]:
        """"""

        return self.env.step_info(
            state=state, action=self.action(action=action), next_state=next_state
        )
