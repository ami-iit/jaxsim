from typing import Any

import jax.numpy as jnp
import jax.tree_util
import jax_dataclasses
import numpy as np
import numpy.typing as npt
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

from jaxgym.jax import JaxDataclassWrapper
from jaxsim import logging

WrapperStateType = StateType
WrapperObsType = npt.NDArray
WrapperActType = npt.NDArray
WrapperRewardType = float


@jax_dataclasses.pytree_dataclass
class ToNumPyWrapper(
    JaxDataclassWrapper[
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
    ]
):
    # class ToNumPyWrapper(
    #     FuncWrapper[
    #         #
    #         StateType,
    #         ObsType,
    #         ActType,
    #         RewardType,
    #         TerminalType,
    #         RenderStateType,
    #         #
    #         WrapperStateType,
    #         WrapperObsType,
    #         WrapperActType,
    #         WrapperRewardType,
    #     ],
    #     Generic[
    #         StateType,
    #         ObsType,
    #         ActType,
    #         RewardType,
    #         TerminalType,
    #         RenderStateType,
    #     ],
    # ):
    """"""

    # def __init__(
    #     self,
    #     env: FuncEnv[
    #         StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    #     ],
    # ):
    #     """"""
    #
    #     self.env = env
    #     assert isinstance(self.env.action_space.sample(), np.ndarray)
    #     assert isinstance(self.env.observation_space.sample(), np.ndarray)

    def __post_init__(self) -> None:
        """"""

        super().__post_init__()

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

        assert isinstance(self.env.action_space.sample(), np.ndarray)
        assert isinstance(self.env.observation_space.sample(), np.ndarray)

    # def transition(
    #     self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    # ) -> WrapperStateType:
    #     """"""
    #
    #     return ToNumPyWrapper.pytree_to_numpy(
    #         self.env.transition(state=state, action=action, rng=rng)
    #     )

    def observation(self, state: WrapperStateType) -> WrapperObsType:
        """"""

        observation = ToNumPyWrapper.pytree_to_numpy(self.env.observation(state=state))
        return np.array(observation, dtype=self.env.observation_space.dtype)

    def reward(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> WrapperRewardType:
        """"""

        return float(
            ToNumPyWrapper.pytree_to_numpy(
                self.env.reward(state=state, action=action, next_state=next_state)
            )
        )

    def terminal(self, state: WrapperStateType) -> TerminalType:
        """"""

        return ToNumPyWrapper.pytree_to_numpy(self.env.terminal(state=state))

    def state_info(self, state: WrapperStateType) -> dict[str, Any]:
        """"""

        return ToNumPyWrapper.pytree_to_numpy(self.env.state_info(state=state))

    def step_info(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
    ) -> dict[str, Any]:
        """"""

        return ToNumPyWrapper.pytree_to_numpy(
            self.env.step_info(state=state, action=action, next_state=next_state)
        )

    @staticmethod
    def pytree_to_numpy(pytree: Any) -> Any:
        """"""

        def convert_leaf(leaf: Any) -> Any:
            """"""

            if (
                isinstance(leaf, (np.ndarray, jnp.ndarray))
                and leaf.size == 1
                and leaf.dtype == "bool"
            ):
                return bool(leaf)

            return np.array(leaf)

        return jax.tree_util.tree_map(lambda l: convert_leaf(l), pytree)
