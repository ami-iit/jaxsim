from typing import Any, Generic

import jax.numpy as jnp
import jax_dataclasses
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

from jaxgym.functional.core import FuncWrapper

WrapperStateType = StateType
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType

# TODO: non si puo' fare wrappando FuncWrapper perche' observation deve chiamare
#       initial, e initial ha bisogno di rng
# Implementare sopra env.FunctionalJaxEnv? -> Il JaxVecEnv fa gia' autoreset di default.


@jax_dataclasses.pytree_dataclass
class AutoResetWrapper(
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
    ],
):
    """"""

    def is_done(self, state: WrapperStateType) -> bool:
        """"""

        info = self.env.step_info()

        return jnp.array(
            [
                self.terminal(state=state),
                "truncated" in info and info["truncated"] is True,
            ],
            dtype=bool,
        ).any()

    def observation(self, state: WrapperStateType) -> WrapperObsType:
        """"""

        return self.env.observation(
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
