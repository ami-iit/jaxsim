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

from jaxgym.jax import JaxDataclassWrapper
from jaxgym.functional import FuncEnv
from jaxsim import logging

WrapperStateType = StateType
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


# NON USARE -> DIRETTAMENTE FATTO IN TimeLimit e bon

@jax_dataclasses.pytree_dataclass
class TimeLimitStableBaselines(
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

    env: FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType]

    def __post_init__(self) -> None:
        """"""

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

    def step_info(
        self, state: WrapperStateType, action: ActType, next_state: WrapperStateType
    ) -> dict[str, Any]:
        """"""

        # Get the step info from the environment
        info = self.env.step_info(state=state, action=action, next_state=next_state)

        return info | {"TimeLimit.truncated": info.get("truncated", False)}

        # has_truncated_key = jax.lax.select(
        #     pred="TimeLimit.truncated" in info, on_true=True, on_false=False
        # )
        #
        # return jax.lax.select(
        #     # pred=jnp.array([]).all(),
        #     pred=info.get("truncated", False),
        #     on_true=info | {"TimeLimit.truncated": True},
        #     on_false=info | {"TimeLimit.truncated": False},
        # )

        # assert "truncated" not in info  # gymnasium
        #
        # # TODO: make a specific wrapper for stable baselines?
        # # 1. add TimeLimit.truncated
        # # 2. add terminal_observation
        # # 3. step_dict -> list[step_dict]
        # # 4. all to numpy
        # assert "TimeLimit.truncated" not in info  # stable-baselines3
        # # TODO: in stable-baselines -> truncated and terminated are mutually exclusive
        #
        # # Activate the truncation flag if the episode is over
        # truncated = jnp.array(
        #     next_state[TimeLimit.ElapsedStepsKey] >= self.max_episode_steps, dtype=bool
        # )
        #
        # # If max_episode_steps=0, this wrapper is a no-op
        # truncated = truncated if self.max_episode_steps != 0 else False
        #
        # # Return the extended step info
        # return info | dict(truncated=truncated) | {"TimeLimit.truncated": truncated}


# @jax.jit
# def has_field(d) -> bool:
#     import jax.lax
#     return jax.lax.select(
#         pred="f" in d,
#         on_true=True,
#         on_false=False,
#     )
