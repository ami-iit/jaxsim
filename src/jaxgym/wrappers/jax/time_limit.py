from typing import Any, ClassVar, Generic

import jax.lax
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

from jaxgym.functional import FuncEnv
from jaxgym.wrappers import StateWrapper
from jaxsim import logging

WrapperStateType = dict[str, Any]
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


@jax_dataclasses.pytree_dataclass
class TimeLimit(
    StateWrapper[
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
        WrapperStateType,
    ],
    Generic[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType],
):
    """"""

    env: FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType]
    max_episode_steps: int = jax_dataclasses.static_field()

    ElapsedStepsKey: ClassVar[str] = "elapsed_steps"
    EnvironmentStateKey: ClassVar[str] = "env"

    def __post_init__(self) -> None:
        """"""

        # TODO assert >=1?
        msg = f"[{self.__class__.__name__}] max_episode_steps={self.max_episode_steps}"
        logging.debug(msg=msg)

    def wrapper_state_to_environment_state(
        self, wrapper_state: WrapperStateType
    ) -> StateType:
        """"""

        return wrapper_state[TimeLimit.EnvironmentStateKey]

    def initial(self, rng: Any = None) -> WrapperStateType:
        """"""

        environment_state = self.env.initial(rng=rng)

        return {
            TimeLimit.EnvironmentStateKey: environment_state,
            TimeLimit.ElapsedStepsKey: 0,
        }

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        elapsed_steps = state[TimeLimit.ElapsedStepsKey]
        elapsed_steps += 1

        # print("+++")
        # print(state)
        # print(self.wrapper_state_to_environment_state(wrapper_state=state))

        environment_state = self.env.transition(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            action=action,
            rng=rng,
        )

        return {
            TimeLimit.EnvironmentStateKey: environment_state,
            TimeLimit.ElapsedStepsKey: elapsed_steps,
        }

    def step_info(
        self, state: WrapperStateType, action: ActType, next_state: WrapperStateType
    ) -> dict[str, Any]:
        """"""

        # Get the step info from the environment
        info = self.env.step_info(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            action=action,
            next_state=self.wrapper_state_to_environment_state(
                wrapper_state=next_state
            ),
        )

        # assert "truncated" not in info  # gymnasium

        # TODO: make a specific wrapper for stable baselines?
        # 1. add TimeLimit.truncated
        # 2. add terminal_observation
        # 3. step_dict -> list[step_dict]
        # 4. all to numpy
        # assert "TimeLimit.truncated" not in info  # stable-baselines3
        # TODO: in stable-baselines -> truncated and terminated are mutually exclusive

        # Activate the truncation flag if the episode is over
        truncated = jnp.array(
            next_state[TimeLimit.ElapsedStepsKey] >= self.max_episode_steps, dtype=bool
        )

        # If max_episode_steps=0, this wrapper is a no-op
        truncated = truncated if self.max_episode_steps != 0 else False

        # Check if any other wrapper already truncated the environment
        truncated = jnp.logical_or(truncated, info.get("truncated", False))
        # truncated = jax.lax.select(
        #     pred="truncated" in info,
        #     on_true=jnp.logical_or(truncated, info["truncated"]),
        #     on_false=truncated,
        # )

        # Handle the case in which the environment has been truncated and is done
        truncated = jax.lax.select(
            pred=self.terminal(state=next_state),
            on_true=False,
            on_false=truncated,
        )

        # Return the extended step info
        return info | dict(truncated=truncated)
        # return info | dict(truncated=truncated) | {"TimeLimit.truncated": truncated}


# @jax.jit
# def has_field(d) -> bool:
#     import jax.lax
#     return jax.lax.select(
#         pred="f" in d,
#         on_true=True,
#         on_false=False,
#     )
