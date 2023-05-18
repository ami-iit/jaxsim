from typing import Any, ClassVar, Generic

import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util
import jax_dataclasses
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

import jaxsim.typing as jtp
from jaxgym.functional import FuncEnv
from jaxgym.wrappers import StateWrapper
from jaxsim import logging

WrapperStateType = StateType
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


@jax_dataclasses.pytree_dataclass
class NaNHandlerWrapper(
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
    """Reset the environment when a NaN is encountered."""

    env: FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType]

    HasNanKey: ClassVar[str] = "has_nan"
    EnvironmentStateKey: ClassVar[str] = "env"

    def __post_init__(self) -> None:
        """"""

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

    def wrapper_state_to_environment_state(
        self, wrapper_state: WrapperStateType
    ) -> StateType:
        """"""

        return wrapper_state[NaNHandlerWrapper.EnvironmentStateKey]

    def initial(self, rng: Any = None) -> WrapperStateType:
        """"""

        return {
            NaNHandlerWrapper.HasNanKey: jnp.array(False),
            NaNHandlerWrapper.EnvironmentStateKey: self.env.initial(rng=rng),
        }

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        # Copy the current state
        old_state = jax.tree_util.tree_map(
            lambda x: x, state[NaNHandlerWrapper.EnvironmentStateKey]
        )

        # Step the environment
        new_state = self.env.transition(
            state=self.wrapper_state_to_environment_state(wrapper_state=state),
            action=action,
            rng=rng,
        )

        new_state_without_nans = jax.tree_util.tree_map(
            lambda leaf_new, leaf_old: jax.lax.select(
                pred=self.pytree_has_nan_values(pytree=leaf_new),
                on_true=leaf_old,
                on_false=leaf_new,
            ),
            new_state,
            old_state,
        )

        return {
            NaNHandlerWrapper.HasNanKey: self.pytree_has_nan_values(pytree=new_state),
            NaNHandlerWrapper.EnvironmentStateKey: new_state_without_nans,
        }

    def step_info(
        self,
        state: WrapperStateType,
        action: WrapperActType,
        next_state: WrapperStateType,
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

        # Activate the truncation flag if the episode is over
        truncated = jnp.array(next_state[NaNHandlerWrapper.HasNanKey], dtype=bool)

        # Check if any other wrapper already truncated the environment
        truncated = jnp.logical_or(truncated, info.get("truncated", False))

        # Handle the case in which the environment has been truncated and is done
        truncated = jax.lax.select(
            pred=self.terminal(state=next_state),
            on_true=False,
            on_false=truncated,
        )

        # Return the extended step info
        return info | dict(truncated=truncated)

    @staticmethod
    def pytree_has_nan_values(pytree: jtp.PyTree) -> jtp.Bool:
        """"""

        pytree_flat, _ = jax.flatten_util.ravel_pytree(pytree=pytree)
        return jnp.isnan(pytree_flat).any()
