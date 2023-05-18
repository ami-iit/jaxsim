from typing import Any

import gymnasium as gym
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

from jaxsim import logging
from jaxgym.jax import JaxDataclassWrapper

WrapperStateType = StateType
WrapperObsType = jnp.ndarray
WrapperActType = jnp.ndarray
WrapperRewardType = RewardType


# TODO: maybe better over JaxEnv to be consistent with JaxVectorEnv?
@jax_dataclasses.pytree_dataclass
class FlattenSpacesWrapper(
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
    """"""

    # Propagate to other Jax wrappers
    def __post_init__(self):
        """"""

        super().__post_init__()

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

    @property
    def action_space(self) -> gym.Space:
        """"""

        return self.env.action_space.to_box()

    @property
    def observation_space(self) -> gym.Space:
        """"""

        return self.env.observation_space.to_box()

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        action_pytree = self.env.action_space.unflatten_sample(x=action)
        return self.env.transition(state=state, action=action_pytree, rng=rng)

    def observation(self, state: WrapperStateType) -> WrapperObsType:
        """"""

        observation_pytree = self.env.observation(state=state)
        return self.env.observation_space.flatten_pytree(pytree=observation_pytree)
