from typing import Any, Callable, Generic

import numpy.typing as npt
import jax.numpy as jnp
import jax.flatten_util
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

from jaxgym.jax import JaxDataclassWrapper
from jaxsim import logging

WrapperStateType = StateType
WrapperObsType = ObsType
WrapperActType = ActType
WrapperRewardType = RewardType


@jax_dataclasses.pytree_dataclass
class ActionNoiseWrapper(
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

    noise_fn: Callable[
        [npt.NDArray, jax.random.PRNGKeyArray], npt.NDArray
    ] = jax_dataclasses.static_field(
        default=lambda action, rng: action
        + 0.05 * jax.random.normal(key=rng, shape=action.shape)
    )

    def __post_init__(self) -> None:
        """"""

        super().__post_init__()

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

    def transition(
        self, state: WrapperStateType, action: WrapperActType, rng: Any = None
    ) -> WrapperStateType:
        """"""

        rng, subkey = jax.random.split(rng, num=2)

        action_flat, restore_fn = jax.flatten_util.ravel_pytree(pytree=action)
        action_noisy_flat = self.noise_fn(action_flat, subkey)
        action_noisy = restore_fn(action_noisy_flat)

        return self.env.transition(state=state, action=action_noisy, rng=rng)
