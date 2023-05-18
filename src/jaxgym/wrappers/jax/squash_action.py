from typing import Generic

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

from jaxgym.functional.func_wrapper import WrapperActType
from jaxgym.jax import JaxDataclassActionWrapper
from jaxsim import logging
from jaxsim import typing as jtp
from jaxsim.utils import Mutability


@jax_dataclasses.pytree_dataclass
class SquashActionWrapper(
    JaxDataclassActionWrapper[
        StateType,
        ObsType,
        ActType,
        RewardType,
        TerminalType,
        RenderStateType,
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

    def __post_init__(self) -> None:
        """"""

        super().__post_init__()

        msg = f"[{self.__class__.__name__}] enabled"
        logging.debug(msg=msg)

        # Replace the action space with the squashed action space.
        # Note: we assume the entire action space is bounded and there are no +-inf.
        # Note: we assume the entire action space is composed by floats (no bools, etc).

        import copy

        with self.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            # First make a copy of the PyTree
            self.action_space = copy.deepcopy(self.env.action_space)
            # self.action_space = jax.tree_util.tree_map(
            #     lambda l: l, self.env.action_space
            # )

            # The update both the low and high bounds
            self.action_space.low = jax.tree_util.tree_map(
                lambda l: -1.0 * jnp.ones_like(l), self.env.action_space.low
            )
            self.action_space.high = jax.tree_util.tree_map(
                lambda l: 1.0 * jnp.ones_like(l), self.env.action_space.high
            )

    def action(self, action: WrapperActType) -> ActType:
        """"""

        return self.unsquash(
            pytree=action,
            low=self.env.action_space.low,
            high=self.env.action_space.high,
        )

    @staticmethod
    def squash(pytree: jtp.PyTree, low: jtp.PyTree, high: jtp.PyTree) -> jtp.PyTree:
        """"""

        pytree_squashed = jax.tree_util.tree_map(
            lambda leaf, l, h: 2 * (leaf - l) / (h - l) - 1,
            pytree,
            low,
            high,
        )

        return pytree_squashed

    @staticmethod
    def unsquash(pytree: jtp.PyTree, low: jtp.PyTree, high: jtp.PyTree) -> jtp.PyTree:
        """"""

        pytree_unsquashed = jax.tree_util.tree_map(
            lambda leaf, l, h: (leaf + 1) * (h - l) / 2 + l,
            pytree,
            low,
            high,
        )

        return pytree_unsquashed
