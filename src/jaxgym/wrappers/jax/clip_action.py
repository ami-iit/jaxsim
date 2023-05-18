from typing import Generic

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


@jax_dataclasses.pytree_dataclass
class ClipActionWrapper(
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

    def action(self, action: WrapperActType) -> ActType:
        """"""

        return self.action_space.clip(x=action)
