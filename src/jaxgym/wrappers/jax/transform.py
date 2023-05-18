import dataclasses
from typing import Callable

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
from jaxgym.jax import JaxDataclassEnv, JaxDataclassWrapper
from jaxgym.wrappers import TransformWrapper
from jaxsim.utils import JaxsimDataclass, Mutability


# TODO: Make Jit and Vmap explicit wrappers so that we can check that env is JaxDataClass?
@jax_dataclasses.pytree_dataclass
# class JaxTransformWrapper(TransformWrapper, JaxsimDataclass):
class JaxTransformWrapper(TransformWrapper, JaxDataclassWrapper):
    """"""

    # env: JaxDataclassEnv[
    #     StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType
    # ]

    function: Callable[[Callable], Callable] = jax_dataclasses.static_field()

    transform_initial: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_transition: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_observation: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_reward: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_terminal: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_state_info: dataclasses.InitVar[bool] = dataclasses.field(default=True)
    transform_step_info: dataclasses.InitVar[bool] = dataclasses.field(default=True)

    def __post_init__(  # noqa
        self,
        transform_initial: bool,
        transform_transition: bool,
        transform_observation: bool,
        transform_reward: bool,
        transform_terminal: bool,
        transform_state_info: bool,
        transform_step_info: bool,
    ) -> None:
        """"""

        JaxDataclassWrapper.__post_init__(self)

        msg = f"[{self.__class__.__name__}] function={self.function}"
        logging.debug(msg=msg)

        with self.mutable_context(mutability=Mutability.MUTABLE):
            # super().__post_init__(
            super().__init__(
                env=self.env,
                function=self.function,
                transform_initial=transform_initial,
                transform_transition=transform_transition,
                transform_observation=transform_observation,
                transform_reward=transform_reward,
                transform_terminal=transform_terminal,
                transform_state_info=transform_state_info,
                transform_step_info=transform_step_info,
            )
