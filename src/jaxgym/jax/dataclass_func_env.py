import jax_dataclasses
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)

import jaxgym.jax.pytree_space as spaces
from jaxgym.functional import FuncEnv
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class JaxDataclassEnv(
    FuncEnv[StateType, ObsType, ActType, RewardType, TerminalType, RenderStateType],
    JaxsimDataclass,
):
    """
    Base class for JAX-based functional environments.

    Note:
        All environments implementing this class must be pytree_dataclasses.
    """

    # Override spaces for JAX, storing them as static fields.
    # Note: currently only PyTree spaces are supported.
    # Note: always sample from these spaces using functional methods in order to
    #       avoid incurring in JIT recompilations.
    _action_space: spaces.PyTree | None = jax_dataclasses.static_field(init=False)
    _observation_space: spaces.PyTree | None = jax_dataclasses.static_field(init=False)

    @property
    def action_space(self) -> spaces.PyTree:
        return self._action_space

    @property
    def observation_space(self) -> spaces.PyTree:
        return self._observation_space
