import jax.numpy as jnp
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RenderStateType,
    RewardType,
    StateType,
    TerminalType,
)
from gymnasium.experimental.vector.vector_env import VectorWrapper

from jaxgym.vector.jax import JaxVectorEnv

WrapperStateType = StateType
WrapperObsType = jnp.ndarray  # TODO jax.typing
WrapperActType = jnp.ndarray
WrapperRewardType = RewardType


# TODO: not dataclass when operating on VectorWrapper -> check other ones
class FlattenSpacesVecWrapper(VectorWrapper):
    """"""

    # TODO: vec_env?
    env: JaxVectorEnv

    def __init__(self, env: JaxVectorEnv) -> None:
        """"""

        if not isinstance(env, JaxVectorEnv):
            raise TypeError(type(env))

        self.action_space = env.action_space.to_box()
        self.observation_space = env.observation_space.to_box()

        super().__init__(env=env)

    def reset(
        self,
        **kwargs
        # *,
        # seed: int | list[int] | None = None,
        # options: dict[str, Any] | None = None,
        # ) -> tuple[ObsType, dict[str, Any]]:
    ):
        """"""

        observation, state_info = self.env.reset(**kwargs)
        # return self.env.observation_space.flatten_pytree(pytree=observation), state_info
        return self.env.observation_space.flatten_sample(pytree=observation), state_info

    def step(self, actions):
        # ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """"""

        observations, rewards, terminals, truncated, step_infos = self.env.step(
            actions=self.env.action_space.unflatten_sample(x=actions)
        )

        if "final_observation" in step_infos:
            step_infos["final_observation"] = self.env.observation_space.flatten_sample(
                pytree=step_infos["final_observation"]
            )

        return (
            self.env.observation_space.flatten_sample(pytree=observations),
            rewards,
            terminals,
            truncated,
            step_infos,
        )
