from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

register(
    id="CartpoleSwingUpEnv-V0",
    entry_point="jaxgym.envs.cartpole:CartpoleSwingUpEnvV0",
    vector_entry_point="jaxgym.envs.cartpole:CartpoleSwingUpVectorEnvV0",
    # max_episode_steps=5_000,
    # reward_threshold=195.0,
    # kwargs=dict(max_episode_steps=5_000, blabla=True),
)
