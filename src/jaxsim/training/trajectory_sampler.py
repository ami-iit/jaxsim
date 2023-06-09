from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses

from jaxsim.gym import Env, EnvironmentState
from jaxsim.gym.typing import *
from jaxsim.training.agent import Agent
from jaxsim.training.memory import Memory
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class TrajectorySampler(JaxsimDataclass):
    env: Env
    agent: Agent
    state: EnvironmentState

    _sampling_fn: Callable = jax_dataclasses.static_field(default=None)

    @staticmethod
    def build(
        env: Env, agent: Agent, state: EnvironmentState, number_of_environments: int = 1
    ) -> "TrajectorySampler":
        if number_of_environments > 1:
            env, state = TrajectorySampler.make_parallel_environment(
                number=number_of_environments, env=env, state=state
            )

        with TrajectorySampler(env=env, agent=agent, state=state).editable(
            validate=False
        ) as sampler:
            # Handle parallel environments
            sampling_fn = (
                TrajectorySampler.Sample_trajectory
                if sampler.number_of_environments() == 1
                else jax.vmap(
                    fun=TrajectorySampler.Sample_trajectory,
                    in_axes=(0, 0, None, None, None),
                )
            )

            # JIT compile sampling function
            sampler._sampling_fn = jax.jit(
                sampling_fn, static_argnames=["horizon_length"]
            )

        return sampler

    def seed(self, seed: int = 0) -> None:
        if self.number_of_environments() == 1:
            state = self.env.seed(state=self.state, seed=seed)

        else:
            with self.state.editable(validate=True) as state:
                state.key = jax.random.split(
                    key=jax.random.PRNGKey(seed=seed), num=self.number_of_environments()
                )

        self.set_mutability(mutable=True, validate=False)
        self.state = state
        self.set_mutability(mutable=False)

        _ = self.reset()

    @staticmethod
    def reset_fn(
        env: Env, state: EnvironmentState
    ) -> Tuple[EnvironmentState, Tuple[Observation, Reward, IsDone, Info]]:
        return env.reset(state=state)

    def reset(self) -> Observation:
        reset_fn_jit = (
            jax.jit(jax.vmap(self.reset_fn))
            if self.number_of_environments() > 1
            else jax.jit(self.reset_fn)
        )

        state, observation = reset_fn_jit(env=self.env, state=self.state)

        self.set_mutability(mutable=True, validate=False)
        self.state = state
        self.set_mutability(mutable=False)

        return observation

    def number_of_environments(self) -> int:
        shape_of_key = self.state.key.shape
        return 1 if len(shape_of_key) == 1 else shape_of_key[0]

    def sample_trajectory(
        self, horizon_length: int = 1, explore: bool = True
    ) -> Memory:
        # We cannot use named arguments: https://github.com/google/jax/issues/7465
        memory, state = self._sampling_fn(
            self.env, self.state, self.agent, int(horizon_length), bool(explore)
        )

        self.set_mutability(mutable=True, validate=True)
        self.state = state
        self.set_mutability(mutable=False)

        return memory

    # ===============
    # Private methods
    # ===============

    @staticmethod
    def make_parallel_environment(
        number: int, env: Env, state: EnvironmentState
    ) -> Tuple[Env, EnvironmentState]:
        envs = jax.tree_map(lambda *l: jnp.stack(l), *[env] * number)
        states = jax.tree_map(lambda *l: jnp.stack(l), *[state] * number)

        return envs, states

    @staticmethod
    def Sample_trajectory(
        env: Env,
        state: EnvironmentState,
        agent: Agent,
        horizon_length: int = 1,
        explore: bool = True,
    ) -> Tuple[Memory, EnvironmentState]:
        # Create the memory object of the entire batch
        memory = jax.tree_map(
            lambda *x0: jnp.stack(x0),
            *[TrajectorySampler.Zero_memory_sample(env=env, state=state, agent=agent)]
            * horizon_length,
        )

        carry_init = (state, memory)

        def body_fun(idx: int, carry: Tuple) -> Tuple:
            # Unpack the carry
            state, memory = carry

            # Get the current observation
            observation = env.get_observation(state=state).flatten()

            # Sample a new action
            subkey, state = state.generate_key()

            action, log_prob_action, value = agent.choose_action(
                observation=observation, explore=explore, key=subkey
            )
            # distribution, value = agent.train_state.apply_fn(
            #     agent.train_state.params, data=observation
            # )
            # action = jax.lax.select(
            #     pred=explore,
            #     on_true=distribution.sample(seed=subkey),
            #     on_false=distribution.mode(),
            # )
            # log_prob_action = distribution.log_prob(value=action)

            # Step the environment with automatic reset
            state, (_, reward, is_done, info) = TrajectorySampler.Step_environment(
                env=env, state=state, action=action
            )

            # Build a single-entry memory object with the sample
            sample = Memory.build(
                states=observation,
                actions=action,
                rewards=reward,
                dones=is_done,
                values=value,
                log_prob_actions=log_prob_action,
                infos=info,
            )

            # Store the new sample
            memory = jax.tree_map(
                lambda stacked, leaf: stacked.at[idx].set(leaf),
                memory,
                TrajectorySampler.Memory_to_memory_1D(sample=sample),
            )

            return state, memory

        state, memory = jax.lax.fori_loop(
            lower=0,
            upper=horizon_length,
            body_fun=body_fun,
            init_val=carry_init,
        )

        return memory, state

    @staticmethod
    def Step_environment(
        env: Env, state: EnvironmentState, action: Action
    ) -> Tuple[EnvironmentState, Tuple[Observation, Reward, IsDone, Info]]:
        # Step the environment
        state, (observation, reward, is_done, info) = env.step(
            action=action, state=state
        )

        # Automatically reset the environment if done
        state = jax.lax.cond(
            pred=is_done,
            true_fun=lambda: env.reset(state=state)[0],
            false_fun=lambda: state,
        )

        return state, (observation, reward, is_done, info)

    @staticmethod
    def Memory_to_memory_1D(sample: Memory) -> Memory:
        # Fix Memory to handle just one sample (S=1)
        def memory_to_sample(leaf):
            l = leaf.squeeze()
            return jnp.array([l]) if l.ndim == 0 else l

        return jax.tree_map(memory_to_sample, sample)

    @staticmethod
    def Zero_memory_sample(env: Env, state: EnvironmentState, agent: Agent) -> Memory:
        _, obs = env.reset(state=state)

        action, log_prob_action, value = agent.choose_action(
            observation=obs.flatten(), explore=False, key=state.key
        )

        _, (obs, reward, is_done, info) = env.step(action=action, state=state)

        sample = Memory.build(
            states=obs.flatten(),
            actions=action,
            rewards=reward,
            dones=is_done,
            values=value,
            log_prob_actions=log_prob_action,
            infos=info,
        )

        zero_sample = jax.tree_map(lambda l: jnp.zeros_like(l), sample)

        return TrajectorySampler.Memory_to_memory_1D(sample=zero_sample)
