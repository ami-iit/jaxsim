import dataclasses
import functools
import time
from typing import Any, Callable, Dict, Tuple, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses
import numpy.typing as npt
from flax.core.frozen_dict import FrozenDict
from rich.console import Console

import jaxsim.typing as jtp
from jaxsim.simulation import systems as sys
from jaxsim.training.agent import Agent
from jaxsim.training.memory import Memory


@jax_dataclasses.pytree_dataclass
class EnvironmentSamplerBuffer:
    key: jtp.ArrayJax
    state: FrozenDict


@dataclasses.dataclass
class EnvironmentSampler:
    parallel: bool
    environment: sys.EnvironmentSystem

    duration: float
    horizon: jtp.VectorJax

    _buffer: EnvironmentSamplerBuffer
    _sample_function: Callable = jax_dataclasses.field(default=None)

    def get_sample_function(
        self,
    ) -> Callable[
        [FrozenDict, jtp.ArrayJax, Dict[str, Any]], Tuple[Memory, sys.EnvironmentSystem]
    ]:
        """
        Build the sampling function for either single or parallel environment.

        Returns:
            The JIT-compiled sampling function.
        """

        if self._sample_function is not None:
            return self._sample_function

        # We split the arguments batched and non-batched.
        # The non-batched arguments will be stored in a kwargs dictionary.
        func = lambda x0, key, kwargs: EnvironmentSampler.step_environment_over_horizon(
            x0=x0, key=key, **kwargs
        )

        if not self.parallel:
            self._sample_function = func
            return self._sample_function

        # Thanks to the definition of the lambda, specifying the batched axes
        # is much more simple in vmap
        self._sample_function = jax.jit(jax.vmap(fun=func, in_axes=(0, 0, None)))
        return self._sample_function

    def _sample(
        self, x0: FrozenDict, key: jtp.Array, extra_args: Dict[str, Any] = None
    ) -> Tuple[Memory, sys.EnvironmentSystem]:
        """
        Call the JIT-compiled sampling functions.

        The first call this method logs the time spent to JIT-compile the function.

        Args:
            x0: The state of the (possibly parallelized) system.
            key: The key of the (possibly parallelized) system.
            extra_args: The non-batch arguments of the system

        Returns:
            The sampled memory and the system with the final state.
        """

        extra_args = extra_args if extra_args is not None else dict()

        if self._sample_function is not None:
            return self.get_sample_function()(x0, key, extra_args)

        console = Console()

        with console.status("[bold green]JIT compiling sampling function...") as _:
            start = time.time()
            out = self.get_sample_function()(x0, key, extra_args)
            elapsed = time.time() - start
            console.log(f"JIT compiled sampler in {elapsed:.3f} seconds.")

        return out

    @staticmethod
    def build(
        environment: sys.EnvironmentSystem,
        t0: float = 0.0,
        tf: float = 5.0,
        dt: float = 0.001,
        seed: int = 0,
        parallel_environments: int = 1,
    ) -> "EnvironmentSampler":
        def build_state_and_key_parallel() -> EnvironmentSamplerBuffer:
            env_list = [
                environment.reset_environment(environment.seed(seed=i + 1))
                for i in range(parallel_environments)
            ]

            def tree_transpose(list_of_trees):
                # return jax.tree_multimap(lambda *xs: jnp.stack(xs), *list_of_trees)
                return jax.tree_map(lambda *xs: jnp.stack(xs), *list_of_trees)

            keys_list = [env.key for env in env_list]
            states_list = [env.state_subsystem for env in env_list]

            key = tree_transpose(keys_list)
            state = tree_transpose(states_list)

            return EnvironmentSamplerBuffer(state=state, key=key)  # noqa

        def build_state_and_key_not_parallel() -> EnvironmentSamplerBuffer:
            env = environment.reset_environment(environment.seed(seed=seed))

            key = env.key
            state = env.state_subsystem

            return EnvironmentSamplerBuffer(state=state, key=key)  # noqa

        buffer = (
            build_state_and_key_parallel()
            if parallel_environments > 1
            else build_state_and_key_not_parallel()
        )

        # Build the sampler
        sampler = EnvironmentSampler(
            environment=environment,
            parallel=parallel_environments > 1,
            horizon=jnp.arange(start=t0, stop=tf, step=dt),
            duration=tf - t0,
            _buffer=buffer,
        )

        return sampler

    def sample(self, agent: Agent, explore: bool = True) -> Memory:
        # Update the horizon
        horizon = self.horizon + self.duration

        # Build the dict of non-batched arguments
        inputs_dict = dict(
            system=self.environment, agent=agent, t=horizon, explore=explore
        )

        # Sample from the environment
        memory, environment = self._sample(
            self._buffer.state, self._buffer.key, inputs_dict
        )

        # Create the new buffer making sure that shapes didn't change
        with jax_dataclasses.copy_and_mutate(self._buffer) as buffer:
            buffer.key = environment.key
            buffer.state = environment.state_subsystem

        # Double check that the state dict didn't change
        assert jax.tree_structure(self._buffer.state) == jax.tree_structure(
            buffer.state
        )

        # Update the sampler state
        self.horizon = horizon
        self._buffer = buffer

        # Return the sampled memory
        return memory

    @staticmethod
    @functools.partial(jax.jit)
    def run_actor(
        agent: Agent, environment: sys.EnvironmentSystem, explore: bool = True
    ) -> Tuple[jtp.VectorJax, jtp.VectorJax, jtp.VectorJax, Agent]:
        key, agent = agent.advance_key2()

        observation = environment.get_observation(system=environment)

        action, log_prob_action, value = agent.choose_action(
            observation=observation,
            explore=jnp.array(explore).any(),
            key=key,
        )

        return action, log_prob_action, value, agent

    @staticmethod
    @functools.partial(jax.jit)
    def step_environment_over_horizon(
        system: sys.EnvironmentSystem,
        agent: Agent,
        t: npt.NDArray,
        x0: FrozenDict = None,
        key: jax.random.PRNGKey = None,
        explore: Union[bool, jtp.VectorJax] = jnp.array(True),
    ) -> Tuple[Memory, sys.EnvironmentSystem]:
        # Handle state and key
        system = system if x0 is None else system.update_subsystem_state(new_state=x0)
        system = system if key is None else system.update_key(key=key)

        # Compute a dummy output of the system, used to initialize the buffer.
        # We flatten the output (dict) in order to allocate a dense jax array
        # used inside a JIT-compiled for loop.
        out, system = system(t0=t[0], tf=t[0], u0=None)
        out_flattened, restore_output_fn = jax.flatten_util.ravel_pytree(out)

        # Create the buffers storing environment data
        values = jnp.zeros(shape=(t.size, 1))
        log_prob_actions = jnp.zeros(shape=(t.size, 1))
        system_output = jnp.zeros(shape=(t.size, out_flattened.size))

        # Generate a new key from the environment used for sampling actions
        agent_key, system = system.generate_subkey()
        agent = jax_dataclasses.replace(agent, key=agent_key)  # noqa

        # Initialize the loop carry
        carry_init = (system_output, log_prob_actions, values, system, agent)

        def body_fun(i: int, carry: Tuple) -> Tuple:
            # Unpack the loop carry
            system_output, log_prob_actions, values, system, agent = carry

            # Execute the actor
            (action, log_prob_action, value, agent) = EnvironmentSampler.run_actor(
                agent=agent, environment=system, explore=explore
            )

            # Update values and log_prob
            values = values.at[i].set(value)
            log_prob_actions = log_prob_actions.at[i].set(log_prob_action)

            # Advance the environment and get its output
            out, system = system(t0=t[i], tf=t[i + 1], u0=FrozenDict(action=action))

            # Store the environment output in the buffer
            out_flattened, _ = jax.flatten_util.ravel_pytree(out)
            system_output = system_output.at[i, :].set(out_flattened)

            # Return the loop carry
            return system_output, log_prob_actions, values, system, agent

        # Execute the rollout. The environment automatically resets when it's done.
        system_output, log_prob_actions, values, system, agent = jax.lax.fori_loop(
            lower=0,
            upper=t.size,
            body_fun=body_fun,
            init_val=carry_init,
        )

        # Unflatten the output
        output_horizon: FrozenDict = jax.vmap(lambda b: restore_output_fn(b))(
            system_output
        )

        # Create the memory object
        memory = Memory.build(
            states=output_horizon["observation"],
            actions=output_horizon["action"],
            rewards=output_horizon["reward"],
            dones=output_horizon["done"],
            values=values,
            log_prob_actions=log_prob_actions,
            infos=output_horizon["info"],
        )

        # Return objects after stepping over the horizon
        return memory, system
