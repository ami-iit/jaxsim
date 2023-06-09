import copy
import dataclasses
import datetime
import functools
import pathlib
from typing import Any, Callable, Dict, Tuple, Union

import flax.training.checkpoints
import flax.training.train_state
import gym.spaces
import jax
import jax.experimental.loops
import jax.numpy as jnp
import jax_dataclasses
import optax
from flax.core.frozen_dict import FrozenDict
from optax._src.alias import ScalarOrSchedule

import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.utils import JaxsimDataclass

from .memory import DataLoader, Memory
from .networks import ActorCriticNetworks, ActorNetwork, CriticNetwork


@jax_dataclasses.pytree_dataclass
class PPOParams:
    # Gradient descent
    alpha: float = jax_dataclasses.field(default=0.0003)
    optimizer: Callable[
        [ScalarOrSchedule], optax.GradientTransformation
    ] = jax_dataclasses.static_field(default=optax.adam)

    # RL Algorithm
    gamma: float = jax_dataclasses.field(default=0.99)
    lambda_gae: float = jax_dataclasses.field(default=0.95)

    # PPO
    # beta_kl: float = jax_dataclasses.static_field(default=0.0)
    beta_kl: float = jax_dataclasses.field(default=0.0)
    target_kl: float = jax_dataclasses.field(default=0.010)
    epsilon_clip: float = jax_dataclasses.field(default=0.2)

    # Other params
    entropy_loss_weight: float = jax_dataclasses.field(default=0.0)


class PPOTrainState(flax.training.train_state.TrainState):
    beta_kl: float


@dataclasses.dataclass(frozen=True)
class CheckpointManager:
    checkpoint_path: pathlib.Path

    def save_best(
        self,
        train_state: PPOTrainState,
        measure: str,
        metric: Union[int, float],
        keep: int = 1,
    ) -> None:
        checkpoint_path = self.checkpoint_path / f"best_{measure}"

        path_to_checkpoint = flax.training.checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_path,
            target=train_state,
            step=metric,
            prefix=f"checkpoint_",
            keep=keep,
            overwrite=True,
        )

        logging.info(msg=f"Saved checkpoint: {path_to_checkpoint}")

    def save_latest(
        self, train_state: PPOTrainState, keep_every_n_steps: int = None
    ) -> None:
        path_to_checkpoint = flax.training.checkpoints.save_checkpoint(
            ckpt_dir=self.checkpoint_path,
            target=train_state,
            step=train_state.step,
            prefix=f"checkpoint_",
            keep=1,
            keep_every_n_steps=keep_every_n_steps,
        )

        logging.info(msg=f"Saved checkpoint: {path_to_checkpoint}")

    def load_best(
        self, dummy_train_state: PPOTrainState, measure: str
    ) -> PPOTrainState:
        checkpoint_path = self.checkpoint_path / f"best_{measure}"

        train_state = flax.training.checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=dummy_train_state,
            prefix=f"checkpoint_",
        )

        return train_state

    def load_latest(self, dummy_train_state: PPOTrainState) -> PPOTrainState:
        train_state = flax.training.checkpoints.restore_checkpoint(
            ckpt_dir=self.checkpoint_path,
            target=dummy_train_state,
            prefix=f"checkpoint_",
        )

        return train_state


@jax_dataclasses.pytree_dataclass
class Agent(JaxsimDataclass):
    key: jax.random.PRNGKey = jax_dataclasses.field(
        default_factory=lambda: jax.random.PRNGKey(seed=0), repr=False
    )

    params: PPOParams = jax_dataclasses.field(default_factory=PPOParams)
    train_state: PPOTrainState = jax_dataclasses.field(default=None, repr=False)

    action_space: gym.spaces.Box = jax_dataclasses.static_field(default=None)
    observation_space: gym.spaces.Box = jax_dataclasses.static_field(default=None)

    checkpoint_manager: CheckpointManager = jax_dataclasses.static_field(default=None)

    _num_timesteps: int = jax_dataclasses.field(default=0)
    _num_iterations: int = jax_dataclasses.field(default=0)

    _min_reward: float = jnp.finfo(jnp.float32).max
    _max_reward: float = jnp.finfo(jnp.float32).min

    @staticmethod
    def build(
        actor: ActorNetwork = None,
        critic: CriticNetwork = None,
        action_space: gym.spaces.Space = None,
        observation_space: gym.spaces.Space = None,
        key: jax.random.PRNGKey = jax.random.PRNGKey(seed=0),
        params: PPOParams = PPOParams(),
        train_state: PPOTrainState = None,  # TODO remove?
        checkpoint_label: str = "training",
        load_checkpoint_from_path: pathlib.Path = None,
    ) -> "Agent":
        date_str = datetime.datetime.now().strftime("%y%m%d_%H%M")
        checkpoint_folder = f"{checkpoint_label}_{date_str}"

        checkpoint_path = (
            pathlib.Path("~/jaxsim_results").expanduser() / checkpoint_folder
        )

        agent = Agent(
            key=key,
            params=params,
            train_state=train_state,
            action_space=action_space,
            observation_space=observation_space,
            checkpoint_manager=CheckpointManager(checkpoint_path),
        )

        if agent.train_state is None:
            # Generate a new network RNG key
            key = agent.advance_key(num_sub_keys=1)

            # Get a dummy observation
            observation_dummy = jnp.zeros(shape=observation_space.shape)

            # Create the actor/critic network
            actor_critic = ActorCriticNetworks(actor=actor, critic=critic)

            # Initialize the actor/critic network
            out, params_critic = actor_critic.init_with_output(key, observation_dummy)
            distribution_dummy, value_dummy = out

            # Check value shape
            if value_dummy.shape != (1,):
                raise ValueError(value_dummy.shape, (1,))

            # Check action shape
            dummy_action = distribution_dummy.sample(seed=key)
            if dummy_action.shape != action_space.shape:
                raise ValueError(dummy_action.shape, action_space.shape)

            # Initialize the actor/critic train state
            train_state = PPOTrainState.create(
                apply_fn=actor_critic.apply,
                params=params_critic,
                tx=agent.params.optimizer(agent.params.alpha),
                beta_kl=agent.params.beta_kl,
            )

            with agent.editable(validate=False) as agent:
                agent.train_state = train_state

            # Replace the actor/critic train state
            # agent = jax_dataclasses.replace(agent, train_state=train_state)  # noqa

        if load_checkpoint_from_path is not None:
            load_checkpoint_from_path = (
                load_checkpoint_from_path.expanduser().absolute()
            )

            if not load_checkpoint_from_path.exists():
                raise FileExistsError(load_checkpoint_from_path)

            with agent.editable(validate=False) as agent:
                agent.train_state = flax.training.checkpoints.restore_checkpoint(
                    ckpt_dir=load_checkpoint_from_path,
                    target=agent.train_state,
                    prefix=f"checkpoint_",
                )

        return agent

    # /tmp/jaxsim/cartpole_20220614_1756/checkpoint_#i
    # /tmp/jaxsim/cartpole_20220614_1756/reward/checkpoint_REWARD
    # /tmp/jaxsim/cartpole_20220614_1756/episode_steps/checkpoint_REWARD

    def save_checkpoint(
        self,
        prefix: str = "checkpoint_",
        checkpoint_path: pathlib.Path = pathlib.Path.cwd() / "checkpoints",
    ) -> None:
        path_to_checkpoint = flax.training.checkpoints.save_checkpoint(
            target=self.train_state,
            prefix=prefix,
            ckpt_dir=checkpoint_path,
            step=self.train_state.step,
        )

        logging.info(msg=f"Save checkpoint: {path_to_checkpoint}")

    def load_checkpoint(
        self, checkpoint_path: pathlib.Path = pathlib.Path.cwd() / "checkpoints"
    ) -> "Agent":
        train_state = flax.training.checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path, target=self.train_state
        )

        with self.editable(validate=True) as agent:
            agent.train_state = train_state

        agent._set_mutability(mutability=self._mutability())
        return agent

        # return jax_dataclasses.replace(self, train_state=train_state)

    def advance_key(self, num_sub_keys: int = 1) -> jax.random.PRNGKey:
        keys = jax.random.split(key=self.key, num=num_sub_keys + 1)

        object.__setattr__(self, "key", keys[0])

        return keys[1:].squeeze()

    def advance_key2(self, num_sub_keys: int = 1) -> Tuple[jax.random.PRNGKey, "Agent"]:
        keys = jax.random.split(key=self.key, num=num_sub_keys + 1)
        return keys[1:].squeeze(), jax_dataclasses.replace(self, key=keys[0])  # noqa

    def choose_action(
        self,
        observation: jtp.Vector,
        explore: bool = True,
        key: jax.random.PRNGKey = None,
    ) -> Tuple[jtp.VectorJax, jtp.VectorJax, jtp.VectorJax]:
        # Get a new key if not passed
        key = key if key is not None else self.advance_key()

        # Infer π_θ(⋅|sₜ) and V_ϕ(sₜ)
        distribution, value = self.train_state.apply_fn(
            self.train_state.params, data=observation
        )

        # Sample an action from π_θ(⋅|sₜ)
        action = jax.lax.select(
            pred=explore,
            on_true=distribution.sample(seed=key),
            on_false=distribution.mode(),
        )

        # Compute log-likelihood of action: log[π_θ(aₜ|sₜ)]
        log_prob_action = distribution.log_prob(value=action)

        return action, log_prob_action, value
        # return (
        #     jnp.array(action).squeeze(),
        #     jnp.array(log_prob_action).squeeze(),
        #     jnp.array(value).squeeze(),
        # )

    @staticmethod
    @functools.partial(jax.jit)
    def estimate_advantage_gae_jit(
        train_state: PPOTrainState, memory: Memory, gamma: float, lambda_gae: float
    ) -> jtp.VectorJax:
        # Closure that computes V(o). Used to boostrap the return when necessary.
        value_of = lambda observation: train_state.apply_fn(
            train_state.params, data=observation
        )[1].squeeze()

        return Agent.estimate_advantage_gae(
            memory=memory, gamma=gamma, lambda_gae=lambda_gae, V=value_of
        )

    @staticmethod
    def estimate_advantage_gae(
        memory: Memory,
        gamma: float,
        lambda_gae: float,
        V: Callable[[jtp.ArrayJax], jtp.ArrayJax] = lambda _: 0.0,
    ) -> jtp.VectorJax:
        r = memory.rewards
        mask = 1 - memory.dones.astype(dtype=int)

        # Extract additional data from the info dictionary
        is_terminal = memory.infos["is_terminal"]
        next_observations = memory.infos["terminal_observation"]

        # The last trajectory in the memory is likely truncated, i.e. is_done[-1] = 0.
        # In order to estimate Â, we get the next observation stored in the info dict
        # and boostrap the return of the last observation of the trajectory with TD(0).
        Vs = jnp.hstack([memory.values.squeeze(), V(next_observations[-1])])

        with jax.experimental.loops.Scope() as s:
            # Allocate the Âₜ array (adding a trailing zero, as Vs).
            # We cannot know how the trajectory will continue from the truncated last
            # trajectory of the memory. In this case, since we have computed the value
            # of the next observation, we estimate A of the last sample using TD(0)
            # returns, i.e. Âₕ = δₕ = rₕ + γ⋅V(sₕ₊₁) - V(sₕ).
            # This can be done either setting Âₕ₊₁ = 0 or, equivalently, λₕ = 0.
            # We proceed with the former case.
            s.A = jnp.zeros_like(Vs)

            # Iteration same as: reversed(mask.size)
            for t in s.range(mask.size - 1, -1, -1):
                # Classic TD(0) boostrap: δₜ = rₜ + γ⋅V(sₜ₊₁) - V(sₜ)
                delta_t_not_done = r[t] + gamma * Vs[t + 1] - Vs[t]

                # Use one step of Monte Carlo if terminal, TD(0) otherwise.
                # This allows to consider two different types of termination:
                # - MC: Reached a terminal state s_T. All the rewards following s_T
                #       are considered to be 0.
                # - TD(0): The trajectory reached the maximum length, and it was
                #          truncated early (common in continuous control). We cannot
                #          assume that the reward would be 0 after truncation, therefore
                #          we boostrap the return with TD(0) (similarly to what we do
                #          for the last truncated trajectory of the memory).
                delta_t_done = jax.lax.select(
                    pred=is_terminal.astype(dtype=bool)[t],
                    on_true=r[t] - Vs[t],
                    on_false=r[t] + gamma * V(next_observations[t]) - Vs[t],
                )

                # Select the right δₜ
                delta_t = jax.lax.select(
                    pred=memory.dones.astype(dtype=bool)[t],
                    on_true=delta_t_done,
                    on_false=delta_t_not_done,
                )

                # Compute the advantage estimate Âₜ
                A_t = delta_t + gamma * lambda_gae * s.A[t + 1] * mask[t]
                s.A = s.A.at[t].set(A_t.squeeze())

            # Remove the trailing value we added to handle the last entry of the memory
            A = s.A[:-1]

        return jax.lax.stop_gradient(jnp.vstack(A))

    @staticmethod
    def compute_reward_to_go(
        memory: Memory,
        gamma: float = 1.0,
        V: Callable[[jtp.ArrayJax], jtp.ArrayJax] = lambda _: 0.0,
    ) -> jtp.VectorJax:
        assert memory.flat is True

        r = memory.rewards.squeeze()
        r_to_go = jnp.zeros_like(r)

        dones = memory.dones.astype(dtype=bool).squeeze()
        is_terminal = memory.infos["is_terminal"].astype(dtype=bool).squeeze()
        next_observations = memory.infos["terminal_observation"].squeeze()

        with jax.experimental.loops.Scope() as s:
            # Approximate the return of the state following the last one with its value.
            # Note: we store next_observation in the info dict for this reason.
            s.r_to_go = jnp.hstack([r_to_go, V(next_observations[-1])])

            # Iteration same as: reversed(dones.size)
            for t in s.range(dones.size - 1, -1, -1):
                # Monte Carlo discounted accumulation.
                # Note: considering r_to_go[-1], this is TD(0) for the last memory entry.
                r_to_go_not_done = r[t] + gamma * s.r_to_go[t + 1]

                # If done, use one step of Monte Carlo if terminal, TD(0) otherwise
                r_to_go_done = jax.lax.select(
                    pred=is_terminal[t],
                    on_true=r[t],
                    on_false=r[t] + gamma * V(next_observations[t]),
                )

                # Select the right Rₜ
                r_to_go_t = jax.lax.select(
                    pred=dones[t],
                    on_true=r_to_go_done,
                    on_false=r_to_go_not_done,
                )

                # Store Rₜ in the buffer
                s.r_to_go = s.r_to_go.at[t].set(r_to_go_t.squeeze())

            # Remove the trailing value we added to handle the last entry of the memory
            r_to_go = s.r_to_go[:-1]

        return jax.lax.stop_gradient(jnp.vstack(r_to_go))

    @staticmethod
    @functools.partial(jax.jit)
    def compute_reward_to_go_jit(
        train_state: PPOTrainState, memory: Memory, gamma: float
    ) -> jtp.VectorJax:
        # Closure that computes V(o). Used to boostrap the return when necessary.
        value_of = lambda observation: train_state.apply_fn(
            train_state.params, data=observation
        )[1].squeeze()

        return Agent.compute_reward_to_go(memory=memory, gamma=gamma, V=value_of)

    @staticmethod
    def explained_variance(y_hat: jtp.Array, y: jtp.Array) -> jtp.Array:
        assert y_hat.ndim == y.ndim == 1

        var_y = jnp.var(y)

        return jax.lax.select(
            pred=(var_y == 0.0),
            on_true=jnp.nan,
            on_false=(1 - jnp.var(y - y_hat) / var_y),
        )

    @staticmethod
    @functools.partial(jax.jit)
    def train_actor_critic(
        train_state_target: PPOTrainState,
        train_state_behavior: PPOTrainState,
        memory: Memory,
        returns_target: jtp.VectorJax,
        policy_gradient_loss_weight: jtp.VectorJax,
        ppo_params: PPOParams = PPOParams(),
    ) -> Tuple[PPOTrainState, Dict]:
        # Adjust 1D arrays
        returns_target = returns_target.squeeze()
        policy_gradient_loss_weight = policy_gradient_loss_weight.squeeze()

        # Assume memory has vertical 1D arrays
        mem_values = jnp.vstack(memory.values.squeeze())
        mem_actions = jnp.vstack(memory.actions.squeeze())
        mem_observations = jnp.vstack(memory.states.squeeze())
        mem_log_prob_actions = jnp.vstack(memory.log_prob_actions.squeeze())

        # Loss function for both the actor and critic networks.
        # Note: we do not support sharing layers, therefore weighting differently
        #       the two losses should not be relevant.
        def loss_fn(params: flax.core.FrozenDict[str, Any]) -> Tuple[float, Dict]:
            # Infer π_θₙ(⋅|sₜ) and V_ϕₙ(sₜ) with the new parameters (θₙ, ϕₙ)
            new_distributions, new_values = train_state_target.apply_fn(
                params,
                data=mem_observations,
            )

            # ======
            # Critic
            # ======

            # The loss uses the returns as targets. Returns could be computed with
            # a (possibly discounted) reward-to-go or from the estimated advantage.

            # Compute the MSE loss
            new_values = new_values.squeeze()
            critic_loss = jnp.linalg.norm(new_values - returns_target, ord=2)
            # TODO: should this be norm squared?

            # Compute the explained variance. It should start as a very negative number
            # and progressively converge towards 1.0 when the value function learned to
            # approximate correctly the sampled return.
            returns = policy_gradient_loss_weight.flatten() + mem_values.flatten()
            explained_variance = Agent.explained_variance(
                y=returns, y_hat=mem_values.flatten()
            )

            # =====
            # Actor
            # =====

            # Refer to https://arxiv.org/abs/1707.06347 for the surrogate functions
            # used for both the CLIP and the KLPEN versions of PPO.

            # Infer π_θₒ(⋅|sₜ) with the old parameters θₒ. Used only for KLPEN.
            old_distributions, old_values = train_state_behavior.apply_fn(
                train_state_behavior.params,
                data=mem_observations,
            )

            # Compute new log-likelihood of actions: log[π_θₙ(aₜ|sₜ)]
            new_log_prob_actions = new_distributions.log_prob(value=mem_actions)

            # Rename the old log-likelihood of actions: log[π_θₒ(aₜ|sₜ)]
            old_log_prob_actions = mem_log_prob_actions.squeeze()

            # Compute the ratio rₜ(θₙ) of the likelihoods
            prob_action_ratio = jnp.exp(new_log_prob_actions - old_log_prob_actions)

            # Compute the CPI surrogate objective
            L = prob_action_ratio * policy_gradient_loss_weight

            # Compute the clipped version of the ratio of the likelihoods
            prob_action_ratio_clipped = jnp.clip(
                a=prob_action_ratio,
                a_min=(1.0 - ppo_params.epsilon_clip),
                a_max=(1.0 + ppo_params.epsilon_clip),
            )

            # Compute the clip ratio
            clipped_elements = jnp.where(
                prob_action_ratio != prob_action_ratio_clipped, 1, 0
            )
            clip_ratio = clipped_elements.sum() / clipped_elements.size

            # Apply the CLIP surrogate objective.
            # Note: 'epsilon_clip' is zero if CLIP is not enabled.
            L = jax.lax.select(
                pred=(ppo_params.epsilon_clip == 0),
                on_true=L,
                on_false=jnp.minimum(
                    L, prob_action_ratio_clipped * policy_gradient_loss_weight
                ),
            ).mean()

            # Compute the additional KLPEN surrogate objective term.
            # Note: 'beta_kl' is zero if KLPEN is not enabled.
            distr_kl = old_distributions.kl_divergence(other_dist=new_distributions)
            ppo_klpen_term = train_state_behavior.beta_kl * distr_kl.mean()

            # Apply the KLPEN surrogate objective term
            L -= ppo_klpen_term

            # Compute the loss to minimize from the surrogate objective
            actor_loss = -L

            # Optional entropy reward
            entropy_mean = new_distributions.entropy().mean()
            entropy_reward = ppo_params.entropy_loss_weight * entropy_mean

            return (
                actor_loss - entropy_reward + 0.100 * critic_loss,
                dict(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    entropy=entropy_mean,
                    entropy_reward=entropy_reward,
                    kl=distr_kl.mean(),
                    beta_kl=train_state_behavior.beta_kl,
                    clip_ratio=clip_ratio,
                    explained_variance=explained_variance,
                ),
            )

        # Commented-out code to check gradients wrt finite differences
        # from jax.test_util import check_grads
        # check_grads(loss_fn, (train_state_target.params,), order=1, eps=1e-4)

        # Compute the loss and its gradient wrt the NN parameters
        (total_loss, loss_fn_data), grads = jax.value_and_grad(
            fun=loss_fn, has_aux=True
        )(train_state_target.params)

        # Pass the gradient to the optimizer and get a new state
        new_train_state = train_state_target.apply_gradients(grads=grads)

        return new_train_state, dict(total_loss=total_loss, **loss_fn_data)

    @staticmethod
    @functools.partial(jax.jit)
    def adaptive_update_beta_ppo_kl_pen(
        train_state_behavior: PPOTrainState,
        train_state_target: PPOTrainState,
        ppo_params: PPOParams,
        memory: Memory,
    ) -> PPOTrainState:
        # Refer to https://arxiv.org/abs/1707.06347 (Sec. 4) for the update rule of β_KL.
        # We use the default heuristic 1.5 and 2 parameters as reported in the paper.

        # Infer π_θₙ(⋅|sₜ) with the new parameters
        new_distributions, _ = train_state_target.apply_fn(
            train_state_target.params, data=memory.states
        )

        # Infer π_θₒ(⋅|sₜ) with the old parameters
        old_distributions, _ = train_state_behavior.apply_fn(
            train_state_behavior.params, data=memory.states
        )

        # Compute the KL divergence of the old policy from the new policy
        kl = old_distributions.kl_divergence(other_dist=new_distributions).mean()

        # Get the old β_KL used as weight of PPO-KLPEN surrogate objective
        beta_kl = train_state_behavior.beta_kl

        # Increase β_KL
        train_state_target = jax.lax.cond(
            pred=(kl > ppo_params.target_kl * 1.5),
            true_fun=lambda _: train_state_target.replace(beta_kl=beta_kl * 2.0),
            false_fun=lambda _: train_state_target,
            operand=(),
        )

        # Decrease β_KL
        train_state_target = jax.lax.cond(
            pred=(kl < ppo_params.target_kl / 1.5),
            true_fun=lambda _: train_state_target.replace(beta_kl=beta_kl * 0.5),
            false_fun=lambda _: train_state_target,
            operand=(),
        )

        return train_state_target

    def train(
        self,
        memory: Memory,
        num_epochs: int = 1,
        batch_size: int = 512,
        print_report: bool = False,
    ) -> "Agent":
        # Truncate the last trajectory by modifying the last is_done. In this way we can
        # bootstrap correctly its return considering the last sample as non-terminal.
        # Also flatten the memory if it was sampled from parallel environments.
        memory_flat = memory.truncate_last_trajectory().flatten()

        mean_reward = memory_flat.rewards.mean()
        self.checkpoint_manager.save_best(
            train_state=self.train_state, measure="reward", metric=mean_reward, keep=5
        )

        self.checkpoint_manager.save_latest(
            train_state=self.train_state, keep_every_n_steps=25
        )

        # Update the training metadata
        with self.editable(validate=True) as agent:
            agent._num_iterations += 1
            agent._num_timesteps += len(memory)

        # Update the training metadata
        # agent = jax_dataclasses.replace(
        #     self,  # noqa
        #     _num_iterations=(self._num_iterations + 1),
        #     _num_timesteps=(self._num_timesteps + len(memory)),
        # )

        # Log the min/max reward ever seen
        min_reward = jnp.array([agent._min_reward, memory.rewards.min()]).min()
        max_reward = jnp.array([agent._max_reward, memory.rewards.max()]).max()
        agent = jax_dataclasses.replace(
            agent, _min_reward=min_reward, _max_reward=max_reward  # noqa
        )

        # Estimate the advantages Â_π_θₒ(sₜ, aₜ) with GAE used to train the actor.
        advantages = Agent.estimate_advantage_gae_jit(
            train_state=agent.train_state,
            memory=memory_flat,
            gamma=agent.params.gamma,
            lambda_gae=agent.params.lambda_gae,
        )

        # Compute the rewards-to-go R̂ₜ used to train the critic.
        # Note: same of estimating Â_GAE with λ=1.0 and γ=1.0.
        rewards_to_go = Agent.compute_reward_to_go_jit(
            train_state=agent.train_state,
            memory=memory_flat,
            gamma=agent.params.gamma,
        )

        # Select the returns to use. We can use any of the following:
        # - Rewards-to-go: R = R̂ₜ
        # - GAE advantages plus values: R = Â_GAE + V
        # returns = rewards_to_go
        returns = advantages + memory_flat.values

        # Store the behavior train state (old parameters θₒ and ϕₒ).
        # Note: for safety, we make sure that params do not get overridden by the
        #       next optimization by taking their deep copy.
        train_state_behaviour = agent.train_state.replace(
            params=copy.deepcopy(agent.train_state.params)
        )

        for epoch_idx in range(num_epochs):
            for batch_slice in DataLoader(memory=memory_flat).batch_slices_generator(
                batch_size=batch_size,
                shuffle=True,
                seed=epoch_idx,
                allow_partial_batch=False,
            ):
                # Perform one step of gradient descent
                train_state, extra_data = Agent.train_actor_critic(
                    train_state_behavior=train_state_behaviour,
                    train_state_target=agent.train_state,
                    memory=memory_flat[batch_slice],
                    returns_target=returns[batch_slice],
                    policy_gradient_loss_weight=advantages[batch_slice],
                    ppo_params=agent.params,
                )

                # Update the agent with the new train state
                agent = jax_dataclasses.replace(agent, train_state=train_state)

                # TODO
                # Create the log data of the training step
                log_data = FrozenDict(
                    extra_data, reward_range=(agent._min_reward, agent._max_reward)
                )

                # Print to output
                # TODO: logging?
                print(log_data)

        # ======================================
        # Update the KL parameters for PPO-KLPEN
        # ======================================

        # @jax.jit
        # def update_beta_kl(agent: Agent) -> Agent:
        #
        #     # Update the train state with the adjusted β_KL
        #     train_state_kl = Agent.adaptive_update_beta_ppo_kl_pen(
        #         train_state_behavior=train_state_behaviour,
        #         train_state_target=agent.train_state,
        #         ppo_params=agent.params,
        #         memory=memory_flat,
        #     )
        #
        #     # Update the agent with the new train state
        #     return jax_dataclasses.replace(agent, train_state=train_state_kl)

        # agent = jax.lax.cond(
        #     pred=agent.params.beta_kl != 0.0,
        #     true_fun=update_beta_kl,
        #     false_fun=lambda agent: agent,
        #     operand=agent,
        # )

        if agent.params.beta_kl != 0.0:
            # Update the train state with the adjusted β_KL
            train_state_kl = Agent.adaptive_update_beta_ppo_kl_pen(
                train_state_behavior=train_state_behaviour,
                train_state_target=agent.train_state,
                ppo_params=agent.params,
                memory=memory_flat,
            )

            # Update the agent with the new train state
            agent = jax_dataclasses.replace(agent, train_state=train_state_kl)

        if not print_report:
            return agent

        # ===================
        # Print training data
        # ===================

        avg_length_of_trajectories = []

        for trajectory in memory.trajectories():
            avg_length_of_trajectories.append(trajectory.rewards.size)

        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Iteration #{agent._num_iterations}")
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="cyan", no_wrap=True)

        table.add_row("Epochs", f"{num_epochs}")
        table.add_row("Timesteps", f"{len(memory)}")
        table.add_row("Total timesteps", f"{agent._num_timesteps}")
        table.add_row("Avg reward", f"{float(memory_flat.rewards.mean()):10.4f}")
        console = Console()
        print()
        console.print(table)

        return agent
