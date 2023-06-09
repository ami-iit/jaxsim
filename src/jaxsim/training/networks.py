from typing import Any, Callable, Sequence, Tuple

import distrax
import flax.linen as nn
import gym.spaces
import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.training import distributions


class CriticNetwork(nn.Module):
    layer_sizes: Sequence[int]

    activation: Callable[[jtp.Array], jtp.Array] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.normal()

    bias: bool = True
    activate_final: bool = False

    def setup(self):
        logging.debug("Configuring critic network...")

        # Automatically add a final layer so that the NN outputs a scalar value V(s)
        all_layer_sizes = np.hstack([self.layer_sizes, 1])

        self.layers = [
            nn.Dense(
                features=size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )
            for size in all_layer_sizes
        ]

        logging.debug(f"    Layers: {all_layer_sizes.tolist()}")
        activated = [
            self.activate_layer(layer_idx=idx) for idx, _ in enumerate(all_layer_sizes)
        ]
        logging.debug(f"    Activated: {activated}")
        logging.debug(f"    Activation function: {self.activation.__name__}")
        logging.debug(f"    Kernel initializer: {self.kernel_init}")

    def activate_layer(self, layer_idx: int) -> bool:
        return (layer_idx != len(self.layers) - 1) or self.activate_final

    def __call__(self, data: jtp.Array) -> jtp.Array:
        x = data

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x) if self.activate_layer(layer_idx=idx) else x

        return x


class ActorNetwork(nn.Module):
    layer_sizes: Sequence[int]
    action_space: gym.spaces.Box

    activation: Callable[[jtp.Array], jtp.Array] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.normal()

    bias: bool = True
    activate_final: bool = False

    log_std_min: float = None
    log_std_max: float = jnp.log(1.0)
    log_std_init_val: float = jnp.log(0.50)

    def setup(self) -> None:
        logging.debug("Configuring actor network...")

        # Automatically add a final layer so that the NN outputs an array containing
        # the means μ of the action distribution
        all_layer_sizes = np.hstack([self.layer_sizes, self.action_space.sample().size])

        # Get the index of the last layer
        last_layer_idx = all_layer_sizes.size - 1

        def get_kernel_init(layer_idx: int) -> Callable[..., Any]:
            # For the last layer we use a xavier_normal initializer with a variance
            # 1/100 smaller than the default (which is 1.0)
            last_layer_kernel_init = jax.nn.initializers.normal(stddev=1e-2 / 100)

            return (
                self.kernel_init
                if layer_idx != last_layer_idx
                else last_layer_kernel_init
            )

        # Fully-connected network for the mean μ
        self.layers = [
            nn.Dense(
                features=size,
                kernel_init=get_kernel_init(layer_idx=layer_idx),
                use_bias=self.bias,
            )
            for layer_idx, size in enumerate(all_layer_sizes)
        ]

        # State-independent bias variables for −the log of- the variance σ
        self.log_std = self.param(
            "log_std",
            lambda rng, shape: self.log_std_init_val * jnp.ones(shape),
            self.action_space.sample().size,
        )

        logging.debug(f"    Layers: {all_layer_sizes.tolist()}")
        activated = [
            self.activate_layer(layer_idx=idx) for idx, _ in enumerate(all_layer_sizes)
        ]
        logging.debug(f"    Activated: {activated}")
        logging.debug(f"    Activation function: {self.activation.__name__}")
        logging.debug(f"    Kernel initializer: {self.kernel_init}")
        logging.debug(
            "    log(σ): {:.4f} (min={}, max={})".format(
                self.log_std_init_val, self.log_std_min, self.log_std_max
            )
        )

    def activate_layer(self, layer_idx: int) -> bool:
        return (layer_idx != len(self.layers) - 1) or self.activate_final

    def __call__(self, data: jtp.Array) -> distrax.Distribution:
        x = data

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x) if self.activate_layer(layer_idx=idx) else x

        # The mean μ is the NN output
        mu = x

        # ==========================
        # Distributional exploration
        # ==========================

        # Helper to clip the standard deviation
        def clip_log_std(value: jtp.Vector) -> jtp.Vector:
            return jnp.clip(a=value, a_min=self.log_std_min, a_max=self.log_std_max)

        # Clip log(σ) if limits are defined
        log_std_limits = np.array([self.log_std_min, self.log_std_max])
        log_std = clip_log_std(self.log_std) if log_std_limits.any() else self.log_std

        # Compute σ taking the exponential
        std = jnp.exp(log_std)

        # Return the actor distribution
        return distributions.TanhSquashedMultivariateNormalDiag(
            # return distributions.GaussianSquashedMultivariateNormalDiag(
            loc=mu,
            scale_diag=std,
            low=self.action_space.low,
            high=self.action_space.high,
        )


class ActorCriticNetworks(nn.Module):
    actor: ActorNetwork
    critic: CriticNetwork

    def __call__(self, data: jtp.Array) -> Tuple[distrax.Distribution, jtp.Array]:
        observation = data

        value = self.critic(data=observation)
        distribution = self.actor(data=observation)

        return distribution, value
