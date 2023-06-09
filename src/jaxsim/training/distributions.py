from typing import Optional, Sequence, Tuple, Union

import distrax
import jax.numpy as jnp
import jax.scipy.special
from distrax import MultivariateNormalDiag
from distrax._src.distributions import distribution


class SquashedMultivariateNormalDiagBase(distribution.Distribution):
    def __init__(
        self,
        low: distribution.Array,
        high: distribution.Array,
        loc: Optional[distribution.Array] = None,
        scale_diag: Optional[distribution.Array] = None,
    ):
        self.normal = MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        self.low, _ = jnp.broadcast_arrays(low, self.normal.loc)
        self.high, _ = jnp.broadcast_arrays(high, self.normal.loc)

    def event_shape(self) -> Tuple[int, ...]:
        return self.normal.event_shape

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> distribution.Array:
        samples_normal = self.normal._sample_n(key=key, n=n)
        return self._squash(unsquashed_sample=samples_normal)

    def _squash(self, unsquashed_sample: distribution.Array) -> distribution.Array:
        raise NotImplementedError

    def _unsquash(self, squashed_sample: distribution.Array) -> distribution.Array:
        raise NotImplementedError

    def _log_abs_det_grad_squash(
        self, unsquashed_sample: distribution.Array
    ) -> distribution.Array:
        raise NotImplementedError

    def entropy(self) -> distribution.Array:
        raise NotImplementedError

    def mean(self) -> distribution.Array:
        mean_normal = self.normal.mean()
        return self._squash(unsquashed_sample=mean_normal)

    def median(self) -> distribution.Array:
        median_normal = self.normal.median()
        return self._squash(unsquashed_sample=median_normal)

    def variance(self) -> distribution.Array:
        raise NotImplementedError

    def stddev(self) -> distribution.Array:
        raise NotImplementedError

    def mode(self) -> distribution.Array:
        mode_normal = self.normal.mode()
        return self._squash(unsquashed_sample=mode_normal)

    def cdf(self, value: distribution.Array) -> distribution.Array:
        raise NotImplementedError

    def log_cdf(self, value: distribution.Array) -> distribution.Array:
        raise NotImplementedError

    def sample(
        self,
        *,
        seed: Union[distribution.IntLike, distribution.PRNGKey],
        sample_shape: Union[distribution.IntLike, Sequence[distribution.IntLike]] = ()
    ) -> distribution.Array:
        # Sample from the MultivariateNormal distribution
        unsquashed_sample = self.normal.sample(seed=seed, sample_shape=sample_shape)

        # Squash the sample into [low, high]
        squashed_sample = self._squash(unsquashed_sample=unsquashed_sample)
        assert squashed_sample.shape == unsquashed_sample.shape

        return squashed_sample

    def log_prob(self, value: distribution.Array) -> distribution.Array:
        # Unsquash from [low, high] to ]-∞, ∞[
        value_unsquashed = self._unsquash(squashed_sample=value)

        # Compute the log-prob of the underlying normal distribution
        log_prob_norm = self.normal.log_prob(value=value_unsquashed)

        # Compute the correction term due to squashing
        log_abs_det_grad_squash = self._log_abs_det_grad_squash(
            unsquashed_sample=value_unsquashed
        )

        # Adjust the log-prob with the gradient of the squashing function
        return log_prob_norm - log_abs_det_grad_squash

    def kl_divergence(
        self, other_dist: "SquashedMultivariateNormalDiagBase", **kwargs
    ) -> distribution.Array:
        if not isinstance(other_dist, type(self)):
            raise (TypeError(other_dist), type(self))

        # The squashing function does not influence the KL divergence
        return self.normal.kl_divergence(other_dist=other_dist.normal)


class GaussianSquashedMultivariateNormalDiag(SquashedMultivariateNormalDiagBase):
    Epsilon: float = 1e-5
    ScaleOfSquashingGaussian = 0.5 * 1.8137

    def __init__(
        self,
        low: distribution.Array,
        high: distribution.Array,
        loc: Optional[distribution.Array] = None,
        scale_diag: Optional[distribution.Array] = None,
    ):
        # Clip the mean of the underlying gaussian in a valid range
        loc = jnp.clip(a=loc, a_min=-4, a_max=4)

        # Initialize base class
        super().__init__(low=low, high=high, loc=loc, scale_diag=scale_diag)

        # Create the gaussian from which we take the CDF as squashing function.
        # Note: the default scale approximates the CDF to tanh.
        self.squash_dist = distrax.MultivariateNormalDiag(
            loc=jnp.array([0.0]), scale_diag=jnp.array([self.ScaleOfSquashingGaussian])
        )

    def _squash_dist_cdfi(self, value: distribution.Array) -> distribution.Array:
        def unstandardize(
            dist: MultivariateNormalDiag, value_std: distribution.Array
        ) -> distribution.Array:
            return value_std * dist.scale_diag + dist.loc

        return unstandardize(
            dist=self.squash_dist, value_std=jax.scipy.special.ndtri(p=value)
        )

    def _squash(self, unsquashed_sample: distribution.Array) -> distribution.Array:
        # Squash the input sample into the [0, 1] interval
        squashed_sample = jnp.vectorize(self.squash_dist.cdf)(unsquashed_sample)

        # Adjust boundaries in order to prevent getting infinite log-prob
        clipped_squashed_sample = jnp.clip(
            squashed_sample, a_min=self.Epsilon, a_max=(1 - self.Epsilon)
        )

        # Project the squashed sample into the output space [low, high]
        return clipped_squashed_sample * (self.high - self.low) + self.low

    def _unsquash(self, squashed_sample: distribution.Array) -> distribution.Array:
        import jaxsim

        # if (
        #     not jaxsim.utils.tracing(squashed_sample)
        #     and jnp.where(squashed_sample > self.high, True, False).any()
        # ):
        #     raise ValueError(squashed_sample, self.high)
        #
        # if (
        #     not jaxsim.utils.tracing(squashed_sample)
        #     and jnp.where(squashed_sample < self.low, True, False).any()
        # ):
        #     raise ValueError(squashed_sample, self.low)
        # Project the squashed sample into the normalized space [0, 1]
        normalized_squashed_example = (squashed_sample - self.low) / (
            self.high - self.low
        )

        # Unsquash the sample
        return self._squash_dist_cdfi(value=normalized_squashed_example)

    def _log_abs_det_grad_squash(
        self, unsquashed_sample: distribution.Array
    ) -> distribution.Array:
        # Compute the log-grad of the squashing function
        log_grad = jnp.vectorize(self.squash_dist.log_prob)(unsquashed_sample)
        log_grad += jnp.log(self.high - self.low)  # TODO: add this again

        # Adjust size
        log_grad = log_grad if log_grad.ndim > 1 else jnp.array([log_grad])

        # Sum over sample dimension (i.e. return a single value for each sample)
        return log_grad.sum(axis=1)

    def entropy(self) -> distribution.Array:
        return (
            -self.normal.kl_divergence(other_dist=self.squash_dist)
            + jnp.log(self.high - self.low).sum()
        )


class TanhSquashedMultivariateNormalDiag(SquashedMultivariateNormalDiagBase):
    Epsilon: float = 1e-6

    def __init__(
        self,
        low: distribution.Array,
        high: distribution.Array,
        loc: Optional[distribution.Array] = None,
        scale_diag: Optional[distribution.Array] = None,
    ):
        # Clip the mean of the underlying gaussian in a valid range
        loc = jnp.clip(a=loc, a_min=-4, a_max=4)

        # Initialize base class
        super().__init__(low=low, high=high, loc=loc, scale_diag=scale_diag)

    def _squash(self, unsquashed_sample: distribution.Array) -> distribution.Array:
        # Squash the input sample into the [0, 1] interval
        squashed_sample = (jnp.tanh(unsquashed_sample) + 1) / 2

        # Project the squashed sample into the output space [low, high]
        return squashed_sample * (self.high - self.low) + self.low

    def _unsquash(self, squashed_sample: distribution.Array) -> distribution.Array:
        import jaxsim

        # if (
        #     not jaxsim.utils.tracing(squashed_sample)
        #     and jnp.where(squashed_sample > self.high, True, False).any()
        # ):
        #     raise ValueError(squashed_sample, self.high)
        # if (
        #     not jaxsim.utils.tracing(squashed_sample)
        #     and jnp.where(squashed_sample < self.low, True, False).any()
        # ):
        #     raise ValueError(squashed_sample, self.low)
        # Project the squashed sample into the normalized space [0, 1]
        normalized_squashed_sample = (squashed_sample - self.low) / (
            self.high - self.low
        )

        # Project into [-1, 1]
        normalized_squashed_sample = normalized_squashed_sample * 2 - 1.0

        # Clip so that tanh output is stabilized
        clipped_squashed_sample = jnp.clip(
            normalized_squashed_sample, -1.0 + self.Epsilon, 1.0 - self.Epsilon
        )

        # Unsquash the sample
        return jnp.arctanh(clipped_squashed_sample)

    def _log_abs_det_grad_squash(
        self, unsquashed_sample: distribution.Array
    ) -> distribution.Array:
        # Compute the log-grad of the squashing function
        log_grad = jnp.log(1 - jnp.tanh(unsquashed_sample) ** 2 + self.Epsilon)
        log_grad += jnp.log(self.high - self.low) - jnp.log(2)

        # Adjust size
        log_grad = log_grad if log_grad.ndim > 1 else jnp.array([log_grad])

        # Sum over sample dimension (i.e. return a single value for each sample)
        return log_grad.sum(axis=1)

    def entropy(self) -> distribution.Array:
        return jnp.zeros_like(self.low).sum(axis=1)
