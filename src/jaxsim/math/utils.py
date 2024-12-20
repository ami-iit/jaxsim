import jax.numpy as jnp

import jaxsim.typing as jtp


def safe_norm(array: jtp.ArrayLike, axis=None) -> jtp.Array:
    """
    Provides a calculation for an array norm so that it is safe
    to compute the gradient and the NaNs are handled.

    Args:
        array: The array for which to compute the norm.
        axis: The axis for which to compute the norm.
    """

    is_zero = jnp.allclose(array, 0.0)

    array = jnp.where(is_zero, jnp.ones_like(array), array)

    norm = jnp.linalg.norm(array, axis=axis)

    return jnp.where(is_zero, 0.0, norm)
