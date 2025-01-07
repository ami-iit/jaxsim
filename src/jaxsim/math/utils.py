import jax.numpy as jnp

import jaxsim.typing as jtp


def safe_norm(array: jtp.ArrayLike, axis=None) -> jtp.Array:
    """
    Provides a calculation for an array norm so that it is safe
    to compute the gradient and handle NaNs.

    Args:
        array: The array for which to compute the norm.
        axis: The axis for which to compute the norm.

    Returns:
        The norm of the array with handling for zero arrays to avoid NaNs.
    """

    # Check if the entire array is composed of zeros.
    is_zero = jnp.allclose(array, 0.0)

    # Replace zeros with an array of ones temporarily to avoid division by zero.
    # This ensures the computation of norm does not produce NaNs or Infs.
    array = jnp.where(is_zero, jnp.ones_like(array), array)

    # Compute the norm of the array along the specified axis.
    norm = jnp.linalg.norm(array, axis=axis)

    # Use `jnp.where` to set the norm to 0.0 where the input array was all zeros.
    # This usage supports potential batch processing for future scalability.
    return jnp.where(is_zero, 0.0, norm)
