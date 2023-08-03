from typing import Any

import jax._src.core
import jax.abstract_arrays
import jax.flatten_util
import jax.interpreters.partial_eval


def tracing(var: Any) -> bool | jax.Array:
    """Returns True if the variable is being traced by JAX, False otherwise."""

    return jax.numpy.array(
        [
            isinstance(var, t)
            for t in (
                jax._src.core.Tracer,
                jax.abstract_arrays.ShapedArray,
                jax.interpreters.partial_eval.DynamicJaxprTracer,
            )
        ]
    ).any()


def not_tracing(var: Any) -> bool:
    """Returns True if the variable is not being traced by JAX, False otherwise."""

    return True if tracing(var) is False else False
