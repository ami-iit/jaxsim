from typing import Any

import jax._src.core
import jax.flatten_util
import jax.interpreters.partial_eval


def tracing(var: Any) -> bool | jax.Array:
    """Return True if the variable is being traced by JAX, False otherwise."""

    return isinstance(
        var, jax._src.core.Tracer | jax.interpreters.partial_eval.DynamicJaxprTracer
    )


def not_tracing(var: Any) -> bool | jax.Array:
    """Return True if the variable is not being traced by JAX, False otherwise."""

    return True if tracing(var) is False else False
