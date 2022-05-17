import jax.abstract_arrays
import jax.interpreters.partial_eval


def tracing(var) -> bool:

    return isinstance(
        var,
        (
            jax.abstract_arrays.ShapedArray,
            jax.interpreters.partial_eval.DynamicJaxprTracer,
        ),
    )
