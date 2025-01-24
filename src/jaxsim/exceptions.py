import os

import jax


def raise_if(
    condition: bool | jax.Array, exception: type, msg: str, *args, **kwargs
) -> None:
    """
    Raise a host-side exception if a condition is met. Useful in jit-compiled functions.

    Args:
        condition:
            The boolean condition of the evaluated expression that triggers
            the exception during runtime.
        exception: The type of exception to raise.
        msg:
            The message to display when the exception is raised. The message can be a
            format string (fmt), whose fields are filled with the args and kwargs.
        *args: The arguments to fill the format string.
        **kwargs: The keyword arguments to fill the format string
    """

    # Disable host callback if running on unsupported hardware or if the user
    # explicitly disabled it.
    if jax.devices()[0].platform in {"tpu", "METAL"} or not os.environ.get(
        "JAXSIM_ENABLE_EXCEPTIONS", 0
    ):
        return

    # Check early that the format string is well-formed.
    try:
        _ = msg.format(*args, **kwargs)
    except Exception as e:
        msg = "Error in formatting exception message with args={} and kwargs={}"
        raise ValueError(msg.format(args, kwargs)) from e

    def _raise_exception(condition: bool, *args, **kwargs) -> None:
        """The function called by the JAX callback."""

        if condition:
            raise exception(msg.format(*args, **kwargs))

    def _callback(args, kwargs) -> None:
        """The function that calls the JAX callback, executed only when needed."""

        jax.debug.callback(_raise_exception, condition, *args, **kwargs)

    # Since running a callable on the host is expensive, we prevent its execution
    # if the condition is False with a low-level conditional expression.
    def _run_callback_only_if_condition_is_true(*args, **kwargs) -> None:
        return jax.lax.cond(
            condition,
            _callback,
            lambda args, kwargs: None,
            args,
            kwargs,
        )

    return _run_callback_only_if_condition_is_true(*args, **kwargs)


def raise_runtime_error_if(
    condition: bool | jax.Array, msg: str, *args, **kwargs
) -> None:
    """
    Raise a RuntimeError if a condition is met. Useful in jit-compiled functions.
    """

    return raise_if(condition, RuntimeError, msg, *args, **kwargs)


def raise_value_error_if(
    condition: bool | jax.Array, msg: str, *args, **kwargs
) -> None:
    """
    Raise a ValueError if a condition is met. Useful in jit-compiled functions.
    """

    return raise_if(condition, ValueError, msg, *args, **kwargs)
