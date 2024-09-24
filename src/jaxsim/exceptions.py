import jax


def handle_if(
    condition: bool | jax.Array, exception: type, msg: str, *args, **kwargs
) -> None:
    """
    Handle an exception based on a condition, either by raising it or calling it
    (if the exception type does not inherit from BaseException). Useful in jit-compiled functions.

    Args:
        condition:
            The boolean condition of the evaluated expression that triggers
            the exception during runtime.
        exception: The type of exception to raise or call.
        msg:
            The message to display when the exception is raised. The message can be a
            format string (fmt), whose fields are filled with the args and kwargs.
    """
    # Check early that the format string is well-formed.
    try:
        _ = msg.format(*args, **kwargs)
    except Exception as e:
        msg = "Error in formatting exception message with args={} and kwargs={}"
        raise ValueError(msg.format(args, kwargs)) from e

    def _handle_message(condition: bool, *args, **kwargs) -> None:
        """The function called by the JAX callback."""
        if condition:
            formatted_msg = msg.format(*args, **kwargs)
            # If the exception is a subclass of BaseException, we raise it.
            if issubclass(exception, BaseException):
                raise exception(formatted_msg)
            else:
                # Otherwise, we call it.
                raise exception(formatted_msg)

    def _callback(args, kwargs) -> None:
        """The function that calls the JAX callback, executed only when needed."""
        jax.debug.callback(_handle_message, condition, *args, **kwargs)

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

    return handle_if(condition, RuntimeError, msg, *args, **kwargs)


def raise_value_error_if(
    condition: bool | jax.Array, msg: str, *args, **kwargs
) -> None:

    return handle_if(condition, ValueError, msg, *args, **kwargs)
