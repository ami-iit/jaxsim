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
            The message to display when the exception is raised. It can be a fmt string,
            and users can pass additional arguments to format the string.
    """

    # Check early that the fmt string is well-formed.
    _ = msg.format(*args, **kwargs)

    def _raise_exception(condition: bool, *args, **kwargs) -> None:
        if condition:
            raise exception(msg.format(*args, **kwargs))

    def _callback(args, kwargs) -> None:
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

    return raise_if(condition, RuntimeError, msg, *args, **kwargs)


def raise_value_error_if(
    condition: bool | jax.Array, msg: str, *args, **kwargs
) -> None:

    return raise_if(condition, ValueError, msg, *args, **kwargs)
