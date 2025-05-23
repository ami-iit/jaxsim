import io
from contextlib import redirect_stdout

import jax
import jax.numpy as jnp
import pytest
from jax.errors import JaxRuntimeError

from jaxsim import exceptions


def test_exceptions_in_jit_functions():

    msg_during_jit = "Compiling jit_compiled_function"

    @jax.jit
    def jit_compiled_function(data: jax.Array) -> jax.Array:

        # This message is compiled only during JIT compilation.
        print(msg_during_jit)

        # Condition that will trigger the exception.
        failed_if_42_plus = jnp.allclose(data, 42)

        # Raise a ValueError if the condition is met.
        # The fmt string is built from kwargs.
        exceptions.raise_value_error_if(
            condition=failed_if_42_plus,
            msg="Raising ValueError since data={num}",
            num=data,
        )

        # Condition that will trigger the exception.
        failed_if_42_minus = jnp.allclose(data, -42)

        # Raise a RuntimeError if the condition is met.
        # The fmt string is built from args.
        exceptions.raise_runtime_error_if(
            failed_if_42_minus,
            "Raising RuntimeError since data={}",
            data,
        )

        return data

    # In the first call, the function will be compiled and print the message.
    with jax.log_compiles(), io.StringIO() as buf, redirect_stdout(buf):

        data = 40
        out = jit_compiled_function(data=data)
        stdout = buf.getvalue()
        assert out == data

    assert msg_during_jit in stdout
    assert jit_compiled_function._cache_size() == 1

    # In the second call, the function won't be compiled and won't print the message.
    with jax.log_compiles(), io.StringIO() as buf, redirect_stdout(buf):

        data = 41
        out = jit_compiled_function(data=data)
        stdout = buf.getvalue()
        assert out == data

    assert msg_during_jit not in stdout
    assert jit_compiled_function._cache_size() == 1

    # Let's trigger a ValueError exception by passing 42.
    data = 42
    with pytest.raises(
        JaxRuntimeError,
        match=f"ValueError: Raising ValueError since data={data}",
    ):
        _ = jit_compiled_function(data=data)

    assert jit_compiled_function._cache_size() == 1

    # Let's trigger a RuntimeError exception by passing -42.
    data = -42
    with pytest.raises(
        JaxRuntimeError,
        match=f"RuntimeError: Raising RuntimeError since data={data}",
    ):
        _ = jit_compiled_function(data=data)

    assert jit_compiled_function._cache_size() == 1
