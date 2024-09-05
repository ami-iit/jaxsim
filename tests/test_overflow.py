import pytest

import jax
import jax.numpy as jnp
import jaxlib

import jaxsim.api as js
from jaxsim import exceptions


def test_time_overflow(jaxsim_model_box_32bit: js.model.JaxSimModel):

    model = jaxsim_model_box_32bit

    # Build the data of the model.
    data = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.5]),
    )

    # Initialize the simulation time.
    data = data.replace(time_ns=jnp.array(2**32 - 1, dtype=jnp.uint32))
    dt = 1e-9
    t0_ns = data.time_ns

    @jax.jit
    def _advance_time(
        data: js.data.JaxSimModelData, dt: float
    ) -> js.data.JaxSimModelData:

        exceptions.raise_if(
            condition=(tf := (t0_ns + jnp.array(dt * 1e9).astype(t0_ns.dtype))) < t0_ns,
            exception=OverflowError,
            msg="The simulation time overflowed the maximum integer value. Consider using x64 by setting `JAX_ENABLE_X64=1`.",
        )

        data = data.replace(time_ns=tf)

    with pytest.raises(
        jaxlib.xla_extension.XlaRuntimeError,
    ):

        data = _advance_time(data, dt)
