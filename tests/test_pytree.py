import io
import pathlib
from contextlib import redirect_stdout

import chex
import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js


def test_call_jit_compiled_function_passing_different_objects(
    ergocub_model_description_path: pathlib.Path, jaxsim_model_box
):

    # Create a first model from the URDF.
    ergocub_model1 = js.model.JaxSimModel.build_from_model_description(
        model_description=ergocub_model_description_path
    )

    # Create a second model from the URDF.
    ergocub_model2 = js.model.JaxSimModel.build_from_model_description(
        model_description=ergocub_model_description_path
    )

    box_model = jaxsim_model_box

    # The objects should be different, but the comparison should return True.
    assert id(ergocub_model1) != id(ergocub_model2)
    assert ergocub_model1 == ergocub_model2
    assert hash(ergocub_model1) == hash(ergocub_model2)

    # If this function has never been compiled by any other test, JAX will
    # jit-compile it here.
    _ = js.contact.estimate_good_contact_parameters(model=ergocub_model1)

    # Now JAX should not compile it again.
    with jax.log_compiles(), io.StringIO() as buf, redirect_stdout(buf):
        # Beyond running without any JIT recompilations, the following function
        # should work on different JaxSimModel objects without raising any errors
        # related to the comparison of Static fields.
        _ = js.contact.estimate_good_contact_parameters(model=ergocub_model2)
        stdout = buf.getvalue()

    assert (
        f"Compiling {js.contact.estimate_good_contact_parameters.__name__}"
        not in stdout
    )

    # Define a new JIT-compiled function and check that is not recompiled for
    # different model objects having the same pytree structure.
    @jax.jit
    @chex.assert_max_traces(n=1)
    def my_jit_function(model: js.model.JaxSimModel, data: js.data.JaxSimModelData):
        # Return random elements from model and data, just to have something returned.
        return (
            jnp.sum(model.kin_dyn_parameters.link_parameters.mass),
            data.base_position,
        )

    data1 = js.data.JaxSimModelData.build(model=ergocub_model1)

    _ = my_jit_function(model=ergocub_model1, data=data1)

    # This should not retrace the function, as ergocub_model2 has the same
    # pytree structure as ergocub_model1.
    _ = my_jit_function(model=ergocub_model2, data=data1)

    # Calling the function with a different model object will retrace it, as
    # expected. Therefore, an AssertionError should be raised.
    with pytest.raises(
        AssertionError, match="Function 'my_jit_function' is traced > 1 times!"
    ):
        data3 = js.data.JaxSimModelData.build(model=box_model)
        _ = my_jit_function(model=box_model, data=data3)
