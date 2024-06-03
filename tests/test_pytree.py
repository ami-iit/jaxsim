import io
import pathlib
from contextlib import redirect_stdout

import jax

import jaxsim.api as js


def test_call_jit_compiled_function_passing_different_objects(
    ergocub_model_description_path: pathlib.Path,
):

    # Create a first model from the URDF.
    model1 = js.model.JaxSimModel.build_from_model_description(
        model_description=ergocub_model_description_path,
        is_urdf=True,
    )

    # Create a second model from the URDF.
    model2 = js.model.JaxSimModel.build_from_model_description(
        model_description=ergocub_model_description_path,
        is_urdf=True,
    )

    # The objects should be different, but the comparison should return True.
    assert id(model1) != id(model2)
    assert model1 == model2
    assert hash(model1) == hash(model2)

    # If this function has never been compiled by any other test, JAX will
    # jit-compile it here.
    _ = js.contact.estimate_good_soft_contacts_parameters(model=model1)

    # Now JAX should not compile it again.
    with jax.log_compiles():
        with io.StringIO() as buf, redirect_stdout(buf):
            # Beyond running without any JIT recompilations, the following function
            # should work on different JaxSimModel objects without raising any errors
            # related to the comparison of Static fields.
            _ = js.contact.estimate_good_soft_contacts_parameters(model=model2)
            stdout = buf.getvalue()

    assert (
        f"Compiling {js.contact.estimate_good_soft_contacts_parameters.__name__}"
        not in stdout
    )
