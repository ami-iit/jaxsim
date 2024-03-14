import io
from contextlib import redirect_stdout

import jax
import rod.builder.primitives
import rod.urdf.exporter

import jaxsim.api as js


# https://github.com/ami-iit/jaxsim/issues/103
def test_call_jit_compiled_function_passing_different_objects():

    # Create on-the-fly a ROD model of a box.
    rod_model = (
        rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=1.0, name="box")
        .build_model()
        .add_link()
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    # Export the URDF string.
    urdf_string = rod.urdf.exporter.UrdfExporter.sdf_to_urdf_string(
        sdf=rod_model, pretty=True
    )

    model1 = js.model.JaxSimModel.build_from_model_description(
        model_description=urdf_string,
        is_urdf=True,
    )

    model2 = js.model.JaxSimModel.build_from_model_description(
        model_description=urdf_string,
        is_urdf=True,
    )

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
