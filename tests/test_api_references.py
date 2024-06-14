import jax
import jax.numpy as jnp
import pytest
from jaxlib.xla_extension import XlaRuntimeError

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import VelRepr


def get_random_references(
    model: js.model.JaxSimModel | None = None,
    data: js.data.JaxSimModelData | None = None,
    *,
    velocity_representation: jtp.VelRepr,
    key: jax.Array,
) -> tuple[js.data.JaxSimModelData, js.references.JaxSimModelReferences]:

    _, subkey = jax.random.split(key, num=2)

    _, subkey1, subkey2 = jax.random.split(subkey, num=3)

    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base.
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    return references


def test_raise_errors_link_forces(
    jaxsim_model_box: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_model_box

    _, subkey1, subkey2 = jax.random.split(prng_key, num=3)

    data = js.data.random_model_data(model=model, key=subkey1)

    # ================
    # VelRepr.Inertial
    # ================

    references_inertial = get_random_references(
        model=model, data=None, velocity_representation=VelRepr.Inertial, key=subkey2
    )

    # `model` is None and `link_names` is not None.
    with pytest.raises(
        ValueError, match="Link names cannot be provided without a model"
    ):
        references_inertial.link_forces(model=None, link_names=model.link_names())

    # ============
    # VelRepr.Body
    # ============

    references_body = get_random_references(
        model=model, data=data, velocity_representation=VelRepr.Body, key=subkey2
    )

    # `model` is None and `link_names` is not None.
    with pytest.raises(
        ValueError, match="Link names cannot be provided without a model"
    ):
        references_body.link_forces(model=None, link_names=model.link_names())

    # `model` is not None and `data` is None.
    with pytest.raises(
        XlaRuntimeError,
        match="Missing model data to use a representation different from `VelRepr.Inertial`",
    ):
        references_body.link_forces(model=model, data=None)


def test_raise_errors_apply_link_forces(
    jaxsim_model_box: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_model_box

    _, subkey1, subkey2 = jax.random.split(prng_key, num=3)

    data = js.data.random_model_data(model=model, key=subkey1)

    # ================
    # VelRepr.Inertial
    # ================

    references_inertial = get_random_references(
        model=model, data=None, velocity_representation=VelRepr.Inertial, key=subkey2
    )

    # `model` is None
    with pytest.raises(
        ValueError,
        match="Link names cannot be provided without a model",
    ):
        references_inertial.apply_link_forces(
            forces=jnp.zeros(6), model=None, data=None, link_names=model.link_names()
        )

    # ============
    # VelRepr.Body
    # ============

    references_body = get_random_references(
        model=model, data=data, velocity_representation=VelRepr.Body, key=subkey2
    )

    # `model` is None
    with pytest.raises(
        ValueError,
        match="Link names cannot be provided without a model",
    ):
        references_body.apply_link_forces(
            forces=jnp.zeros(6), model=None, data=None, link_names=model.link_names()
        )

    # `model` is None
    with pytest.raises(
        XlaRuntimeError,
        match="Missing model to use a representation different from `VelRepr.Inertial`",
    ):
        references_body.apply_link_forces(forces=jnp.zeros(6), model=None, data=None)

    # `model` is not None and `data` is None.
    with pytest.raises(
        XlaRuntimeError,
        match="Missing model data to use a representation different from `VelRepr.Inertial`",
    ):
        references_body.apply_link_forces(
            forces=jnp.zeros(6), model=model, data=None, link_names=model.link_names()
        )
