import jax
import pytest

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


def test_com_properties(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    p_com_idt = kin_dyn.com_position()
    p_com_js = js.com.com_position(model=model, data=data)
    assert pytest.approx(p_com_idt) == p_com_js
