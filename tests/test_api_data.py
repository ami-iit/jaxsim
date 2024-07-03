import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js
from jaxsim import VelRepr
from jaxsim.utils import Mutability

from . import utils_idyntree


def test_data_valid(
    jaxsim_models_all: js.model.JaxSimModel,
):

    model = jaxsim_models_all
    data = js.data.JaxSimModelData.build(model=model)

    assert data.valid(model=model)


def test_data_joint_indexing(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    assert data.joint_positions(
        model=model, joint_names=model.joint_names()
    ) == pytest.approx(data.joint_positions())

    assert data.joint_positions() == pytest.approx(
        data.state.physics_model.joint_positions
    )

    assert data.joint_velocities(
        model=model, joint_names=model.joint_names()
    ) == pytest.approx(data.joint_velocities())

    assert data.joint_velocities() == pytest.approx(
        data.state.physics_model.joint_velocities
    )


def test_data_switch_velocity_representation(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    # =====
    # Tests
    # =====

    new_base_linear_velocity = jnp.array([1.0, -2.0, 3.0])
    old_base_linear_velocity = data.state.physics_model.base_linear_velocity

    # The following should not change the original `data` object since it raises.
    with pytest.raises(RuntimeError):
        with data.switch_velocity_representation(
            velocity_representation=VelRepr.Inertial
        ):
            with data.mutable_context(mutability=Mutability.MUTABLE):
                data.state.physics_model.base_linear_velocity = new_base_linear_velocity
            raise RuntimeError("This is raised on purpose inside this context")

    assert data.state.physics_model.base_linear_velocity == pytest.approx(
        old_base_linear_velocity
    )

    # The following instead should result to an updated `data` object.
    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        with data.mutable_context(mutability=Mutability.MUTABLE):
            data.state.physics_model.base_linear_velocity = new_base_linear_velocity

    assert data.state.physics_model.base_linear_velocity == pytest.approx(
        new_base_linear_velocity
    )


def test_data_change_velocity_representation(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    # =====
    # Tests
    # =====

    kin_dyn_inertial = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    with data.switch_velocity_representation(VelRepr.Mixed):
        kin_dyn_mixed = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
            model=model, data=data
        )

    with data.switch_velocity_representation(VelRepr.Body):
        kin_dyn_body = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
            model=model, data=data
        )

    assert data.base_velocity() == pytest.approx(kin_dyn_inertial.base_velocity())

    if not model.floating_base():
        return

    with data.switch_velocity_representation(VelRepr.Mixed):
        assert data.base_velocity() == pytest.approx(kin_dyn_mixed.base_velocity())
        assert data.base_velocity()[0:3] != pytest.approx(
            data.state.physics_model.base_linear_velocity
        )
        assert data.base_velocity()[3:6] == pytest.approx(
            data.state.physics_model.base_angular_velocity
        )

    with data.switch_velocity_representation(VelRepr.Body):
        assert data.base_velocity() == pytest.approx(kin_dyn_body.base_velocity())
        assert data.base_velocity()[0:3] != pytest.approx(
            data.state.physics_model.base_linear_velocity
        )
        assert data.base_velocity()[3:6] != pytest.approx(
            data.state.physics_model.base_angular_velocity
        )
