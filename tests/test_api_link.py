import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


def test_link_index(
    jaxsim_models_types: js.model.JaxSimModel,
):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    for idx, link_name in enumerate(model.link_names()):
        assert js.link.name_to_idx(model=model, link_name=link_name) == idx

    assert js.link.names_to_idxs(
        model=model, link_names=model.link_names()
    ) == pytest.approx(jnp.arange(model.number_of_links()))

    assert (
        js.link.idxs_to_names(
            model=model,
            link_indices=tuple(
                js.link.names_to_idxs(
                    model=model, link_names=model.link_names()
                ).tolist()
            ),
        )
        == model.link_names()
    )


def test_link_inertial_properties(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=VelRepr.Inertial,
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    for link_name, link_idx in zip(
        model.link_names(),
        js.link.names_to_idxs(model=model, link_names=model.link_names()),
    ):
        if link_name == model.base_link():
            continue

        assert js.link.mass(model=model, link_index=link_idx) == pytest.approx(
            kin_dyn.link_mass(link_name=link_name)
        ), link_name

        assert js.link.spatial_inertia(
            model=model, link_index=link_idx
        ) == pytest.approx(kin_dyn.link_spatial_inertia(link_name=link_name)), link_name


def test_link_transforms(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=VelRepr.Inertial,
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    W_H_LL_model = js.model.forward_kinematics(model=model, data=data)

    W_H_LL_links = jax.vmap(
        lambda idx: js.link.transform(model=model, data=data, link_index=idx)
    )(jnp.arange(model.number_of_links()))

    assert W_H_LL_model == pytest.approx(W_H_LL_links)

    for W_H_L, link_name in zip(W_H_LL_links, model.link_names()):

        assert W_H_L == pytest.approx(
            kin_dyn.frame_transform(frame_name=link_name)
        ), link_name


def test_link_jacobians(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    J_WL_links = jax.vmap(
        lambda idx: js.link.jacobian(model=model, data=data, link_index=idx)
    )(jnp.arange(model.number_of_links()))

    for J_WL, link_name in zip(J_WL_links, model.link_names()):
        assert J_WL == pytest.approx(
            kin_dyn.jacobian_frame(frame_name=link_name), abs=1e-9
        ), link_name

    # The following is true only in inertial-fixed representation.
    if data.velocity_representation is VelRepr.Inertial:
        J_WL_model = js.model.generalized_free_floating_jacobian(model=model, data=data)
        assert J_WL_model == pytest.approx(J_WL_links)

    for link_name, link_idx in zip(
        model.link_names(),
        js.link.names_to_idxs(model=model, link_names=model.link_names()),
    ):
        v_WL_idt = kin_dyn.frame_velocity(frame_name=link_name)
        v_WL_js = js.link.velocity(model=model, data=data, link_index=link_idx)
        assert v_WL_js == pytest.approx(v_WL_idt), link_name


def test_link_bias_acceleration(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    for name, index in zip(
        model.link_names(),
        js.link.names_to_idxs(model=model, link_names=model.link_names()),
    ):
        Jν_idt = kin_dyn.frame_bias_acc(frame_name=name)
        Jν_js = js.link.bias_acceleration(model=model, data=data, link_index=index)
        assert pytest.approx(Jν_idt) == Jν_js
