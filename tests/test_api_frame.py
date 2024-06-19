import jax
import jax.numpy as jnp
import jaxlib.xla_extension
import pytest

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


def test_frame_index(jaxsim_models_types: js.model.JaxSimModel):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    for idx, frame_name in enumerate(model.frame_names()):
        frame_index = n_l + idx
        assert js.frame.name_to_idx(model=model, frame_name=frame_name) == frame_index
        assert js.frame.idx_to_name(model=model, frame_index=frame_index) == frame_name
        assert (
            js.frame.idx_of_parent_link(model=model, frame_index=frame_index)
            < model.number_of_links()
        )

    assert js.frame.names_to_idxs(
        model=model, frame_names=model.frame_names()
    ) == pytest.approx(jnp.arange(n_l, n_l + n_f))

    assert (
        js.frame.idxs_to_names(
            model=model,
            frame_indices=tuple(
                js.frame.names_to_idxs(
                    model=model, frame_names=model.frame_names()
                ).tolist()
            ),
        )
        == model.frame_names()
    )

    with pytest.raises(ValueError):
        _ = js.frame.name_to_idx(model=model, frame_name="non_existent_frame")

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=-1)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=n_l - 1)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=n_l + n_f)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=-1)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=n_l - 1)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=n_l + n_f)


def test_frame_transforms(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # Get all names of frames in the iDynTree model.
    frame_names = [
        frame.name
        for frame in model.description.frames
        if frame.name in kin_dyn.frame_names()
    ]

    # Skip some entry of models with many frames.
    frame_names = [
        name
        for name in frame_names
        if "skin" not in name or "laser" not in name or "depth" not in name
    ]

    # Get indices of frames.
    frame_indices = tuple(
        frame.index
        for frame in model.description.frames
        if frame.index is not None and frame.name in frame_names
    )

    # =====
    # Tests
    # =====

    assert len(frame_indices) == len(frame_names)

    for frame_name in frame_names:

        W_H_F_js = js.frame.transform(
            model=model,
            data=data,
            frame_index=js.frame.name_to_idx(model=model, frame_name=frame_name),
        )
        W_H_F_idt = kin_dyn.frame_transform(frame_name=frame_name)
        assert W_H_F_js == pytest.approx(W_H_F_idt, abs=1e-6), frame_name


def test_frame_jacobians(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # Get all names of frames in the iDynTree model.
    frame_names = [
        frame.name
        for frame in model.description.frames
        if frame.name in kin_dyn.frame_names()
    ]

    # Lower the number of frames for models with many frames.
    if model.name().lower() == "ergocub":
        assert any("sole" in name for name in frame_names)
        frame_names = [name for name in frame_names if "sole" in name]

    # Get indices of frames.
    frame_indices = tuple(
        frame.index
        for frame in model.description.frames
        if frame.index is not None and frame.name in frame_names
    )

    # =====
    # Tests
    # =====

    assert len(frame_indices) == len(frame_names)

    for frame_name, frame_index in zip(frame_names, frame_indices):

        J_WL_js = js.frame.jacobian(model=model, data=data, frame_index=frame_index)
        J_WL_idt = kin_dyn.jacobian_frame(frame_name=frame_name)
        assert J_WL_js == pytest.approx(J_WL_idt, abs=1e-9)
