import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


def test_frame_index(jaxsim_models_types: js.model.JaxSimModel):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    if len(model.description.get().frames) == 0:
        return

    frame_indices = jnp.array(
        [
            frame.index
            for frame in model.description.get().frames
            if frame.index is not None
        ]
    )
    frame_names = np.array([frame.name for frame in model.description.get().frames])

    for frame_idx, frame_name in zip(frame_indices, frame_names):
        assert js.frame.name_to_idx(model=model, frame_name=frame_name) == frame_idx

    assert js.frame.names_to_idxs(
        model=model, frame_names=tuple(frame_names)
    ) == pytest.approx(frame_indices)

    assert js.frame.idxs_to_names(
        model=model, frame_indices=frame_indices
    ) == pytest.approx(frame_names)


def test_frame_transforms(
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

    if len(model.description.get().frames) == 0:
        return

    # Get indexes of frames that are not links
    frame_indexes = [frame.index for frame in model.description.get().frames]
    frame_names = [frame.name for frame in model.description.get().frames]

    # =====
    # Tests
    # =====

    W_H_F_frames = [
        js.frame.transform(model=model, data=data, frame_index=idx)
        for idx in frame_indexes
    ]

    for W_H_F, frame_name in zip(W_H_F_frames, frame_names):
        assert W_H_F == pytest.approx(
            kin_dyn.frame_transform(frame_name=frame_name)
        ), frame_name


def test_frame_jacobians(
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

    if len(model.description.get().frames) == 0:
        print("No frames detected in model. Skipping test")
        return

    frame_indexes = [
        frame.index
        for frame in model.description.get().frames
        if frame.index is not None
        # and frame.name == "l_forearm"
        and frame.name in kin_dyn.frame_names()
    ]
    frame_names = [
        frame.name
        for frame in model.description.get().frames
        # if frame.name == "l_forearm"
        if frame.name in kin_dyn.frame_names()
    ]

    print(f"considered frames: {frame_names}")

    # =====
    # Tests
    # =====

    assert len(frame_indexes) == len(frame_names)

    for frame_name, frame_idx in zip(frame_names, frame_indexes):
        print(f"Checking frame {frame_name}...")
        J_WL_js = js.frame.jacobian(model=model, data=data, frame_index=frame_idx)
        J_WL_iDynTree = kin_dyn.jacobian_frame(frame_name=frame_name)
        assert J_WL_js.shape == J_WL_iDynTree.shape, frame_name
        if J_WL_js != pytest.approx(J_WL_iDynTree, abs=1e-9):
            print("Jacobians from Jaxsim and iDynTree do not match:")
            print("J_WL_js")
            print(J_WL_js)
            print("J_WL_iDynTree")
            print(J_WL_iDynTree)
        else:
            print("Success")
