import jax
import pytest

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


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
