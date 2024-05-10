import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsim.api as js
from jaxsim import VelRepr, logging

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
    jaxsim_model_single_pendulum: js.model.JaxSimModel,
    prng_key: jax.Array,
):
    # TODO: add more models to the test
    model = jaxsim_model_single_pendulum

    data = js.data.JaxSimModelData.zero(model=model)

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    if len(model.description.get().frames) == 0:
        logging.debug("No frames detected in model. Skipping test")
        return

    # Get indexes of frames that are not links
    frame_indexes = [
        frame.index
        for frame in model.description.get().frames
        if frame.index is not None and frame.name in kin_dyn.frame_names()
    ]
    frame_names = [
        frame.name
        for frame in model.description.get().frames
        if frame.name in kin_dyn.frame_names()
    ]

    # =====
    # Tests
    # =====

    results = []
    for frame_name, frame_idx in zip(frame_names, frame_indexes):
        W_H_F_js = js.frame.transform(model=model, data=data, frame_index=frame_idx)
        W_H_F_iDynTree = kin_dyn.frame_transform(frame_name=frame_name)
        result = W_H_F_js == pytest.approx(W_H_F_iDynTree)
        parent_link_name_iDynTree = kin_dyn.frame_parent_link_name(
            frame_name=frame_name
        )
        logging.debug(
            f'In iDynTree the frame "{frame_name}" is connected to link {parent_link_name_iDynTree}'
        )
        logging.debug(
            f'In Jaxsim the frame "{frame_name}" is connected to link {model.description.get().frames[frame_idx - model.number_of_links()].parent.name}'
        )

        if not result:
            logging.error(f"Assertion failed for frame: {frame_name}")
            logging.debug("W_H_F_js:")
            logging.debug(W_H_F_js)
            logging.debug("W_H_F_iDynTree:")
            logging.debug(W_H_F_iDynTree)
        results.append(result)

    assert all(results), "At least one assertion failed"


def test_frame_jacobians(
    jaxsim_model_single_pendulum: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):
    # TODO: add more models to the test
    model = jaxsim_model_single_pendulum

    data = js.data.JaxSimModelData.zero(model=model)

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    if len(model.description.get().frames) == 0:
        logging.debug("No frames detected in model. Skipping test")
        return

    frame_indexes = [
        frame.index
        for frame in model.description.get().frames
        if frame.index is not None and frame.name in kin_dyn.frame_names()
    ]
    frame_names = [
        frame.name
        for frame in model.description.get().frames
        if frame.name in kin_dyn.frame_names()
    ]

    logging.debug(f"Frames considered: {frame_names}")

    # =====
    # Tests
    # =====

    assert len(frame_indexes) == len(frame_names)

    results = []
    for frame_name, frame_idx in zip(frame_names, frame_indexes):
        J_WL_js = js.frame.jacobian(model=model, data=data, frame_index=frame_idx)
        J_WL_iDynTree = kin_dyn.jacobian_frame(frame_name=frame_name)
        result = J_WL_js.shape == J_WL_iDynTree.shape, frame_name
        if J_WL_js != pytest.approx(J_WL_iDynTree, abs=1e-9):
            logging.error("Jacobians from Jaxsim and iDynTree do not match:")
            logging.debug("J_WL_js")
            logging.debug(J_WL_js)
            logging.debug("J_WL_iDynTree")
            logging.debug(J_WL_iDynTree)
        results.append(result)

    assert all(results), "At least one assertion failed"
