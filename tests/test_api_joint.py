import jax.numpy as jnp
import jaxlib.xla_extension
import pytest

import jaxsim.api as js


def test_joint_index(
    jaxsim_models_types: js.model.JaxSimModel,
):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    for idx, joint_name in enumerate(model.joint_names()):
        assert js.joint.name_to_idx(model=model, joint_name=joint_name) == idx
        assert js.joint.idx_to_name(model=model, joint_index=idx) == joint_name

    assert js.joint.names_to_idxs(
        model=model, joint_names=model.joint_names()
    ) == pytest.approx(jnp.arange(model.number_of_joints()))

    assert (
        js.joint.idxs_to_names(
            model=model,
            joint_indices=tuple(
                js.joint.names_to_idxs(
                    model=model, joint_names=model.joint_names()
                ).tolist()
            ),
        )
        == model.joint_names()
    )

    with pytest.raises(ValueError):
        _ = js.joint.name_to_idx(model=model, joint_name="non_existent_joint")

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.joint.idx_to_name(model=model, joint_index=-1)

    with pytest.raises(jaxlib.xla_extension.XlaRuntimeError):
        _ = js.joint.idx_to_name(model=model, joint_index=model.number_of_joints())
