from typing import Sequence

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp

from . import model as Model

# =======================
# Index-related functions
# =======================


def name_to_idx(model: Model.JaxSimModel, *, joint_name: str) -> jtp.Int:
    """
    Convert the name of a joint to its index.

    Args:
        model: The model to consider.
        joint_name: The name of the joint.

    Returns:
        The index of the joint.
    """

    return jnp.array(
        model.physics_model.description.joints_dict[joint_name].index, dtype=int
    )


def idx_to_name(model: Model.JaxSimModel, *, joint_index: jtp.IntLike) -> str:
    """
    Convert the index of a joint to its name.

    Args:
        model: The model to consider.
        joint_index: The index of the joint.

    Returns:
        The name of the joint.
    """

    d = {j.index: j.name for j in model.physics_model.description.joints_dict.values()}
    return d[joint_index]


def names_to_idxs(model: Model.JaxSimModel, *, joint_names: Sequence[str]) -> jax.Array:
    """
    Convert a sequence of joint names to their corresponding indices.

    Args:
        model: The model to consider.
        joint_names: The names of the joints.

    Returns:
        The indices of the joints.
    """

    return jnp.array(
        [
            # Note: the index of the joint for RBDAs starts from 1, but
            # the index for accessing the right element starts from 0.
            # Therefore, there is a -1.
            model.physics_model.description.joints_dict[name].index - 1
            for name in joint_names
        ],
        dtype=int,
    )


def idxs_to_names(
    model: Model.JaxSimModel, *, joint_indices: Sequence[jtp.IntLike] | jtp.VectorLike
) -> tuple[str, ...]:
    """
    Convert a sequence of joint indices to their corresponding names.

    Args:
        model: The model to consider.
        joint_indices: The indices of the joints.

    Returns:
        The names of the joints.
    """

    d = {
        j.index - 1: j.name
        for j in model.physics_model.description.joints_dict.values()
    }

    return tuple(d[i] for i in joint_indices)
