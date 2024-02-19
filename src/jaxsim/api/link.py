from typing import Sequence

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp

from . import data as Data
from . import model as Model

# =======================
# Index-related functions
# =======================


def name_to_idx(model: Model.JaxSimModel, *, link_name: str) -> jtp.Int:
    """
    Convert the name of a link to its index.

    Args:
        model: The model to consider.
        link_name: The name of the link.

    Returns:
        The index of the link.
    """

    return jnp.array(
        model.physics_model.description.links_dict[link_name].index, dtype=int
    )


def idx_to_name(model: Model.JaxSimModel, *, link_index: jtp.IntLike) -> str:
    """
    Convert the index of a link to its name.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The name of the link.
    """

    d = {l.index: l.name for l in model.physics_model.description.links_dict.values()}
    return d[link_index]


def names_to_idxs(model: Model.JaxSimModel, *, link_names: Sequence[str]) -> jax.Array:
    """
    Convert a sequence of link names to their corresponding indices.

    Args:
        model: The model to consider.
        link_names: The names of the links.

    Returns:
        The indices of the links.
    """

    return jnp.array(
        [model.physics_model.description.links_dict[name].index for name in link_names],
        dtype=int,
    )


def idxs_to_names(
    model: Model.JaxSimModel, *, link_indices: Sequence[jtp.IntLike] | jtp.VectorLike
) -> tuple[str, ...]:
    """
    Convert a sequence of link indices to their corresponding names.

    Args:
        model: The model to consider.
        link_indices: The indices of the links.

    Returns:
        The names of the links.
    """

    d = {l.index: l.name for l in model.physics_model.description.links_dict.values()}
    return tuple(d[i] for i in link_indices)


# =========
# Link APIs
# =========


def mass(model: Model.JaxSimModel, *, link_index: jtp.IntLike) -> jtp.Float:
    """"""

    return model.physics_model._link_masses[link_index].astype(float)


def spatial_inertia(model: Model.JaxSimModel, *, link_index: jtp.IntLike) -> jtp.Matrix:
    """"""

    return model.physics_model._link_spatial_inertias[link_index]


@jax.jit
def transform(
    model: Model.JaxSimModel, data: Data.JaxSimModelData, *, link_index: jtp.IntLike
) -> jtp.Matrix:
    """
    Compute the SE(3) transform from the world frame to the link frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.

    Returns:
        The 4x4 matrix representing the transform.
    """

    from jaxsim.helpers.model import forward_kinematics

    return forward_kinematics(model=model, data=data)[link_index]


@jax.jit
def com_position(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    in_link_frame: jtp.BoolLike = True,
) -> jtp.Vector:
    """
    Compute the position of the center of mass of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        in_link_frame:
            Whether to return the position in the link frame or in the world frame.

    Returns:
        The 3D position of the center of mass of the link.
    """

    from jaxsim.math.inertia import Inertia

    _, L_p_CoM, _ = Inertia.to_params(
        M=spatial_inertia(model=model, link_index=link_index)
    )

    def com_in_link_frame():
        return L_p_CoM.squeeze()

    def com_in_inertial_frame():
        W_H_L = transform(link_index=link_index, model=model, data=data)
        W_p̃_CoM = W_H_L @ jnp.hstack([L_p_CoM.squeeze(), 1])

        return W_p̃_CoM[0:3].squeeze()

    return jax.lax.select(
        pred=in_link_frame,
        on_true=com_in_link_frame(),
        on_false=com_in_inertial_frame(),
    )
