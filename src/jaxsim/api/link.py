from typing import Sequence

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp

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
