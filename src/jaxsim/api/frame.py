import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.api as js
import jaxsim.typing as jtp

# =======================
# Index-related functions
# =======================


@functools.partial(jax.jit, static_argnames="frame_name")
def name_to_idx(model: js.model.JaxSimModel, *, frame_name: str) -> jtp.Int:
    """
    Convert the name of a frame to its index.

    Args:
        model: The model to consider.
        frame_name: The name of the frame.

    Returns:
        The index of the frame.
    """

    frame_names = np.array([frame.name for frame in model.description.get().frames])

    if frame_name in frame_names:
        idx_in_list = np.argwhere(frame_names == frame_name)
        return jnp.array(idx_in_list + model.number_of_links()).squeeze().astype(int)
    return jnp.array(-1).astype(int)


def idx_to_name(model: js.model.JaxSimModel, *, frame_index: jtp.IntLike) -> str:
    """
    Convert the index of a frame to its name.

    Args:
        model: The model to consider.
        frame_index: The index of the frame.

    Returns:
        The name of the frame.
    """

    return model.description.get().frames[frame_index - model.number_of_links()].name


@functools.partial(jax.jit, static_argnames="frame_names")
def names_to_idxs(
    model: js.model.JaxSimModel, *, frame_names: Sequence[str]
) -> jax.Array:
    """
    Convert a sequence of frame names to their corresponding indices.

    Args:
        model: The model to consider.
        frame_names: The names of the frames.

    Returns:
        The indices of the frames.
    """

    return jnp.array(
        [name_to_idx(model=model, frame_name=frame_name) for frame_name in frame_names]
    ).astype(int)


def idxs_to_names(
    model: js.model.JaxSimModel, *, frame_indices: Sequence[jtp.IntLike]
) -> tuple[str, ...]:
    """
    Convert a sequence of frame indices to their corresponding names.

    Args:
        model: The model to consider.
        frame_indices: The indices of the frames.

    Returns:
        The names of the frames.
    """

    return tuple(
        idx_to_name(model=model, frame_index=frame_index)
        for frame_index in frame_indices
    )


# =========
# Frame APIs
# =========


@functools.partial(jax.jit, static_argnames=("frame_index",))
def transform(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
) -> jtp.Matrix:
    """
    Compute the SE(3) transform from the world frame to the specified frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame for which the transform is requested.

    Returns:
        The 4x4 matrix representing the transform.
    """

    frame = model.description.get().frames[frame_index - model.number_of_links()]
    parent_link_index = frame.parent.index
    W_H_parent = js.link.transform(model=model, data=data, link_index=parent_link_index)
    parent_H_frame = frame.pose

    return W_H_parent @ parent_H_frame
