import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp

from .common import VelRepr

# =======================
# Index-related functions
# =======================


def idx_of_parent_link(model: js.model.JaxSimModel, *, frame_idx: jtp.IntLike) -> int:
    """
    Get the index of the link to which the frame is rigidly attached.

    Args:
        model: The model to consider.
        frame_idx: The index of the frame.

    Returns:
        The index of the frame's parent link.
    """

    # Get the intermediate representation parsed from the model description.
    ir = model.description

    # Extract the indices of the frame and the link it is attached to.
    F = ir.frames[frame_idx - model.number_of_links()]
    L = ir.links_dict[F.parent.name].index

    return int(L)


def name_to_idx(model: js.model.JaxSimModel, *, frame_name: str) -> int:
    """
    Convert the name of a frame to its index.

    Args:
        model: The model to consider.
        frame_name: The name of the frame.

    Returns:
        The index of the frame.
    """

    frame_names = np.array([frame.name for frame in model.description.frames])

    if frame_name in frame_names:
        idx_in_list = np.argwhere(frame_names == frame_name)
        return int(idx_in_list.squeeze().tolist()) + model.number_of_links()

    return -1


def idx_to_name(model: js.model.JaxSimModel, *, frame_index: jtp.IntLike) -> str:
    """
    Convert the index of a frame to its name.

    Args:
        model: The model to consider.
        frame_index: The index of the frame.

    Returns:
        The name of the frame.
    """

    return model.description.frames[frame_index - model.number_of_links()].name


@functools.partial(jax.jit, static_argnames=["frame_names"])
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


# ==========
# Frame APIs
# ==========


@functools.partial(jax.jit, static_argnames=["frame_index"])
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

    # Compute the necessary transforms.
    L = idx_of_parent_link(model=model, frame_idx=frame_index)
    W_H_L = js.link.transform(model=model, data=data, link_index=L)

    # Get the static frame pose wrt the parent link.
    frame = model.description.frames[frame_index - model.number_of_links()]
    L_H_F = frame.pose

    # Combine the transforms computing the frame pose.
    return W_H_L @ L_H_F


@functools.partial(jax.jit, static_argnames=["frame_index", "output_vel_repr"])
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating jacobian of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The 6×(6+n) free-floating jacobian of the frame.

    Note:
        The input representation of the free-floating jacobian is the active
        velocity representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the index of the parent link.
    L = idx_of_parent_link(model=model, frame_idx=frame_index)

    # Compute the Jacobian of the parent link using body-fixed output representation.
    L_J_WL = js.link.jacobian(
        model=model, data=data, link_index=L, output_vel_repr=VelRepr.Body
    )

    # Adjust the output representation
    match output_vel_repr:
        case VelRepr.Inertial:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_X_L = jaxlie.SE3.from_matrix(W_H_L).adjoint()
            W_J_WL = W_X_L @ L_J_WL
            O_J_WL_I = W_J_WL

        case VelRepr.Body:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            F_H_L = jaxsim.math.Transform.inverse(W_H_F) @ W_H_L
            F_X_L = jaxlie.SE3.from_matrix(F_H_L).adjoint()
            F_J_WL = F_X_L @ L_J_WL
            O_J_WL_I = F_J_WL

        case VelRepr.Mixed:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            F_H_L = jaxsim.math.Transform.inverse(W_H_F) @ W_H_L
            FW_H_F = W_H_F.at[0:3, 3].set(jnp.zeros(3))
            FW_H_L = FW_H_F @ F_H_L
            FW_X_L = jaxlie.SE3.from_matrix(FW_H_L).adjoint()
            FW_J_WL = FW_X_L @ L_J_WL
            O_J_WL_I = FW_J_WL

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I
