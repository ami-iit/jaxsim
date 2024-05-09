import functools

import jax

import jaxsim.api as js
import jaxsim.typing as jtp


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
