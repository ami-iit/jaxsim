import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp

from .common import VelRepr

# =======================
# Index-related functions
# =======================


@functools.partial(jax.jit, static_argnames="link_name")
def name_to_idx(model: js.model.JaxSimModel, *, link_name: str) -> jtp.Int:
    """
    Convert the name of a link to its index.

    Args:
        model: The model to consider.
        link_name: The name of the link.

    Returns:
        The index of the link.
    """

    if link_name in model.kin_dyn_parameters.link_names:
        return (
            jnp.array(
                np.argwhere(np.array(model.kin_dyn_parameters.link_names) == link_name)
            )
            .squeeze()
            .astype(int)
        )
    return jnp.array(-1).astype(int)


def idx_to_name(model: js.model.JaxSimModel, *, link_index: jtp.IntLike) -> str:
    """
    Convert the index of a link to its name.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The name of the link.
    """

    return model.kin_dyn_parameters.link_names[link_index]


@functools.partial(jax.jit, static_argnames="link_names")
def names_to_idxs(
    model: js.model.JaxSimModel, *, link_names: Sequence[str]
) -> jax.Array:
    """
    Convert a sequence of link names to their corresponding indices.

    Args:
        model: The model to consider.
        link_names: The names of the links.

    Returns:
        The indices of the links.
    """

    return jnp.array(
        [name_to_idx(model=model, link_name=name) for name in link_names],
    ).astype(int)


def idxs_to_names(
    model: js.model.JaxSimModel, *, link_indices: Sequence[jtp.IntLike] | jtp.VectorLike
) -> tuple[str, ...]:
    """
    Convert a sequence of link indices to their corresponding names.

    Args:
        model: The model to consider.
        link_indices: The indices of the links.

    Returns:
        The names of the links.
    """

    return tuple(idx_to_name(model=model, link_index=idx) for idx in link_indices)


# =========
# Link APIs
# =========


@jax.jit
def mass(model: js.model.JaxSimModel, *, link_index: jtp.IntLike) -> jtp.Float:
    """
    Return the mass of the link.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The mass of the link.
    """

    return model.kin_dyn_parameters.link_parameters.mass[link_index].astype(float)


@jax.jit
def spatial_inertia(
    model: js.model.JaxSimModel, *, link_index: jtp.IntLike
) -> jtp.Matrix:
    """
    Compute the 6D spatial inertial of the link.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The 6×6 matrix representing the spatial inertia of the link expressed in
        the link frame (body-fixed representation).
    """

    link_parameters = jax.tree_util.tree_map(
        lambda l: l[link_index], model.kin_dyn_parameters.link_parameters
    )

    return js.kin_dyn_parameters.LinkParameters.spatial_inertia(link_parameters)


@jax.jit
def transform(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
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

    return js.model.forward_kinematics(model=model, data=data)[link_index]


@jax.jit
def com_position(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
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


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating jacobian of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The 6×(6+n) free-floating jacobian of the link.

    Note:
        The input representation of the free-floating jacobian is the active
        velocity representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Compute the doubly left-trivialized free-floating jacobian.
    L_J_WL_B = jaxsim.rbda.jacobian(
        model=model,
        link_index=link_index,
        joint_positions=data.joint_positions(),
    )

    # We want to return the Jacobian O_J_WL_I, where O is the representation of the
    # output 6D velocity, and I is the representation of the input generalized velocity.

    # Change the input representation matching the one of data.
    match data.velocity_representation:

        case VelRepr.Body:
            L_J_WL_I = L_J_WL_B

        case VelRepr.Inertial:

            W_H_B = data.base_transform()

            B_X_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
            B_T_W = jax.scipy.linalg.block_diag(B_X_W, jnp.eye(model.dofs()))

            L_J_WL_I = L_J_WL_B @ B_T_W

        case VelRepr.Mixed:

            W_H_B = data.base_transform()
            BW_H_B = jnp.array(W_H_B).at[0:3, 3].set(jnp.zeros(3))

            B_X_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            B_T_BW = jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))

            L_J_WL_I = L_J_WL_B @ B_T_BW

        case _:
            raise ValueError(data.velocity_representation)

    # Change the output representation matching specified one.
    match output_vel_repr:

        case VelRepr.Body:
            O_J_WL_I = L_J_WL_I

        case VelRepr.Inertial:

            W_H_L = transform(model=model, data=data, link_index=link_index)
            W_X_L = jaxlie.SE3.from_matrix(W_H_L).adjoint()
            O_J_WL_I = W_X_L @ L_J_WL_I

        case VelRepr.Mixed:

            W_H_L = transform(model=model, data=data, link_index=link_index)
            LW_H_L = jnp.array(W_H_L).at[0:3, 3].set(jnp.zeros(3))
            LW_X_L = jaxlie.SE3.from_matrix(LW_H_L).adjoint()
            O_J_WL_I = LW_X_L @ L_J_WL_I

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def velocity(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Vector:
    """
    Compute the 6D velocity of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        output_vel_repr:
            The output velocity representation of the link velocity.

    Returns:
        The 6D velocity of the link in the specified velocity representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the link jacobian having I as input representation (taken from data)
    # and O as output representation, specified by the user (or taken from data).
    O_J_WL_I = jacobian(
        model=model,
        data=data,
        link_index=link_index,
        output_vel_repr=output_vel_repr,
    )

    # Get the generalized velocity in the input velocity representation.
    I_ν = data.generalized_velocity()

    # Compute the link velocity in the output velocity representation.
    return O_J_WL_I @ I_ν
