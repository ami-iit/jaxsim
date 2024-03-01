import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.physics.algos.jacobian
import jaxsim.typing as jtp
from jaxsim.high_level.common import VelRepr

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

    return Model.forward_kinematics(model=model, data=data)[link_index]


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


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def jacobian(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
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
        The 6x(6+dofs) free-floating jacobian of the link.

    Note:
        The input representation of the free-floating jacobian is the active
        velocity representation.
    """

    if output_vel_repr is None:
        output_vel_repr = data.velocity_representation

    # Compute the doubly left-trivialized free-floating jacobian
    L_J_WL_B = jaxsim.physics.algos.jacobian.jacobian(
        model=model.physics_model,
        body_index=link_index,
        q=data.joint_positions(),
    )

    match data.velocity_representation:

        case VelRepr.Body:
            L_J_WL_target = L_J_WL_B

        case VelRepr.Inertial:
            dofs = model.dofs()
            W_H_B = data.base_transform()

            B_X_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, dofs))

            B_T_W = jnp.vstack(
                [
                    jnp.block([B_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(dofs)]),
                ]
            )

            L_J_WL_target = L_J_WL_B @ B_T_W

        case VelRepr.Mixed:
            dofs = model.dofs()
            W_H_B = data.base_transform()
            BW_H_B = jnp.array(W_H_B).at[0:3, 3].set(jnp.zeros(3))

            B_X_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, dofs))

            B_T_BW = jnp.vstack(
                [
                    jnp.block([B_X_BW, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(dofs)]),
                ]
            )

            L_J_WL_target = L_J_WL_B @ B_T_BW

        case _:
            raise ValueError(data.velocity_representation)

    match output_vel_repr:
        case VelRepr.Body:
            return L_J_WL_target

        case VelRepr.Inertial:
            W_H_L = transform(model=model, data=data, link_index=link_index)
            W_X_L = jaxlie.SE3.from_matrix(W_H_L).adjoint()
            return W_X_L @ L_J_WL_target

        case VelRepr.Mixed:
            W_H_L = transform(model=model, data=data, link_index=link_index)
            LW_H_L = jnp.array(W_H_L).at[0:3, 3].set(jnp.zeros(3))
            LW_X_L = jaxlie.SE3.from_matrix(LW_H_L).adjoint()
            return LW_X_L @ L_J_WL_target

        case _:
            raise ValueError(output_vel_repr)
