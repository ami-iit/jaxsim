import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp

from .common import VelRepr


@jax.jit
def com_position(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    """
    Compute the position of the center of mass of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the center of mass of the model w.r.t. the world frame.
    """

    m = js.model.total_mass(model=model)

    W_H_L = js.model.forward_kinematics(model=model, data=data)
    W_H_B = data.base_transform()
    B_H_W = jaxlie.SE3.from_matrix(W_H_B).inverse().as_matrix()

    def B_p̃_LCoM(i) -> jtp.Vector:
        m = js.link.mass(model=model, link_index=i)
        L_p_LCoM = js.link.com_position(
            model=model, data=data, link_index=i, in_link_frame=True
        )
        return m * B_H_W @ W_H_L[i] @ jnp.hstack([L_p_LCoM, 1])

    com_links = jax.vmap(B_p̃_LCoM)(jnp.arange(model.number_of_links()))

    B_p̃_CoM = (1 / m) * com_links.sum(axis=0)
    B_p̃_CoM = B_p̃_CoM.at[3].set(1)

    return (W_H_B @ B_p̃_CoM)[0:3].astype(float)


@jax.jit
def com_linear_velocity(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the linear velocity of the center of mass of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The linear velocity of the center of mass of the model in the
        active representation.

    Note:
        The linear velocity of the center of mass  is expressed in the mixed frame
        :math:`G = ({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`[C] = [W]` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`[C] = [B]` if the active velocity representation is body-fixed.
    """

    # Extract the linear component of the 6D average centroidal velocity.
    # This is expressed in G[B] in body-fixed representation, and in G[W] in
    # inertial-fixed or mixed representation.
    G_vl_WG = average_centroidal_velocity(model=model, data=data)[0:3]

    return G_vl_WG


@jax.jit
def centroidal_momentum(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the centroidal momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The centroidal momentum of the model.

    Note:
        The centroidal momentum is expressed in the mixed frame
        :math:`({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`C = W` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`C = B` if the active velocity representation is body-fixed.
    """

    ν = data.generalized_velocity()
    G_J = centroidal_momentum_jacobian(model=model, data=data)

    return G_J @ ν


@jax.jit
def centroidal_momentum_jacobian(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    r"""
    Compute the Jacobian of the centroidal momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The Jacobian of the centroidal momentum of the model.

    Note:
        The frame corresponding to the output representation of this Jacobian is either
        :math:`G[W]`, if the active velocity representation is inertial-fixed or mixed,
        or :math:`G[B]`, if the active velocity representation is body-fixed.

    Note:
        This Jacobian is also known in the literature as Centroidal Momentum Matrix.
    """

    # Compute the Jacobian of the total momentum with body-fixed output representation.
    # We convert the output representation either to G[W] or G[B] below.
    B_Jh = js.model.total_momentum_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Body
    )

    W_H_B = data.base_transform()
    B_H_W = jaxsim.math.Transform.inverse(W_H_B)

    W_p_CoM = com_position(model=model, data=data)

    match data.velocity_representation:
        case VelRepr.Inertial | VelRepr.Mixed:
            W_H_G = W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
        case VelRepr.Body:
            W_H_G = W_H_GB = W_H_B.at[0:3, 3].set(W_p_CoM)
        case _:
            raise ValueError(data.velocity_representation)

    # Compute the transform for 6D forces.
    G_Xf_B = jaxsim.math.Adjoint.from_transform(transform=B_H_W @ W_H_G).T

    return G_Xf_B @ B_Jh


@jax.jit
def locked_centroidal_spatial_inertia(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
):
    """
    Compute the locked centroidal spatial inertia of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The locked centroidal spatial inertia of the model.
    """

    with data.switch_velocity_representation(VelRepr.Body):
        B_Mbb_B = js.model.locked_spatial_inertia(model=model, data=data)

    W_H_B = data.base_transform()
    W_p_CoM = com_position(model=model, data=data)

    match data.velocity_representation:
        case VelRepr.Inertial | VelRepr.Mixed:
            W_H_G = W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
        case VelRepr.Body:
            W_H_G = W_H_GB = W_H_B.at[0:3, 3].set(W_p_CoM)
        case _:
            raise ValueError(data.velocity_representation)

    B_H_G = jaxlie.SE3.from_matrix(jaxsim.math.Transform.inverse(W_H_B) @ W_H_G)

    B_Xv_G = B_H_G.adjoint()
    G_Xf_B = B_Xv_G.transpose()

    return G_Xf_B @ B_Mbb_B @ B_Xv_G


@jax.jit
def average_centroidal_velocity(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the average centroidal velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The average centroidal velocity of the model.

    Note:
        The average velocity is expressed in the mixed frame
        :math:`G = ({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`[C] = [W]` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`[C] = [B]` if the active velocity representation is body-fixed.
    """

    ν = data.generalized_velocity()
    G_J = average_centroidal_velocity_jacobian(model=model, data=data)

    return G_J @ ν


@jax.jit
def average_centroidal_velocity_jacobian(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    r"""
    Compute the Jacobian of the average centroidal velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The Jacobian of the average centroidal velocity of the model.

    Note:
        The frame corresponding to the output representation of this Jacobian is either
        :math:`G[W]`, if the active velocity representation is inertial-fixed or mixed,
        or :math:`G[B]`, if the active velocity representation is body-fixed.
    """

    G_J = centroidal_momentum_jacobian(model=model, data=data)
    G_Mbb = locked_centroidal_spatial_inertia(model=model, data=data)

    return jnp.linalg.inv(G_Mbb) @ G_J
