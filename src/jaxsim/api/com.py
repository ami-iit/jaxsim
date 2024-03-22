import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
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
