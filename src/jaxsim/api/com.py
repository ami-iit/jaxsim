import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp


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
