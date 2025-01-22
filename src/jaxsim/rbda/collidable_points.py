import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Skew


def collidable_points_pos_vel(
    model: js.model.JaxSimModel,
    *,
    link_transforms: jtp.Matrix,
    link_velocities: jtp.Matrix,
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """

    Compute the position and linear velocity of the enabled collidable points in the world frame.

    Args:
        model: The model to consider.
        link_transforms: The SE(3) transforms of the model links in the world frame.
        link_velocities: The 6D velocities of the model links in the world frame.

    Returns:
        A tuple containing the position and linear velocity of the enabled collidable points.
    """

    # Get the indices of the enabled collidable points.
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    L_p_Ci = model.kin_dyn_parameters.contact_parameters.point[
        indices_of_enabled_collidable_points
    ]

    if len(indices_of_enabled_collidable_points) == 0:
        return jnp.array(0).astype(float), jnp.empty(0).astype(float)

    W_H_LL = link_transforms

    W_v_Wi = link_velocities

    # ==================================================
    # Compute position and velocity of collidable points
    # ==================================================

    def process_point_kinematics(
        Li_p_C: jtp.Vector, W_H_L: jtp.Vector, parent_body: jtp.Int
    ) -> tuple[jtp.Vector, jtp.Vector]:

        # Compute the position of the collidable point.
        W_p_Ci = (W_H_L[parent_body] @ jnp.hstack([Li_p_C, 1]))[0:3]

        # Compute the linear part of the mixed velocity Ci[W]_v_{W,Ci}.
        CW_vl_WCi = (
            jnp.block([jnp.eye(3), -Skew.wedge(vector=W_p_Ci).squeeze()])
            @ W_v_Wi[parent_body].squeeze()
        )

        return W_p_Ci, CW_vl_WCi

    # Process all the collidable points in parallel.
    W_p_Ci, CW_vl_WC = jax.vmap(process_point_kinematics)(
        L_p_Ci,
        W_H_LL,
        parent_link_idx_of_enabled_collidable_points,
    )

    return W_p_Ci, CW_vl_WC
