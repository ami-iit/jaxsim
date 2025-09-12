import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


def collidable_shapes_pos_vel(
    model: js.model.JaxSimModel,
    *,
    link_transforms: jtp.Matrix,
    link_velocities: jtp.Matrix,
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """

    Compute the position and linear velocity of the enabled collidable shapes in the world frame.

    Args:
        model: The model to consider.
        link_transforms: The transforms from the world frame to each link.
        link_velocities: The linear and angular velocities of each link.

    Returns:
        A tuple containing the position and linear velocity of the enabled collidable shapes.
    """

    # Get the indices of the enabled collidable shapes.
    indices_of_enabled_collidable_shapes = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_shapes
    )

    parent_link_idx_of_enabled_collidable_shapes = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_shapes]

    L_p_Ci = model.kin_dyn_parameters.contact_parameters.shape[
        indices_of_enabled_collidable_shapes
    ]

    if len(indices_of_enabled_collidable_shapes) == 0:
        return jnp.array(0).astype(float), jnp.empty(0).astype(float)

    def process_shape_kinematics(
        Li_p_C: jtp.Vector, parent_body: jtp.Int
    ) -> tuple[jtp.Vector, jtp.Vector]:

        # Compute the position of the collidable shape.
        W_p_Ci = (link_transforms[parent_body] @ jnp.hstack([Li_p_C, 1]))[0:3]

        return W_p_Ci

    # Process all the collidable shapes in parallel.
    W_p_Ci = jax.vmap(process_shape_kinematics)(
        L_p_Ci,
        parent_link_idx_of_enabled_collidable_shapes,
    )

    return W_p_Ci, link_velocities[:, :3]
