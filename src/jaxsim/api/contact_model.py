from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


@jax.jit
@js.common.named_scope
def link_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_torques: jtp.VectorLike | None = None,
) -> jtp.Matrix:
    """
    Compute the 6D contact forces of all links of the model in inertial representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D external forces to apply to the links expressed in inertial representation
        joint_torques:
            The joint torques acting on the joints.

    Returns:
        A `(nL, 6)` array containing the stacked 6D contact forces of the links,
        expressed in inertial representation.
    """

    # Compute the contact forces for each collidable point with the active contact model.
    W_f_C, aux_data = model.contact_model.compute_contact_forces(
        model=model,
        data=data,
        link_forces=link_forces,
        joint_force_references=joint_torques,
    )

    # Compute the 6D forces applied to the links equivalent to the forces applied
    # to the frames associated to the collidable points.
    W_f_L = link_forces_from_contact_forces(model=model, contact_forces=W_f_C)

    wrench_pair_constr_inertial = aux_data["constr_wrenches_inertial"]

    constraints = model.kin_dyn_parameters.get_constraints(model)
    # Get the couples of parent link indices of each couple of frames.
    frame_idxs_1, frame_idxs_2, types = zip(*constraints, strict=False)
    frame_idxs_1 = jnp.array(frame_idxs_1)
    frame_idxs_2 = jnp.array(frame_idxs_2)

    jax.debug.print("frame_idxs_1: \n{}", frame_idxs_1)
    jax.debug.print("frame_idxs_2: \n{}", frame_idxs_2)

    parent_link_indices = jax.vmap(
        lambda frame_idx_1, frame_idx_2: (
            js.frame.idx_of_parent_link(model, frame_index=frame_idx_1),
            js.frame.idx_of_parent_link(model, frame_index=frame_idx_2),
        )
    )(frame_idxs_1, frame_idxs_2)
    parent_link_indices = jnp.array(parent_link_indices)
    jax.debug.print("parent_link_indices: \n{}", parent_link_indices.shape)

    # Apply each constraint wrench to its corresponding parent link in W_f_L.
    def apply_wrench(i, W_f_L):
        parent_indices = parent_link_indices[:, i]
        wrench_pair = wrench_pair_constr_inertial[:, i]
        jax.debug.print("parent_indices: \n{}", parent_indices)
        jax.debug.print("wrench_pair: \n{}", wrench_pair)
        W_f_L = W_f_L.at[parent_indices[0]].add(wrench_pair[0])
        W_f_L = W_f_L.at[parent_indices[1]].add(wrench_pair[1])
        return W_f_L

    W_f_L = jax.lax.fori_loop(0, parent_link_indices.shape[0], apply_wrench, W_f_L)

    jax.debug.print("W_f_L: \n{}", W_f_L)

    return W_f_L


@staticmethod
def link_forces_from_contact_forces(
    model: js.model.JaxSimModel,
    *,
    contact_forces: jtp.MatrixLike,
) -> jtp.Matrix:
    """
    Compute the link forces from the contact forces.

    Args:
        model: The robot model considered by the contact model.
        data: The data of the considered model.
        contact_forces: The contact forces computed by the contact model.

    Returns:
        The 6D contact forces applied to the links and expressed in the frame of
        the velocity representation of data.
    """

    # Get the object storing the contact parameters of the model.
    contact_parameters = model.kin_dyn_parameters.contact_parameters

    # Extract the indices corresponding to the enabled collidable points.
    indices_of_enabled_collidable_points = (
        contact_parameters.indices_of_enabled_collidable_points
    )

    # Convert the contact forces to a JAX array.
    W_f_C = jnp.atleast_2d(jnp.array(contact_forces, dtype=float).squeeze())

    # Construct the vector defining the parent link index of each collidable point.
    # We use this vector to sum the 6D forces of all collidable points rigidly
    # attached to the same link.
    parent_link_index_of_collidable_points = jnp.array(
        contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    # Create the mask that associate each collidable point to their parent link.
    # We use this mask to sum the collidable points to the right link.
    mask = parent_link_index_of_collidable_points[:, jnp.newaxis] == jnp.arange(
        model.number_of_links()
    )

    # Sum the forces of all collidable points rigidly attached to a body.
    # Since the contact forces W_f_C are expressed in the world frame,
    # we don't need any coordinate transformation.
    W_f_L = mask.T @ W_f_C

    return W_f_L
