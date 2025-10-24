from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.exceptions
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import Adjoint, Cross, Transform
from jaxsim.rbda.contacts import SoftContacts, detection

from .common import VelRepr


@jax.jit
@js.common.named_scope
def contact_point_kinematics(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """
    Compute the position and 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position and velocity of the collidable points in the world frame.

    Note:
        The collidable point velocity is the plain coordinate derivative of the position.
        If we attach a frame C = (p_C, [C]) to the collidable point, it corresponds to
        the linear component of the mixed 6D frame velocity.
    """

    _, _, _, W_p_Ci, W_ṗ_Ci = jax.vmap(
        lambda shape_transform, shape_type, shape_size, link_transform, link_velocity: jaxsim.rbda.contacts.common.compute_penetration_data(
            model,
            shape_transform=shape_transform,
            shape_type=shape_type,
            shape_size=shape_size,
            link_transforms=link_transform,
            link_velocities=link_velocity,
        )
    )(
        model.kin_dyn_parameters.contact_parameters.transform,
        model.kin_dyn_parameters.contact_parameters.shape_type,
        model.kin_dyn_parameters.contact_parameters.shape_size,
        data._link_transforms[
            jnp.array(model.kin_dyn_parameters.contact_parameters.body)
        ],
        data._link_velocities[
            jnp.array(model.kin_dyn_parameters.contact_parameters.body)
        ],
    )

    return W_p_Ci, W_ṗ_Ci


@jax.jit
@js.common.named_scope
def contact_point_positions(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the position of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the collidable points in the world frame.
    """

    W_p_Ci, _ = contact_point_kinematics(model=model, data=data)

    return W_p_Ci


@jax.jit
@js.common.named_scope
def contact_point_velocities(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The 3D velocity of the collidable points.
    """

    _, W_ṗ_Ci = contact_point_kinematics(model=model, data=data)

    return W_ṗ_Ci


@functools.partial(jax.jit, static_argnames=["link_names"])
@js.common.named_scope
def in_contact(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_names: tuple[str, ...] | None = None,
) -> jtp.Vector:
    """
    Return whether the links are in contact with the terrain.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_names:
            The names of the links to consider. If None, all links are considered.

    Returns:
        A boolean vector indicating whether the links are in contact with the terrain.
    """

    if link_names is not None and set(link_names).difference(model.link_names()):
        raise ValueError("One or more link names are not part of the model")

    # Get the indices of the enabled collidable points.
    indices_of_enabled_contact_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_contact_points
    )

    parent_link_idx_of_enabled_contact_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_contact_points]

    W_p_Ci = contact_point_positions(model=model, data=data)

    terrain_height = jax.vmap(lambda x, y: model.terrain.height(x=x, y=y))(
        W_p_Ci[:, 0], W_p_Ci[:, 1]
    )

    below_terrain = W_p_Ci[:, 2] <= terrain_height

    link_idxs = (
        js.link.names_to_idxs(link_names=link_names, model=model)
        if link_names is not None
        else jnp.arange(model.number_of_links())
    )

    links_in_contact = jax.vmap(
        lambda link_index: jnp.where(
            parent_link_idx_of_enabled_contact_points == link_index,
            below_terrain,
            jnp.zeros_like(below_terrain, dtype=bool),
        ).any()
    )(link_idxs)

    return links_in_contact


def estimate_good_soft_contacts_parameters(
    *args, **kwargs
) -> jaxsim.rbda.contacts.ContactParamsTypes:
    """
    Estimate good soft contacts parameters. Deprecated, use `estimate_good_contact_parameters` instead.
    """

    msg = "This method is deprecated, please use `{}`."
    logging.warning(msg.format(estimate_good_contact_parameters.__name__))
    return estimate_good_contact_parameters(*args, **kwargs)


def estimate_good_contact_parameters(
    model: js.model.JaxSimModel,
    *,
    standard_gravity: jtp.FloatLike = jaxsim.math.STANDARD_GRAVITY,
    static_friction_coefficient: jtp.FloatLike = 0.5,
    number_of_active_contact_points_steady_state: jtp.IntLike = 1,
    damping_ratio: jtp.FloatLike = 1.0,
    max_penetration: jtp.FloatLike | None = None,
) -> jaxsim.rbda.contacts.ContactParamsTypes:
    """
    Estimate good contact parameters.

    Args:
        model: The model to consider.
        standard_gravity: The standard gravity acceleration.
        static_friction_coefficient: The static friction coefficient.
        number_of_active_contact_points_steady_state:
            The number of active collidable points in steady state.
        damping_ratio: The damping ratio.
        max_penetration: The maximum penetration allowed.

    Returns:
        The estimated good contacts parameters.

    Note:
        This is primarily a convenience function for soft-like contact models.
        However, it provides with some good default parameters also for the other ones.

    Note:
        This method provides a good set of contacts parameters.
        The user is encouraged to fine-tune the parameters based on the
        specific application.
    """
    if max_penetration is None:
        zero_data = js.data.JaxSimModelData.build(model=model)
        W_pz_CoM = js.com.com_position(model=model, data=zero_data)[2]
        if model.floating_base():
            W_pz_C = contact_point_positions(model=model, data=zero_data)[:, -1]
            W_pz_CoM = W_pz_CoM - W_pz_C.min()

        # Consider as default a 1% of the model center of mass height.
        max_penetration = 0.01 * W_pz_CoM

    nc = number_of_active_contact_points_steady_state
    return model.contact_model._parameters_class().build_default_from_jaxsim_model(
        model=model,
        standard_gravity=standard_gravity,
        static_friction_coefficient=static_friction_coefficient,
        max_penetration=max_penetration,
        number_of_active_contact_points_steady_state=nc,
        damping_ratio=damping_ratio,
    )


@jax.jit
@js.common.named_scope
def transforms(model: js.model.JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Array:
    r"""
    Return the pose of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The stacked SE(3) matrices of all enabled collidable points.

    Note:
        The output shape is (nL, 3, 4, 4), where nL is the number of links.
        Three candidate contact points are considered for each collidable shape.
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    # Get the transforms of the parent link of all collidable points.
    W_H_L = data._link_transforms

    # Index transforms by the body (parent link) of each collision shape
    body_indices = jnp.array(model.kin_dyn_parameters.contact_parameters.body)
    W_H_L_indexed = W_H_L[body_indices]

    def _process_single_shape(shape_type, shape_size, shape_transform, W_H_Li):
        # Apply the collision shape transform to get W_H_S
        W_H_S = W_H_Li @ shape_transform

        _, W_H_C = jax.lax.switch(
            shape_type,
            (detection.box_plane, detection.cylinder_plane, detection.sphere_plane),
            model.terrain,
            shape_size,
            W_H_S,
        )

        return W_H_C

    return jax.vmap(_process_single_shape)(
        model.kin_dyn_parameters.contact_parameters.shape_type,
        model.kin_dyn_parameters.contact_parameters.shape_size,
        model.kin_dyn_parameters.contact_parameters.transform,
        W_H_L_indexed,
    )


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
@js.common.named_scope
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Array:
    r"""
    Return the free-floating Jacobian of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The stacked :math:`6 \times (6+n)` free-floating jacobians of the frames associated to the
        enabled collidable points.

    Note:
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    output_vel_repr = output_vel_repr or data.velocity_representation

    # Compute link-level Jacobians (n_links, 6, 6+n)
    W_J_WL = js.model.generalized_free_floating_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Inertial
    )

    # Compute contact transforms (n_shapes, n_contacts_per_shape, 4, 4)
    W_H_C = transforms(model=model, data=data)

    # Index Jacobians by the body (parent link) of each collision shape
    body_indices = jnp.array(model.kin_dyn_parameters.contact_parameters.body)
    W_J_WL_indexed = W_J_WL[body_indices]  # (n_shapes, 6, 6+n)

    # Repeat for each contact point per shape: (n_shapes*n_contacts_per_shape, 6, 6+n)
    W_J_WC_flat = jnp.repeat(W_J_WL_indexed, 3, axis=0)

    # Flatten contact transforms (n_shapes*n_contacts_per_shape, 4, 4)
    W_H_C_flat = W_H_C.reshape(-1, 4, 4)

    # Transform Jacobian based on velocity representation
    match output_vel_repr:

        case VelRepr.Inertial:
            return W_J_WC_flat

        case VelRepr.Body:

            def transform_jacobian(H_C, J_WC):
                return jaxsim.math.Adjoint.from_transform(H_C, inverse=True) @ J_WC

        case VelRepr.Mixed:

            def transform_jacobian(H_C, J_WC):
                H_CW = H_C.at[0:3, 0:3].set(jnp.eye(3))
                return jaxsim.math.Adjoint.from_transform(H_CW, inverse=True) @ J_WC

        case _:
            raise ValueError(f"Unsupported velocity representation: {output_vel_repr}")

    # Single vmap over all contact points
    return jax.vmap(transform_jacobian)(W_H_C_flat, W_J_WC_flat)


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
@js.common.named_scope
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivative.

    Returns:
        The derivative of the :math:`6 \times (6+n)` free-floating jacobian of the enabled collidable points.

    Note:
        The input representation of the free-floating jacobian derivative is the active
        velocity representation.
    """

    output_vel_repr = output_vel_repr or data.velocity_representation

    # Get the link velocities.
    W_v_WL = data._link_velocities

    # Index link velocities by body (parent link) of each collision shape
    body_indices = jnp.array(model.kin_dyn_parameters.contact_parameters.body)
    W_v_WL_indexed = W_v_WL[body_indices]  # (n_shapes, 6)

    # Compute the contact transforms (n_shapes, n_contacts, 4, 4)
    W_H_C = transforms(model=model, data=data)

    # =====================================================
    # Compute quantities to adjust the input representation
    # =====================================================

    def compute_T(X: jtp.Matrix) -> jtp.Matrix:
        In = jnp.eye(model.dofs())
        T = jax.scipy.linalg.block_diag(X, In)
        return T

    def compute_Ṫ(Ẋ: jtp.Matrix) -> jtp.Matrix:
        On = jnp.zeros(shape=(model.dofs(), model.dofs()))
        Ṫ = jax.scipy.linalg.block_diag(Ẋ, On)
        return Ṫ

    # Compute the operator to change the representation of ν, and its
    # time derivative.
    match data.velocity_representation:
        case VelRepr.Inertial:
            W_X = Adjoint.from_transform(jnp.eye(4))
            W_Ẋ = jnp.zeros((6, 6))
        case VelRepr.Body:
            W_X = Adjoint.from_transform(data.base_transform)
            W_Ẋ = W_X @ Cross.vx(data.base_velocity)
        case VelRepr.Mixed:
            W_H_BW = data.base_transform.at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = Adjoint.from_transform(W_H_BW)
            BW_v_W_BW = data.base_velocity.at[3:6].set(0)
            W_X = W_X_BW
            W_Ẋ = W_X_BW @ Cross.vx(BW_v_W_BW)
        case _:
            raise ValueError(data.velocity_representation)

    T = compute_T(W_X)
    Ṫ = compute_Ṫ(W_Ẋ)

    # =====================================================
    # Compute quantities to adjust the output representation
    # =====================================================

    with data.switch_velocity_representation(VelRepr.Inertial):
        # Compute the Jacobian of the parent link in inertial representation.
        W_J_WL_W = js.model.generalized_free_floating_jacobian(model=model, data=data)

        # Compute the Jacobian derivative of the parent link in inertial representation.
        W_J̇_WL_W = js.model.generalized_free_floating_jacobian_derivative(
            model=model, data=data
        )

    # Index Jacobians by body (parent link) of each collision shape
    W_J_WL_W_indexed = W_J_WL_W[body_indices]  # (n_shapes, 6, 6+n)
    W_J̇_WL_W_indexed = W_J̇_WL_W[body_indices]  # (n_shapes, 6, 6+n)

    def compute_O_J̇_WC_I(W_H_C, W_v_WL, W_J_WL_W, W_J̇_WL_W) -> jtp.Matrix:
        match output_vel_repr:
            case VelRepr.Inertial:
                O_X_W = jnp.eye(6)
                O_Ẋ_W = jnp.zeros((6, 6))
            case VelRepr.Body:
                O_X_W = Adjoint.from_transform(W_H_C, inverse=True)
                O_Ẋ_W = -O_X_W @ Cross.vx(W_v_WL)
            case VelRepr.Mixed:
                W_H_CW = W_H_C.at[0:3, 0:3].set(jnp.eye(3))
                O_X_W = Adjoint.from_transform(Transform.inverse(W_H_CW))
                v_CW = O_X_W @ W_v_WL
                O_Ẋ_W = -O_X_W @ Cross.vx(v_CW.at[:3].set(v_CW[:3]))
            case _:
                raise ValueError(output_vel_repr)

        O_J̇_WC_I = O_Ẋ_W @ W_J_WL_W @ T
        O_J̇_WC_I += O_X_W @ W_J̇_WL_W @ T
        O_J̇_WC_I += O_X_W @ W_J_WL_W @ Ṫ

        return O_J̇_WC_I

    O_J̇_per_shape = jax.vmap(
        lambda H_C_shape, v_WL_shape, J_WL_shape, J̇_WL_shape: jax.vmap(
            compute_O_J̇_WC_I,
            in_axes=(0, None, None, None),  # Map over contacts for W_H_C only
        )(H_C_shape, v_WL_shape, J_WL_shape, J̇_WL_shape),
        in_axes=(0, 0, 0, 0),  # Map over shapes
    )(W_H_C, W_v_WL_indexed, W_J_WL_W_indexed, W_J̇_WL_W_indexed)

    O_J̇_WC = O_J̇_per_shape.reshape(-1, 6, 6 + model.dofs())

    return O_J̇_WC


@jax.jit
@js.common.named_scope
def link_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_torques: jtp.VectorLike | None = None,
) -> tuple[jtp.Matrix, dict[str, jtp.Matrix]]:
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
    W_f_L, aux_dict = model.contact_model.compute_contact_forces(
        model=model,
        data=data,
        **(
            dict(link_forces=link_forces, joint_force_references=joint_torques)
            if not isinstance(model.contact_model, SoftContacts)
            else {}
        ),
    )

    # Compute the 6D forces applied to the links equivalent to the forces applied
    # to the frames associated to the collidable points.
    # W_f_L = link_forces_from_contact_forces(model=model, contact_forces=W_f_C)

    return W_f_L, aux_dict


def link_forces_from_contact_forces(
    model: js.model.JaxSimModel,
    *,
    contact_forces: jtp.MatrixLike,
) -> jtp.Matrix:
    """
    Compute the link forces from the contact forces.

    Args:
        model: The robot model considered by the contact model.
        contact_forces: The contact forces computed by the contact model.

    Returns:
        The 6D contact forces applied to the links and expressed in the frame of
        the velocity representation of data.
    """

    # Get the object storing the contact parameters of the model.
    contact_parameters = model.kin_dyn_parameters.contact_parameters

    # Extract the indices corresponding to the enabled collidable points.
    indices_of_enabled_contact_points = (
        contact_parameters.indices_of_enabled_contact_points
    )

    # Convert the contact forces to a JAX array.
    W_f_C = jnp.atleast_2d(jnp.array(contact_forces, dtype=float).squeeze())

    # Construct the vector defining the parent link index of each collidable point.
    # We use this vector to sum the 6D forces of all collidable points rigidly
    # attached to the same link.
    parent_link_index_of_contact_points = jnp.array(contact_parameters.body, dtype=int)[
        indices_of_enabled_contact_points
    ]

    # Create the mask that associate each collidable point to their parent link.
    # We use this mask to sum the collidable points to the right link.
    mask = parent_link_index_of_contact_points[:, jnp.newaxis] == jnp.arange(
        model.number_of_links()
    )

    # Sum the forces of all collidable points rigidly attached to a body.
    # Since the contact forces W_f_C are expressed in the world frame,
    # we don't need any coordinate transformation.
    W_f_L = mask.T @ W_f_C

    return W_f_L
