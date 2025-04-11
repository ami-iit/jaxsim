from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.exceptions
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import Adjoint, Cross, Transform
from jaxsim.rbda import contacts

from .common import VelRepr


@jax.jit
@js.common.named_scope
def collidable_point_kinematics(
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

    # Switch to inertial-fixed since the RBDAs expect velocities in this representation.
    with data.switch_velocity_representation(VelRepr.Inertial):

        W_p_Ci, W_ṗ_Ci = jaxsim.rbda.collidable_points.collidable_points_pos_vel(
            model=model,
            link_transforms=data._link_transforms,
            link_velocities=data._link_velocities,
        )

    return W_p_Ci, W_ṗ_Ci


@jax.jit
@js.common.named_scope
def collidable_point_positions(
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

    W_p_Ci, _ = collidable_point_kinematics(model=model, data=data)

    return W_p_Ci


@jax.jit
@js.common.named_scope
def collidable_point_velocities(
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

    _, W_ṗ_Ci = collidable_point_kinematics(model=model, data=data)

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
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    W_p_Ci = collidable_point_positions(model=model, data=data)

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
            parent_link_idx_of_enabled_collidable_points == link_index,
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
    static_friction_coefficient: jtp.FloatLike = 0.5,
    **kwargs,
) -> jaxsim.rbda.contacts.ContactParamsTypes:
    """
    Estimate good contact parameters.

    Args:
        model: The model to consider.
        static_friction_coefficient: The static friction coefficient.
        kwargs:
            Additional model-specific parameters passed to the builder method of
            the parameters class.

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

    match model.contact_model:

        case contacts.RelaxedRigidContacts():
            assert isinstance(model.contact_model, contacts.RelaxedRigidContacts)

            parameters = contacts.RelaxedRigidContactsParams.build(
                mu=static_friction_coefficient,
                **kwargs,
            )

        case _:
            raise ValueError(f"Invalid contact model: {model.contact_model}")

    return parameters


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
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    # Get the indices of the enabled collidable points.
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    # Get the transforms of the parent link of all collidable points.
    W_H_L = data._link_transforms[parent_link_idx_of_enabled_collidable_points]

    L_p_Ci = model.kin_dyn_parameters.contact_parameters.point[
        indices_of_enabled_collidable_points
    ]

    # Build the link-to-point transform from the displacement between the link frame L
    # and the implicit contact frame C.
    L_H_C = jax.vmap(lambda L_p_C: jnp.eye(4).at[0:3, 3].set(L_p_C))(L_p_Ci)

    # Compose the work-to-link and link-to-point transforms.
    return jax.vmap(lambda W_H_Li, L_H_Ci: W_H_Li @ L_H_Ci)(W_H_L, L_H_C)


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

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the indices of the enabled collidable points.
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    # Compute the Jacobians of all links.
    W_J_WL = js.model.generalized_free_floating_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Inertial
    )

    # Compute the contact Jacobian.
    # In inertial-fixed output representation, the Jacobian of the parent link is also
    # the Jacobian of the frame C implicitly associated with the collidable point.
    W_J_WC = W_J_WL[parent_link_idx_of_enabled_collidable_points]

    # Adjust the output representation.
    match output_vel_repr:

        case VelRepr.Inertial:
            O_J_WC = W_J_WC

        case VelRepr.Body:

            W_H_C = transforms(model=model, data=data)

            def body_jacobian(W_H_C: jtp.Matrix, W_J_WC: jtp.Matrix) -> jtp.Matrix:
                C_X_W = jaxsim.math.Adjoint.from_transform(
                    transform=W_H_C, inverse=True
                )
                C_J_WC = C_X_W @ W_J_WC
                return C_J_WC

            O_J_WC = jax.vmap(body_jacobian)(W_H_C, W_J_WC)

        case VelRepr.Mixed:

            W_H_C = transforms(model=model, data=data)

            def mixed_jacobian(W_H_C: jtp.Matrix, W_J_WC: jtp.Matrix) -> jtp.Matrix:

                W_H_CW = W_H_C.at[0:3, 0:3].set(jnp.eye(3))

                CW_X_W = jaxsim.math.Adjoint.from_transform(
                    transform=W_H_CW, inverse=True
                )

                CW_J_WC = CW_X_W @ W_J_WC
                return CW_J_WC

            O_J_WC = jax.vmap(mixed_jacobian)(W_H_C, W_J_WC)

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WC


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

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    # Get the index of the parent link and the position of the collidable point.
    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    L_p_Ci = model.kin_dyn_parameters.contact_parameters.point[
        indices_of_enabled_collidable_points
    ]

    # Get the transforms of all the parent links.
    W_H_Li = data._link_transforms

    # Get the link velocities.
    W_v_WLi = data._link_velocities

    # =====================================================
    # Compute quantities to adjust the input representation
    # =====================================================

    def compute_T(model: js.model.JaxSimModel, X: jtp.Matrix) -> jtp.Matrix:
        In = jnp.eye(model.dofs())
        T = jax.scipy.linalg.block_diag(X, In)
        return T

    def compute_Ṫ(model: js.model.JaxSimModel, Ẋ: jtp.Matrix) -> jtp.Matrix:
        On = jnp.zeros(shape=(model.dofs(), model.dofs()))
        Ṫ = jax.scipy.linalg.block_diag(Ẋ, On)
        return Ṫ

    # Compute the operator to change the representation of ν, and its
    # time derivative.
    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_W = jnp.eye(4)
            W_X_W = Adjoint.from_transform(transform=W_H_W)
            W_Ẋ_W = jnp.zeros((6, 6))

            T = compute_T(model=model, X=W_X_W)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_W)

        case VelRepr.Body:
            W_H_B = data._base_transform
            W_X_B = Adjoint.from_transform(transform=W_H_B)
            B_v_WB = data.base_velocity
            B_vx_WB = Cross.vx(B_v_WB)
            W_Ẋ_B = W_X_B @ B_vx_WB

            T = compute_T(model=model, X=W_X_B)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_B)

        case VelRepr.Mixed:
            W_H_B = data._base_transform
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = Adjoint.from_transform(transform=W_H_BW)
            BW_v_WB = data.base_velocity
            BW_v_W_BW = BW_v_WB.at[3:6].set(jnp.zeros(3))
            BW_vx_W_BW = Cross.vx(BW_v_W_BW)
            W_Ẋ_BW = W_X_BW @ BW_vx_W_BW

            T = compute_T(model=model, X=W_X_BW)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_BW)

        case _:
            raise ValueError(data.velocity_representation)

    # =====================================================
    # Compute quantities to adjust the output representation
    # =====================================================

    with data.switch_velocity_representation(VelRepr.Inertial):
        # Compute the Jacobian of the parent link in inertial representation.
        W_J_WL_W = js.model.generalized_free_floating_jacobian(
            model=model,
            data=data,
        )
        # Compute the Jacobian derivative of the parent link in inertial representation.
        W_J̇_WL_W = js.model.generalized_free_floating_jacobian_derivative(
            model=model,
            data=data,
        )

    def compute_O_J̇_WC_I(
        L_p_C: jtp.Vector,
        parent_link_idx: jtp.Int,
        W_H_L: jtp.Matrix,
    ) -> jtp.Matrix:

        match output_vel_repr:
            case VelRepr.Inertial:
                O_X_W = W_X_W = Adjoint.from_transform(  # noqa: F841
                    transform=jnp.eye(4)
                )
                O_Ẋ_W = W_Ẋ_W = jnp.zeros((6, 6))  # noqa: F841

            case VelRepr.Body:
                L_H_C = Transform.from_rotation_and_translation(translation=L_p_C)
                W_H_C = W_H_L[parent_link_idx] @ L_H_C
                O_X_W = C_X_W = Adjoint.from_transform(transform=W_H_C, inverse=True)
                W_v_WC = W_v_WLi[parent_link_idx]
                W_vx_WC = Cross.vx(W_v_WC)
                O_Ẋ_W = C_Ẋ_W = -C_X_W @ W_vx_WC  # noqa: F841

            case VelRepr.Mixed:
                L_H_C = Transform.from_rotation_and_translation(translation=L_p_C)
                W_H_C = W_H_L[parent_link_idx] @ L_H_C
                W_H_CW = W_H_C.at[0:3, 0:3].set(jnp.eye(3))
                CW_H_W = Transform.inverse(W_H_CW)
                O_X_W = CW_X_W = Adjoint.from_transform(transform=CW_H_W)
                CW_v_WC = CW_X_W @ W_v_WLi[parent_link_idx]
                W_v_W_CW = jnp.zeros(6).at[0:3].set(CW_v_WC[0:3])
                W_vx_W_CW = Cross.vx(W_v_W_CW)
                O_Ẋ_W = CW_Ẋ_W = -CW_X_W @ W_vx_W_CW  # noqa: F841

            case _:
                raise ValueError(output_vel_repr)

        O_J̇_WC_I = jnp.zeros(shape=(6, 6 + model.dofs()))
        O_J̇_WC_I += O_Ẋ_W @ W_J_WL_W[parent_link_idx] @ T
        O_J̇_WC_I += O_X_W @ W_J̇_WL_W[parent_link_idx] @ T
        O_J̇_WC_I += O_X_W @ W_J_WL_W[parent_link_idx] @ Ṫ

        return O_J̇_WC_I

    O_J̇_WC = jax.vmap(compute_O_J̇_WC_I, in_axes=(0, 0, None))(
        L_p_Ci, parent_link_idx_of_enabled_collidable_points, W_H_Li
    )

    return O_J̇_WC
