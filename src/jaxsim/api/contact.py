from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Cross, Transform
from jaxsim.rbda.contacts.soft import SoftContactsParams

from .common import VelRepr


@jax.jit
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

    from jaxsim.rbda import collidable_points

    # Switch to inertial-fixed since the RBDAs expect velocities in this representation.
    with data.switch_velocity_representation(VelRepr.Inertial):
        W_p_Ci, W_ṗ_Ci = collidable_points.collidable_points_pos_vel(
            model=model,
            base_position=data.base_position(),
            base_quaternion=data.base_orientation(dcm=False),
            joint_positions=data.joint_positions(model=model),
            base_linear_velocity=data.base_velocity()[0:3],
            base_angular_velocity=data.base_velocity()[3:6],
            joint_velocities=data.joint_velocities(model=model),
        )

    return W_p_Ci, W_ṗ_Ci


@jax.jit
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


@jax.jit
def collidable_point_forces(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the 6D forces applied to each collidable point.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The 6D forces applied to each collidable point expressed in the frame
        corresponding to the active representation.
    """

    f_Ci, _ = collidable_point_dynamics(model=model, data=data)

    return f_Ci


@jax.jit
def collidable_point_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Matrix, dict[str, jtp.Array]]:
    r"""
    Compute the 6D force applied to each collidable point.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D external forces to apply to the links expressed in the same
            representation of data.

    Returns:
        The 6D force applied to each collidable point and additional data based on the contact model configured:
        - Soft: the material deformation rate.
        - Rigid: no additional data.
        - QuasiRigid: no additional data.

    Note:
        The material deformation rate is always returned in the mixed frame
        `C[W] = ({}^W \mathbf{p}_C, [W])`. This is convenient for integration purpose.
        Instead, the 6D forces are returned in the active representation.
    """

    # Compute the position and linear velocities (mixed representation) of
    # all collidable points belonging to the robot.
    W_p_Ci, W_ṗ_Ci = js.contact.collidable_point_kinematics(model=model, data=data)

    # Import privately the contacts classes.
    from jaxsim.rbda.contacts.relaxed_rigid import (
        RelaxedRigidContacts,
        RelaxedRigidContactsState,
    )
    from jaxsim.rbda.contacts.rigid import RigidContacts, RigidContactsState
    from jaxsim.rbda.contacts.soft import SoftContacts, SoftContactsState

    # Build the soft contact model.
    match model.contact_model:

        case SoftContacts():

            assert isinstance(model.contact_model, SoftContacts)
            assert isinstance(data.state.contact, SoftContactsState)

            # Build the contact model.
            soft_contacts = SoftContacts(
                parameters=data.contacts_params, terrain=model.terrain
            )

            # Compute the 6D force expressed in the inertial frame and applied to each
            # collidable point, and the corresponding material deformation rate.
            # Note that the material deformation rate is always returned in the mixed frame
            # C[W] = (W_p_C, [W]). This is convenient for integration purpose.
            W_f_Ci, (CW_ṁ,) = jax.vmap(soft_contacts.compute_contact_forces)(
                W_p_Ci, W_ṗ_Ci, data.state.contact.tangential_deformation
            )
            aux_data = dict(m_dot=CW_ṁ)

        case RigidContacts():
            assert isinstance(model.contact_model, RigidContacts)
            assert isinstance(data.state.contact, RigidContactsState)

            # Build the contact model.
            rigid_contacts = RigidContacts(
                parameters=data.contacts_params, terrain=model.terrain
            )

            # Compute the 6D force expressed in the inertial frame and applied to each
            # collidable point.
            W_f_Ci, _ = rigid_contacts.compute_contact_forces(
                position=W_p_Ci,
                velocity=W_ṗ_Ci,
                model=model,
                data=data,
                link_forces=link_forces,
            )

            aux_data = dict()

        case RelaxedRigidContacts():
            assert isinstance(model.contact_model, RelaxedRigidContacts)
            assert isinstance(data.state.contact, RelaxedRigidContactsState)

            # Build the contact model.
            relaxed_rigid_contacts = RelaxedRigidContacts(
                parameters=data.contacts_params, terrain=model.terrain
            )

            # Compute the 6D force expressed in the inertial frame and applied to each
            # collidable point.
            W_f_Ci, _ = relaxed_rigid_contacts.compute_contact_forces(
                position=W_p_Ci,
                velocity=W_ṗ_Ci,
                model=model,
                data=data,
                link_forces=link_forces,
            )

            aux_data = dict()

        case _:
            raise ValueError(f"Invalid contact model {model.contact_model}")

    # Convert the 6D forces to the active representation.
    f_Ci = jax.vmap(
        lambda W_f_C: data.inertial_to_other_representation(
            array=W_f_C,
            other_representation=data.velocity_representation,
            transform=data.base_transform(),
            is_force=True,
        )
    )(W_f_Ci)

    return f_Ci, aux_data


@functools.partial(jax.jit, static_argnames=["link_names"])
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

    link_names = link_names if link_names is not None else model.link_names()

    if set(link_names).difference(model.link_names()):
        raise ValueError("One or more link names are not part of the model")

    W_p_Ci = collidable_point_positions(model=model, data=data)

    terrain_height = jax.vmap(lambda x, y: model.terrain.height(x=x, y=y))(
        W_p_Ci[:, 0], W_p_Ci[:, 1]
    )

    below_terrain = W_p_Ci[:, 2] <= terrain_height

    links_in_contact = jax.vmap(
        lambda link_index: jnp.where(
            jnp.array(model.kin_dyn_parameters.contact_parameters.body) == link_index,
            below_terrain,
            jnp.zeros_like(below_terrain, dtype=bool),
        ).any()
    )(js.link.names_to_idxs(link_names=link_names, model=model))

    return links_in_contact


@jax.jit
def estimate_good_soft_contacts_parameters(
    model: js.model.JaxSimModel,
    *,
    standard_gravity: jtp.FloatLike = jaxsim.math.StandardGravity,
    static_friction_coefficient: jtp.FloatLike = 0.5,
    number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
    damping_ratio: jtp.FloatLike = 1.0,
    max_penetration: jtp.FloatLike | None = None,
) -> SoftContactsParams:
    """
    Estimate good soft contacts parameters for the given model.

    Args:
        model: The model to consider.
        standard_gravity: The standard gravity constant.
        static_friction_coefficient: The static friction coefficient.
        number_of_active_collidable_points_steady_state:
            The number of active collidable points in steady state supporting
            the weight of the robot.
        damping_ratio: The damping ratio.
        max_penetration:
            The maximum penetration allowed in steady state when the robot is
            supported by the configured number of active collidable points.

    Returns:
        The estimated good soft contacts parameters.

    Note:
        This method provides a good starting point for the soft contacts parameters.
        The user is encouraged to fine-tune the parameters based on the
        specific application.
    """
    from jaxsim.rbda.contacts.soft import SoftContactsParams

    def estimate_model_height(model: js.model.JaxSimModel) -> jtp.Float:
        """"""

        zero_data = js.data.JaxSimModelData.build(
            model=model,
            contacts_params=SoftContactsParams(),
        )

        W_pz_CoM = js.com.com_position(model=model, data=zero_data)[2]

        if model.floating_base():
            W_pz_C = collidable_point_positions(model=model, data=zero_data)[:, -1]
            return 2 * (W_pz_CoM - W_pz_C.min())

        return 2 * W_pz_CoM

    max_δ = (
        max_penetration
        if max_penetration is not None
        else 0.005 * estimate_model_height(model=model)
    )

    nc = number_of_active_collidable_points_steady_state

    sc_parameters = SoftContactsParams.build_default_from_jaxsim_model(
        model=model,
        standard_gravity=standard_gravity,
        static_friction_coefficient=static_friction_coefficient,
        max_penetration=max_δ,
        number_of_active_collidable_points_steady_state=nc,
        damping_ratio=damping_ratio,
    )

    return sc_parameters


@jax.jit
def transforms(model: js.model.JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Array:
    r"""
    Return the pose of the collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The stacked SE(3) matrices of all collidable points.

    Note:
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    # Get the transforms of the parent link of all collidable points.
    W_H_L = jax.vmap(
        lambda parent_link_idx: js.link.transform(
            model=model, data=data, link_index=parent_link_idx
        )
    )(jnp.array(model.kin_dyn_parameters.contact_parameters.body, dtype=int))

    # Build the link-to-point transform from the displacement between the link frame L
    # and the implicit contact frame C.
    L_H_C = jax.vmap(lambda L_p_C: jnp.eye(4).at[0:3, 3].set(L_p_C))(
        model.kin_dyn_parameters.contact_parameters.point
    )

    # Compose the work-to-link and link-to-point transforms.
    return jax.vmap(lambda W_H_Li, L_H_Ci: W_H_Li @ L_H_Ci)(W_H_L, L_H_C)


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Array:
    r"""
    Return the free-floating Jacobian of the collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The stacked :math:`6 \times (6+n)` free-floating jacobians of the frames associated to the
        collidable points.

    Note:
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Compute the Jacobians of all links.
    W_J_WL = js.model.generalized_free_floating_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Inertial
    )

    # Compute the contact Jacobian.
    # In inertial-fixed output representation, the Jacobian of the parent link is also
    # the Jacobian of the frame C implicitly associated with the collidable point.
    W_J_WC = jax.vmap(lambda parent_link_idx: W_J_WL[parent_link_idx])(
        jnp.array(model.kin_dyn_parameters.contact_parameters.body, dtype=int)
    )

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
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the contact points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivative.

    Returns:
        The derivative of the :math:`6 \times (6+n)` free-floating jacobian of the contact points.

    Note:
        The input representation of the free-floating jacobian derivative is the active
        velocity representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the index of the parent link and the position of the collidable point.
    parent_link_idxs = jnp.array(model.kin_dyn_parameters.contact_parameters.body)
    L_p_Ci = jnp.array(model.kin_dyn_parameters.contact_parameters.point)
    contact_idxs = jnp.arange(L_p_Ci.shape[0])

    # Get the transforms of all the parent links.
    W_H_Li = js.model.forward_kinematics(model=model, data=data)

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
            W_H_B = data.base_transform()
            W_X_B = Adjoint.from_transform(transform=W_H_B)
            B_v_WB = data.base_velocity()
            B_vx_WB = Cross.vx(B_v_WB)
            W_Ẋ_B = W_X_B @ B_vx_WB

            T = compute_T(model=model, X=W_X_B)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_B)

        case VelRepr.Mixed:
            W_H_B = data.base_transform()
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = Adjoint.from_transform(transform=W_H_BW)
            BW_v_WB = data.base_velocity()
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
            output_vel_repr=VelRepr.Inertial,
        )
        # Compute the Jacobian derivative of the parent link in inertial representation.
        W_J̇_WL_W = js.model.generalized_free_floating_jacobian_derivative(
            model=model,
            data=data,
            output_vel_repr=VelRepr.Inertial,
        )

    # Get the Jacobian of the collidable points in the mixed representation.
    with data.switch_velocity_representation(VelRepr.Mixed):
        CW_J_WC_BW = jacobian(
            model=model,
            data=data,
            output_vel_repr=VelRepr.Mixed,
        )

    def compute_O_J̇_WC_I(
        L_p_C: jtp.Vector,
        contact_idx: jtp.Int,
        CW_J_WC_BW: jtp.Matrix,
        W_H_L: jtp.Matrix,
    ) -> jtp.Matrix:

        parent_link_idx = parent_link_idxs[contact_idx]

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
                with data.switch_velocity_representation(VelRepr.Inertial):
                    W_nu = data.generalized_velocity()
                W_v_WC = W_J_WL_W[parent_link_idx] @ W_nu
                W_vx_WC = Cross.vx(W_v_WC)
                O_Ẋ_W = C_Ẋ_W = -C_X_W @ W_vx_WC  # noqa: F841

            case VelRepr.Mixed:
                L_H_C = Transform.from_rotation_and_translation(translation=L_p_C)
                W_H_C = W_H_L[parent_link_idx] @ L_H_C
                W_H_CW = W_H_C.at[0:3, 0:3].set(jnp.eye(3))
                CW_H_W = Transform.inverse(W_H_CW)
                O_X_W = CW_X_W = Adjoint.from_transform(transform=CW_H_W)
                with data.switch_velocity_representation(VelRepr.Mixed):
                    CW_v_WC = CW_J_WC_BW @ data.generalized_velocity()
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

    O_J̇_WC = jax.vmap(compute_O_J̇_WC_I, in_axes=(0, 0, 0, None))(
        L_p_Ci, contact_idxs, CW_J_WC_BW, W_H_Li
    )

    return O_J̇_WC
