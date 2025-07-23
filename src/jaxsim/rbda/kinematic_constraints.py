from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr
from jaxsim.api.kin_dyn_parameters import ConstraintMap


def compute_constraint_jacobians(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraint: ConstraintMap,
) -> jtp.Matrix:
    """
    Compute the constraint Jacobian matrix representing the kinematic constraints between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: The considered constraint.

    Returns:
        The resulting constraint Jacobian matrix representing the kinematic constraint
        between the two specified frames, in inertial representation.
    """

    J_WF1 = js.frame.jacobian(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_1,
        # output_vel_repr=VelRepr.Inertial,
    )[:3]
    J_WF2 = js.frame.jacobian(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_2,
        # output_vel_repr=VelRepr.Inertial,
    )[:3]

    return J_WF1 - J_WF2


def compute_constraint_jacobians_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraint: ConstraintMap,
) -> jtp.Matrix:
    """
    Compute the derivative of the constraint Jacobian matrix representing the kinematic constraints between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: The considered constraint.

    Returns:
        The resulting constraint Jacobian derivative matrix representing the kinematic constraint
        between the two specified frames, in inertial representation.
    """

    J̇_WF1 = js.frame.jacobian_derivative(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_1,
        # output_vel_repr=VelRepr.Inertial,
    )[:3]
    J̇_WF2 = js.frame.jacobian_derivative(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_2,
        # output_vel_repr=VelRepr.Inertial,
    )[:3]

    return J̇_WF1 - J̇_WF2


def compute_constraint_baumgarte_term(
    J_constr: jtp.Matrix,
    nu: jtp.Vector,
    W_H_F_constr: tuple[jtp.Matrix, jtp.Matrix],
    constraint: ConstraintMap,
) -> jtp.Vector:
    """
    Compute the Baumgarte stabilization term for kinematic constraints.

    Args:
        J_constr: The constraint Jacobian matrix.
        nu: The generalized velocity vector.
        W_H_F_constr: A tuple containing the homogeneous transformation matrices
            of two frames (W_H_F1 and W_H_F2) with respect to the world frame.
        K_P: The proportional gain for position and orientation error correction.
        K_D: The derivative gain for velocity error correction.
        constraint: The considered constraint.

    Returns:
        The computed Baumgarte stabilization term.
    """
    W_H_F1, W_H_F2 = W_H_F_constr

    W_p_F1 = W_H_F1[0:3, 3]
    W_p_F2 = W_H_F2[0:3, 3]

    K_P = constraint.K_P
    K_D = constraint.K_D

    vel_error = J_constr @ nu
    position_error = W_p_F1 - W_p_F2
    # jax.debug.print(
    #     "Position error norm: {position_error}, Vel error norm: {vel_error}",
    #     position_error=jnp.linalg.norm(position_error),
    #     vel_error=jnp.linalg.norm(vel_error),
    # )
    # R_error = W_R_F2.T @ W_R_F1
    # orientation_error = Rotation.log_vee(R_error)

    baumgarte_term = (
        # K_P * jnp.concatenate([position_error, orientation_error]) + K_D * vel_error
        K_P * position_error
        + K_D * vel_error
    )

    return baumgarte_term


def compute_constraint_transforms(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraint: ConstraintMap,
) -> jtp.Matrix:
    """
    Compute the transformation matrices for a given kinematic constraint between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: The considered constraint.

    Returns:
        A matrix containing the tuple of transformation matrices of the two frames.
    """

    W_H_F1 = js.frame.transform(
        model=model, data=data, frame_index=constraint.frame_idxs_1
    )
    W_H_F2 = js.frame.transform(
        model=model, data=data, frame_index=constraint.frame_idxs_2
    )

    return jnp.array((W_H_F1, W_H_F2))


@jax.jit
@js.common.named_scope
def compute_constraint_wrenches(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_force_references: jtp.VectorLike | None = None,
    link_forces_inertial: jtp.MatrixLike | None = None,
    regularization: jtp.Float = 1e-3,
) -> jtp.Matrix:
    """
    Compute the constraint wrenches for kinematic constraints.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        joint_force_references: The joint force references to apply.
        link_forces_inertial: The link forces applied in inertial representation.
        regularization: The regularization parameter for the constraint solver.

    Returns:
        A tuple containing the constraint wrenches in inertial representation
        and auxiliary information.
    """

    # Retrieve the kinematic constraints, if any.
    kin_constraints = model.kin_dyn_parameters.constraints

    n_kin_constraints = (
        0
        if (kin_constraints is None) or (kin_constraints.frame_idxs_1.shape[0] == 0)
        else 3 * kin_constraints.frame_idxs_1.shape[0]
    )

    # Return empty results if no constraints exist
    if n_kin_constraints == 0:
        return jnp.zeros((0, 2, 6))

    # Build joint forces if not provided
    τ_references = (
        jnp.asarray(joint_force_references, dtype=float)
        if joint_force_references is not None
        else jnp.zeros_like(data.joint_positions)
    )

    # Build link forces if not provided
    W_f_L = (
        jnp.atleast_2d(jnp.array(link_forces_inertial).squeeze())
        if link_forces_inertial is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # Create references object for handling different velocity representations
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=τ_references,
        link_forces=W_f_L,
        velocity_representation=VelRepr.Inertial,
    )

    with (
        data.switch_velocity_representation(VelRepr.Mixed),
        references.switch_velocity_representation(VelRepr.Mixed),
    ):
        BW_ν = data.generalized_velocity

        # Compute free acceleration without constraints
        BW_ν̇_free = jnp.hstack(
            js.model.forward_dynamics_aba(
                model=model,
                data=data,
                link_forces=references.link_forces(model=model, data=data),
                joint_forces=references.joint_force_references(model=model),
            )
        )

        # Compute mass matrix
        M = js.model.free_floating_mass_matrix(model=model, data=data)

        # Compute constraint jacobians and derivatives
        J_constr = jax.vmap(compute_constraint_jacobians, in_axes=(None, None, 0))(
            model, data, kin_constraints
        )

        J̇_constr = jax.vmap(
            compute_constraint_jacobians_derivative, in_axes=(None, None, 0)
        )(model, data, kin_constraints)

        W_H_constr_pairs = jax.vmap(
            compute_constraint_transforms, in_axes=(None, None, 0)
        )(model, data, kin_constraints)

        # Compute Baumgarte stabilization term
        constr_baumgarte_term = jnp.ravel(
            jax.vmap(
                compute_constraint_baumgarte_term,
                in_axes=(0, None, 0, 0),
            )(
                J_constr,
                BW_ν,
                W_H_constr_pairs,
                kin_constraints,
            ),
        )

        # Stack constraint jacobians
        J_constr = jnp.vstack(J_constr)
        J̇_constr = jnp.vstack(J̇_constr)

        # Compute Delassus matrix for constraints
        G_constraints = J_constr @ jnp.linalg.solve(M, J_constr.T)

        # Compute constraint acceleration
        CW_al_free_constr = J_constr @ BW_ν̇_free + J̇_constr @ BW_ν

        # Setup constraint optimization problem
        constraint_regularization = regularization * jnp.ones(n_kin_constraints)
        R = jnp.diag(constraint_regularization)
        A = G_constraints + R
        b = CW_al_free_constr + constr_baumgarte_term

        # Solve for constraint forces
        kin_constr_force_mixed = jnp.linalg.solve(A, -b).reshape(-1, 3)

        # Convert constraint forces to wrenches

    def transform_wrench_pair_efficiently(force_3d, transform_pair):
        """Efficiently create and transform wrench pairs."""
        W_H_F1, W_H_F2 = transform_pair[0], transform_pair[1]

        # Create wrench pair directly
        wrench_F1 = jnp.concatenate([force_3d, jnp.zeros(3)])
        wrench_F2 = jnp.concatenate([-force_3d, jnp.zeros(3)])

        # Transform both at once
        wrench_F1_inertial = (
            ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=wrench_F1,
                transform=W_H_F1,
                other_representation=VelRepr.Mixed,
                is_force=True,
            )
        )
        wrench_F2_inertial = (
            ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=wrench_F2,
                transform=W_H_F2,
                other_representation=VelRepr.Mixed,
                is_force=True,
            )
        )

        return jnp.stack([wrench_F1_inertial, wrench_F2_inertial])

    kin_constr_wrench_pairs_inertial = jax.vmap(transform_wrench_pair_efficiently)(
        kin_constr_force_mixed, W_H_constr_pairs
    )

    return kin_constr_wrench_pairs_inertial
