from __future__ import annotations

import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import VelRepr
from jaxsim.api.kin_dyn_parameters import ConstraintMap
from jaxsim.math.rotation import Rotation


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
        output_vel_repr=VelRepr.Inertial,
    )
    J_WF2 = js.frame.jacobian(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_2,
        output_vel_repr=VelRepr.Inertial,
    )

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
        output_vel_repr=VelRepr.Inertial,
    )
    J̇_WF2 = js.frame.jacobian_derivative(
        model=model,
        data=data,
        frame_index=constraint.frame_idxs_2,
        output_vel_repr=VelRepr.Inertial,
    )

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

    W_R_F1 = W_H_F1[0:3, 0:3]
    W_R_F2 = W_H_F2[0:3, 0:3]

    K_P = constraint.K_P
    K_D = constraint.K_D

    vel_error = J_constr @ nu
    position_error = W_p_F1 - W_p_F2
    R_error = W_R_F2.T @ W_R_F1
    orientation_error = Rotation.log_vee(R_error)

    baumgarte_term = (
        K_P * jnp.concatenate([position_error, orientation_error]) + K_D * vel_error
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
