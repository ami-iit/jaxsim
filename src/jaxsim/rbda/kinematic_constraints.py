from __future__ import annotations

import dataclasses
import enum

import jax
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import VelRepr
from jaxsim.math.rotation import Rotation
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass


@enum.unique
class ConstraintType(enum.IntEnum):
    """
    Enumeration of all supported constraint types.
    """

    Weld = enum.auto()
    Connect = enum.auto()


@jax_dataclasses.pytree_dataclass
class ConstraintMap(JaxsimDataclass):
    """
    Class storing the kinematic constraints of a model.
    """

    frame_names_1: Static[tuple[str, ...]] = dataclasses.field(default_factory=tuple)
    frame_names_2: Static[tuple[str, ...]] = dataclasses.field(default_factory=tuple)
    constraint_types: Static[tuple[ConstraintType, ...]] = dataclasses.field(
        default_factory=tuple
    )

    def add_constraint(
        self, frame_name_1: str, frame_name_2: str, constraint_type: ConstraintType
    ) -> ConstraintMap:
        """
        Add a constraint to the constraint map.

        Args:
            frame_name_1: The name of the first frame.
            frame_name_2: The name of the second frame.
            constraint_type: The type of constraint.

        Returns:
            A new ConstraintMap instance with the added constraint.
        """
        return self.replace(
            frame_names_1=(*self.frame_names_1, frame_name_1),
            frame_names_2=(*self.frame_names_2, frame_name_2),
            constraint_types=(*self.constraint_types, constraint_type),
            validate=False,
        )

    def get_constraints(
        self, model: js.model.JaxSimModel
    ) -> tuple[tuple[int, int, ConstraintType], ...]:
        """
        Get the list of constraints.

        Returns:
            A tuple, in which each element defines a kinematic constraint.
        """
        return jnp.array(
            (
                jax.tree.map(
                    lambda f1: js.frame.name_to_idx(model, frame_name=f1),
                    self.frame_names_1,
                ),
                jax.tree.map(
                    lambda f1: js.frame.name_to_idx(model, frame_name=f1),
                    self.frame_names_2,
                ),
            )
        ).T


def compute_constraint_jacobians(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraint: tuple[int, int],
) -> jtp.Matrix:
    """
    Compute the constraint Jacobian matrix representing the kinematic constraints between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: A tuple containing the indices of the two frames
            (frame_1_idx, frame_2_idx) for which the constraint Jacobian is computed.

    Returns:
        The resulting constraint Jacobian matrix representing the kinematic constraint
        between the two specified frames.
    """
    frame_1_idx, frame_2_idx = constraint

    J_WF1 = js.frame.jacobian(
        model=model,
        data=data,
        frame_index=frame_1_idx,
        output_vel_repr=VelRepr.Inertial,
    )
    J_WF2 = js.frame.jacobian(
        model=model,
        data=data,
        frame_index=frame_2_idx,
        output_vel_repr=VelRepr.Inertial,
    )

    return J_WF1 - J_WF2


def compute_constraint_jacobians_derivative(model: js.model.JaxSimModel, data: js.data.JaxSimModelData, constraint: tuple[int, int]) -> jtp.Matrix:
    """
    Compute the derivative of the constraint Jacobian matrix representing the kinematic constraints between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: A tuple containing the indices of the two frames
            (frame_1_idx, frame_2_idx) for which the constraint Jacobian derivative is computed.

    Returns:
        The resulting constraint Jacobian derivative matrix representing the kinematic constraint
        between the two specified frames.
    """
    frame_1_idx, frame_2_idx = constraint

    J̇_WF1 = js.frame.jacobian_derivative(
        model=model,
        data=data,
        frame_index=frame_1_idx,
        output_vel_repr=VelRepr.Inertial,
    )
    J̇_WF2 = js.frame.jacobian_derivative(
        model=model,
        data=data,
        frame_index=frame_2_idx,
        output_vel_repr=VelRepr.Inertial,
    )

    return J̇_WF1 - J̇_WF2


def compute_constraint_baumgarte_term(J_constr: jtp.Matrix, nu: jtp.Vector, W_H_F_constr: tuple[jtp.Matrix, jtp.Matrix], K_P: float, K_D: float) -> jtp.Vector:
    """
    Compute the Baumgarte stabilization term for kinematic constraints.

    Args:
        J_constr: The constraint Jacobian matrix.
        nu: The generalized velocity vector.
        W_H_F_constr: A tuple containing the homogeneous transformation matrices
            of two frames (W_H_F1 and W_H_F2) with respect to the world frame.
        K_P: The proportional gain for position and orientation error correction.
        K_D: The derivative gain for velocity error correction.

    Returns:
        The computed Baumgarte stabilization term.
    """
    W_H_F1, W_H_F2 = W_H_F_constr

    W_p_F1 = W_H_F1[0:3, 3]
    W_p_F2 = W_H_F2[0:3, 3]

    W_R_F1 = W_H_F1[0:3, 0:3]
    W_R_F2 = W_H_F2[0:3, 0:3]

    vel_error = J_constr @ nu
    position_error = W_p_F1 - W_p_F2
    R_error = W_R_F2.T @ W_R_F1
    orientation_error = Rotation.log_vee(R_error)

    # jax.debug.print(
    #     "Position error: {}\nOrientation error: {}\nVelocity error: {}",
    #     position_error,
    #     orientation_error,
    #     vel_error,
    # )
    baumgarte_term = (
        K_P * jnp.concatenate([position_error, orientation_error])
        + K_D * vel_error
    )

    return baumgarte_term


def compute_constraint_transforms(model: js.model.JaxSimModel, data: js.data.JaxSimModelData, constraint: tuple[int, int]) -> jtp.Matrix:
    """
    Compute the transformation matrices for a given kinematic constraint between two frames.

    Args:
        model: The JaxSim model.
        data: The data of the considered model.
        constraint: A tuple containing the indices of the two frames
            (frame_1_idx, frame_2_idx) for which the transformation matrices are computed.

    Returns:
        A matrix containing the tuple of transformation matrices of the two frames.
    """
    frame_1_idx, frame_2_idx = constraint

    W_H_F1 = js.frame.transform(model=model, data=data, frame_index=frame_1_idx)
    W_H_F2 = js.frame.transform(model=model, data=data, frame_index=frame_2_idx)

    return jnp.array((W_H_F1, W_H_F2))
