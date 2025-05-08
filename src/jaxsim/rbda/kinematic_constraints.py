from __future__ import annotations

import dataclasses
from math import sqrt
from typing import ClassVar

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import VelRepr
from jaxsim.math.rotation import Rotation
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass


@dataclasses.dataclass(frozen=True)
class ConstraintType:
    """
    Enumeration of all supported constraint types.
    """

    Weld: ClassVar[int] = 0
    # TODO: handle Connect constraint
    # Connect: ClassVar[int] = 1


@jax_dataclasses.pytree_dataclass
class ConstraintMap(JaxsimDataclass):
    """
    Class storing the kinematic constraints of a model.
    """

    frame_idxs_1: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    frame_idxs_2: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    constraint_types: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    K_P: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=float)
    )
    K_D: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=float)
    )

    def add_constraint(
        self,
        frame_idx_1: int,
        frame_idx_2: int,
        constraint_type: int,
        K_P: float | None = None,
        K_D: float | None = None,
    ) -> ConstraintMap:
        """
        Add a constraint to the constraint map.

        Args:
            frame_name_1: The name of the first frame.
            frame_name_2: The name of the second frame.
            constraint_type: The type of constraint.
            K_P: The proportional gain for Baumgarte stabilization (default: 1000).
            K_D: The derivative gain for Baumgarte stabilization (default: 2 * sqrt(K_P)).

        Returns:
            A new ConstraintMap instance with the added constraint.
        """

        # Set default values for Baumgarte coefficients if not provided
        if K_P is None:
            K_P = 1000
        if K_D is None:
            K_D = 2 * sqrt(K_P)

        # Create new arrays with the input elements appended
        new_frame_idxs_1 = jnp.append(self.frame_idxs_1, frame_idx_1)
        new_frame_idxs2 = jnp.append(self.frame_idxs_2, frame_idx_2)
        new_constraint_types = jnp.append(self.constraint_types, constraint_type)
        new_K_P = jnp.append(self.K_P, K_P)
        new_K_D = jnp.append(self.K_D, K_D)

        # Return a new ConstraintMap object with updated attributes
        return ConstraintMap(
            frame_idxs_1=new_frame_idxs_1,
            frame_idxs_2=new_frame_idxs2,
            constraint_types=new_constraint_types,
            K_P=new_K_P,
            K_D=new_K_D,
        )


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
        between the two specified frames.
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
        between the two specified frames.
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
        K_P * jnp.concatenate([position_error, orientation_error])
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
