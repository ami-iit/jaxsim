import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.data import JaxSimModelData
from jaxsim.math import Quaternion, Skew

from .common import VelRepr

# ==================================
# Functions defining system dynamics
# ==================================


def system_acceleration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_torques: jtp.VectorLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the system acceleration in the active representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the same
            velocity representation of data.
        joint_torques: The joint torques applied to the joints.

    Returns:
        A tuple containing the base 6D acceleration in the active representation
        and the joint accelerations.
    """

    # ====================
    # Validate input data
    # ====================

    # Build link forces if not provided.
    f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # Store the link forces in a references object.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        velocity_representation=data.velocity_representation,
        link_forces=f_L,
    )

    # Compute forward dynamics.
    #
    # - Joint accelerations: s̈ ∈ ℝⁿ
    # - Base acceleration: v̇_WB ∈ ℝ⁶
    #
    # Note that ABA returns the base acceleration in the velocity representation
    # stored in the `data` object.
    v̇_WB, s̈ = js.model.forward_dynamics_aba(
        model=model,
        data=data,
        joint_forces=joint_torques,
        link_forces=references.link_forces(model=model, data=data),
    )

    return v̇_WB, s̈


@jax.jit
@js.common.named_scope
def system_position_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    r"""
    Compute the dynamics of the system position.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient for adjusting the quaternion norm.

    Returns:
        A tuple containing the derivative of the base position, the derivative of the
        base quaternion, and the derivative of the joint positions.

    Note:
        In inertial-fixed representation, the linear component of the base velocity is not
        the derivative of the base position. In fact, the base velocity is defined as:
        :math:`{} ^W v_{W, B} = \begin{bmatrix} {} ^W \dot{p}_B S({} ^W \omega_{W, B}) {} ^W p _B\\ {} ^W \omega_{W, B} \end{bmatrix}`.
        Where :math:`S(\cdot)` is the skew-symmetric matrix operator.
    """

    ṡ = data.joint_velocities
    W_Q_B = data.base_orientation
    W_ω_WB = data.base_velocity[3:6]
    W_ṗ_B = data.base_velocity[0:3] + Skew.wedge(W_ω_WB) @ data.base_position

    W_Q̇_B = Quaternion.derivative(
        quaternion=W_Q_B,
        omega=W_ω_WB,
        omega_in_body_fixed=False,
        K=baumgarte_quaternion_regularization,
    ).squeeze()

    return W_ṗ_B, W_Q̇_B, ṡ


@jax.jit
@js.common.named_scope
def system_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.Vector | None = None,
    joint_torques: jtp.Vector | None = None,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> JaxSimModelData:
    """
    Compute the dynamics of the system.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        joint_torques: The joint torques acting on the joints.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient used to adjust the norm of the
            quaternion (only used in integrators not operating on the SO(3) manifold).

    Returns:
        A tuple with an `JaxSimModelData` object storing in each of its attributes the
        corresponding derivative, and the dictionary of auxiliary data returned
        by the system dynamics evaluation.
    """

    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        W_v̇_WB, s̈ = system_acceleration(
            model=model,
            data=data,
            joint_torques=joint_torques,
            link_forces=link_forces,
        )

        W_ṗ_B, W_Q̇_B, ṡ = system_position_dynamics(
            model=model,
            data=data,
            baumgarte_quaternion_regularization=baumgarte_quaternion_regularization,
        )

    ode_state_derivative = JaxSimModelData.build(
        model=model,
        base_position=W_ṗ_B,
        base_quaternion=W_Q̇_B,
        joint_positions=ṡ,
        base_linear_velocity=W_v̇_WB[0:3],
        base_angular_velocity=W_v̇_WB[3:6],
        joint_velocities=s̈,
    )

    return ode_state_derivative
