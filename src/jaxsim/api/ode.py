from typing import Any

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Quaternion

from .common import VelRepr
from .ode_data import ODEState

# ==================================
# Functions defining system dynamics
# ==================================


@jax.jit
@js.common.named_scope
def system_velocity_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.Vector | None = None,
    joint_force_references: jtp.Vector | None = None,
) -> tuple[jtp.Vector, jtp.Vector, dict[str, Any]]:
    """
    Compute the dynamics of the system velocity.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        joint_force_references: The joint force references to apply.

    Returns:
        A tuple containing the derivative of the base 6D velocity in inertial-fixed
        representation, the derivative of the joint velocities, and auxiliary data
        returned by the system dynamics evaluation.
    """

    # Build link forces if not provided.
    # These forces are expressed in the frame corresponding to the velocity
    # representation of data.
    O_f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # We expect that the 6D forces included in the `link_forces` argument are expressed
    # in the frame corresponding to the velocity representation of `data`.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        link_forces=O_f_L,
        joint_force_references=joint_force_references,
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # ======================
    # Compute contact forces
    # ======================

    if len(model.kin_dyn_parameters.contact_parameters.body) > 0:
        with data.switch_velocity_representation(VelRepr.Inertial):

            # Compute the 6D forces W_f ∈ ℝ^{n_L × 6} applied to links due to contact
            # with the terrain.
            W_f_L_terrain = js.model.link_contact_forces(
                model=model,
                data=data,
                link_forces=link_forces,
                joint_force_references=joint_force_references,
            )

    # ===========================
    # Compute system acceleration
    # ===========================

    # Compute the total link forces.
    with (
        data.switch_velocity_representation(VelRepr.Inertial),
        references.switch_velocity_representation(VelRepr.Inertial),
    ):

        # Sum the contact forces just computed with the link forces applied by the user.
        references = references.apply_link_forces(
            model=model,
            data=data,
            forces=W_f_L_terrain,
            additive=True,
        )

        # Get the link forces in inertial-fixed representation.
        f_L_total = references.link_forces(model=model, data=data)

        # Compute the system acceleration in inertial-fixed representation.
        # This representation is useful for integration purpose.
        W_v̇_WB, s̈ = system_acceleration(
            model=model,
            data=data,
            joint_force_references=joint_force_references,
            link_forces=f_L_total,
        )

    return W_v̇_WB, s̈


def system_acceleration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_force_references: jtp.VectorLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the system acceleration in the active representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the same
            velocity representation of data.
        joint_force_references: The joint force references to apply.

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

    # Build joint torques if not provided.
    τ_references = (
        jnp.atleast_1d(joint_force_references.squeeze())
        if joint_force_references is not None
        else jnp.zeros_like(data.joint_positions())
    ).astype(float)

    # ====================
    # Enforce joint limits
    # ====================

    τ_position_limit = jnp.zeros_like(τ_references).astype(float)

    if model.dofs() > 0:

        # Stiffness and damper parameters for the joint position limits.
        k_j = jnp.array(
            model.kin_dyn_parameters.joint_parameters.position_limit_spring
        ).astype(float)
        d_j = jnp.array(
            model.kin_dyn_parameters.joint_parameters.position_limit_damper
        ).astype(float)

        # Compute the joint position limit violations.
        lower_violation = jnp.clip(
            data.state.physics_model.joint_positions
            - model.kin_dyn_parameters.joint_parameters.position_limits_min,
            max=0.0,
        )

        upper_violation = jnp.clip(
            data.state.physics_model.joint_positions
            - model.kin_dyn_parameters.joint_parameters.position_limits_max,
            min=0.0,
        )

        # Compute the joint position limit torque.
        τ_position_limit -= jnp.diag(k_j) @ (lower_violation + upper_violation)

        τ_position_limit -= (
            jnp.positive(τ_position_limit)
            * jnp.diag(d_j)
            @ data.state.physics_model.joint_velocities
        )

    # ====================
    # Joint friction model
    # ====================

    τ_friction = jnp.zeros_like(τ_references).astype(float)

    if model.dofs() > 0:

        # Static and viscous joint friction parameters
        kc = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_static
        ).astype(float)
        kv = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_viscous
        ).astype(float)

        # Compute the joint friction torque.
        τ_friction = -(
            jnp.diag(kc) @ jnp.sign(data.state.physics_model.joint_velocities)
            + jnp.diag(kv) @ data.state.physics_model.joint_velocities
        )

    # ========================
    # Compute forward dynamics
    # ========================

    # Compute the total joint forces.
    τ_total = τ_references + τ_friction + τ_position_limit

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
        joint_forces=τ_total,
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
    """
    Compute the dynamics of the system position.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient for adjusting the quaternion norm.

    Returns:
        A tuple containing the derivative of the base position, the derivative of the
        base quaternion, and the derivative of the joint positions.
    """

    ṡ = data.joint_velocities(model=model)
    W_Q_B = data.base_orientation(dcm=False)

    with data.switch_velocity_representation(velocity_representation=VelRepr.Mixed):
        W_ṗ_B = data.base_velocity()[0:3]

    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        W_ω_WB = data.base_velocity()[3:6]

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
    joint_force_references: jtp.Vector | None = None,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> ODEState:
    """
    Compute the dynamics of the system.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        joint_force_references: The joint force references to apply.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient used to adjust the norm of the
            quaternion (only used in integrators not operating on the SO(3) manifold).

    Returns:
        A tuple with an `ODEState` object storing in each of its attributes the
        corresponding derivative, and the dictionary of auxiliary data returned
        by the system dynamics evaluation.
    """

    # Compute the accelerations and the material deformation rate.
    W_v̇_WB, s̈ = system_velocity_dynamics(
        model=model,
        data=data,
        joint_force_references=joint_force_references,
        link_forces=link_forces,
    )

    # Extract the velocities.
    W_ṗ_B, W_Q̇_B, ṡ = system_position_dynamics(
        model=model,
        data=data,
        baumgarte_quaternion_regularization=baumgarte_quaternion_regularization,
    )

    # Create an ODEState object populated with the derivative of each leaf.
    # Our integrators, operating on generic pytrees, will be able to handle it
    # automatically as state derivative.
    ode_state_derivative = ODEState.build_from_jaxsim_model(
        model=model,
        base_position=W_ṗ_B,
        base_quaternion=W_Q̇_B,
        joint_positions=ṡ,
        base_linear_velocity=W_v̇_WB[0:3],
        base_angular_velocity=W_v̇_WB[3:6],
        joint_velocities=s̈,
    )

    return ode_state_derivative
