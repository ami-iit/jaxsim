import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


def compute_resultant_torques(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_force_references: jtp.Vector | None = None,
) -> jtp.Vector:
    """
    Compute the resultant torques acting on the joints.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_force_references: The joint force references to apply.

    Returns:
        The resultant torques acting on the joints.
    """

    # Build joint torques if not provided.
    τ_references = (
        jnp.atleast_1d(joint_force_references.squeeze())
        if joint_force_references is not None
        else jnp.zeros_like(data.joint_positions)
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
            data.joint_positions
            - model.kin_dyn_parameters.joint_parameters.position_limits_min,
            max=0.0,
        )

        upper_violation = jnp.clip(
            data.joint_positions
            - model.kin_dyn_parameters.joint_parameters.position_limits_max,
            min=0.0,
        )

        # Compute the joint position limit torque.
        τ_position_limit -= jnp.diag(k_j) @ (lower_violation + upper_violation)

        τ_position_limit -= (
            jnp.positive(τ_position_limit) * jnp.diag(d_j) @ data.joint_velocities
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
            jnp.diag(kc) @ jnp.sign(data.joint_velocities)
            + jnp.diag(kv) @ data.joint_velocities
        )

    # ===============================
    # Compute the total joint forces.
    # ===============================

    τ_total = τ_references + τ_friction + τ_position_limit

    return τ_total
