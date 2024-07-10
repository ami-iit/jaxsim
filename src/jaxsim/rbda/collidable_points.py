import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Skew

from . import utils


def collidable_points_pos_vel(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.Vector,
    base_quaternion: jtp.Vector,
    joint_positions: jtp.Vector,
    base_linear_velocity: jtp.Vector,
    base_angular_velocity: jtp.Vector,
    joint_velocities: jtp.Vector,
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """

    Compute the position and linear velocity of collidable points in the world frame.

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.
        base_linear_velocity:
            The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in inertial-fixed representation.
        joint_velocities: The velocities of the joints.

    Returns:
        A tuple containing the position and linear velocity of collidable points.
    """

    if len(model.kin_dyn_parameters.contact_parameters.body) == 0:
        return jnp.array(0).astype(float), jnp.empty(0).astype(float)

    W_p_B, W_Q_B, s, W_v_WB, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        joint_velocities=joint_velocities,
    )

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the base transform.
    W_H_B = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=W_Q_B),
        translation=W_p_B,
    )

    # Compute the parent-to-child adjoints and the motion subspaces of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi, S = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=s, base_transform=W_H_B.as_matrix()
    )

    # Allocate buffer of transforms world -> link and initialize the base pose.
    W_X_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    W_X_i = W_X_i.at[0].set(Adjoint.inverse(i_X_λi[0]))

    # Allocate buffer of 6D inertial-fixed velocities and initialize the base velocity.
    W_v_Wi = jnp.zeros(shape=(model.number_of_links(), 6))
    W_v_Wi = W_v_Wi.at[0].set(W_v_WB)

    # ====================
    # Propagate kinematics
    # ====================

    PropagateTransformsCarry = tuple[jtp.Matrix, jtp.Matrix]
    propagate_transforms_carry: PropagateTransformsCarry = (W_X_i, W_v_Wi)

    def propagate_kinematics(
        carry: PropagateTransformsCarry, i: jtp.Int
    ) -> tuple[PropagateTransformsCarry, None]:

        ii = i - 1
        W_X_i, W_v_Wi = carry

        # Compute the parent to child 6D transform.
        λi_X_i = Adjoint.inverse(adjoint=i_X_λi[i])

        # Compute the world to child 6D transform.
        W_Xi_i = W_X_i[λ[i]] @ λi_X_i
        W_X_i = W_X_i.at[i].set(W_Xi_i)

        # Propagate the 6D velocity.
        W_vi_Wi = W_v_Wi[λ[i]] + W_X_i[i] @ (S[i] * ṡ[ii]).squeeze()
        W_v_Wi = W_v_Wi.at[i].set(W_vi_Wi)

        return (W_X_i, W_v_Wi), None

    (W_X_i, W_v_Wi), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=propagate_transforms_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(W_X_i, W_v_Wi), None]
    )

    # ==================================================
    # Compute position and velocity of collidable points
    # ==================================================

    def process_point_kinematics(
        Li_p_C: jtp.Vector, parent_body: jtp.Int
    ) -> tuple[jtp.Vector, jtp.Vector]:

        # Compute the position of the collidable point.
        W_p_Ci = (
            Adjoint.to_transform(adjoint=W_X_i[parent_body]) @ jnp.hstack([Li_p_C, 1])
        )[0:3]

        # Compute the linear part of the mixed velocity Ci[W]_v_{W,Ci}.
        CW_vl_WCi = (
            jnp.block([jnp.eye(3), -Skew.wedge(vector=W_p_Ci).squeeze()])
            @ W_v_Wi[parent_body].squeeze()
        )

        return W_p_Ci, CW_vl_WCi

    # Process all the collidable points in parallel.
    W_p_Ci, CW_vl_WC = jax.vmap(process_point_kinematics)(
        model.kin_dyn_parameters.contact_parameters.point,
        jnp.array(model.kin_dyn_parameters.contact_parameters.body),
    )

    return W_p_Ci, CW_vl_WC
