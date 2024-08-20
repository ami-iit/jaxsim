import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint

from . import utils


def forward_kinematics_model(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
) -> jtp.Array:
    """
    Compute the forward kinematics.

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.

    Returns:
        A 3D array containing the SE(3) transforms of all links belonging to the model.
    """

    W_p_B, W_Q_B, s, _, _, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
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
    i_X_λi, _ = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=s, base_transform=W_H_B.as_matrix()
    )

    # Allocate the buffer of transforms world -> link and initialize the base pose.
    W_X_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    W_X_i = W_X_i.at[0].set(Adjoint.inverse(i_X_λi[0]))

    # ========================
    # Propagate the kinematics
    # ========================

    PropagateKinematicsCarry = tuple[jtp.Matrix]
    propagate_kinematics_carry: PropagateKinematicsCarry = (W_X_i,)

    def propagate_kinematics(
        carry: PropagateKinematicsCarry, i: jtp.Int
    ) -> tuple[PropagateKinematicsCarry, None]:

        (W_X_i,) = carry

        W_X_i_i = W_X_i[λ[i]] @ Adjoint.inverse(i_X_λi[i])
        W_X_i = W_X_i.at[i].set(W_X_i_i)

        return (W_X_i,), None

    (W_X_i,), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=propagate_kinematics_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(W_X_i,), None]
    )

    return jax.vmap(Adjoint.to_transform)(W_X_i)


def forward_kinematics(
    model: js.model.JaxSimModel,
    link_index: jtp.Int,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
) -> jtp.Matrix:
    """
    Compute the forward kinematics of a specific link.

    Args:
        model: The model to consider.
        link_index: The index of the link to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.

    Returns:
        The SE(3) transform of the link.
    """

    return forward_kinematics_model(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
    )[link_index]
