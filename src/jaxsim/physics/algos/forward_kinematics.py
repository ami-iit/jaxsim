from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def forward_kinematics_model(
    model: PhysicsModel, q: jtp.Vector, xfb: jtp.Vector
) -> jtp.Array:
    """
    Compute the forward kinematics transformations for all links in an articulated body or robot.

    Args:
        model (PhysicsModel): The physics model of the articulated body or robot.
        q (jtp.Vector): Joint positions (Generalized coordinates).
        xfb (jtp.Vector): The base pose vector, including the quaternion (first 4 elements) and translation (last 3 elements).

    Returns:
        jtp.Array: A 3D array containing the forward kinematics transformations for all links.
    """

    x_fb, q, _, _, _, _ = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=None, tau=None, f_ext=None
    )

    W_X_0 = Adjoint.from_quaternion_and_translation(
        quaternion=x_fb[0:4], translation=x_fb[4:7]
    )

    # This is the 6D velocity transform from i-th link frame to the world frame
    W_X_i = jnp.zeros(shape=[model.NB, 6, 6])
    W_X_i = W_X_i.at[0].set(W_X_0)

    i_X_pre = model.joint_transforms(q=q)
    pre_X_λi = model.tree_transforms

    # This is the parent-to-child 6D velocity transforms of all links
    i_X_λi = jnp.zeros_like(i_X_pre)

    # Parent array mapping: i -> λ(i).
    # Exception: λ(0) must not be used, it's initialized to -1.
    λ = model.parent

    PropagateKinematicsCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax]
    propagate_kinematics_carry = (i_X_λi, W_X_i)

    def propagate_kinematics(
        carry: PropagateKinematicsCarry, i: jtp.Int
    ) -> Tuple[PropagateKinematicsCarry, None]:
        i_X_λi, W_X_i = carry

        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        W_X_i_i = W_X_i[λ[i]] @ Adjoint.inverse(i_X_λi[i])
        W_X_i = W_X_i.at[i].set(W_X_i_i)

        return (i_X_λi, W_X_i), None

    (_, W_X_i), _ = jax.lax.scan(
        f=propagate_kinematics,
        init=propagate_kinematics_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    return jnp.stack([Adjoint.to_transform(adjoint=X) for X in list(W_X_i)])


def forward_kinematics(
    model: PhysicsModel, body_index: jtp.Int, q: jtp.Vector, xfb: jtp.Vector
) -> jtp.Matrix:
    return forward_kinematics_model(model=model, q=q, xfb=xfb)[body_index]
