from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def jacobian(model: PhysicsModel, body_index: jtp.Int, q: jtp.Vector) -> jtp.Matrix:
    """
    Compute the Jacobian matrix for a specific link in an articulated body or robot.

    Args:
        model (PhysicsModel): The physics model of the articulated body or robot.
        body_index (jtp.Int): The index of the link for which to compute the Jacobian matrix.
        q (jtp.Vector): Joint positions (Generalized coordinates).

    Returns:
        jtp.Matrix: The Jacobian matrix for the specified link.
    """
    _, q, _, _, _, _ = utils.process_inputs(physics_model=model, q=q)

    S = model.motion_subspaces(q=q)
    i_X_pre = model.joint_transforms(q=q)
    pre_X_λi = model.tree_transforms
    i_X_λi = jnp.zeros_like(i_X_pre)

    i_X_0 = jnp.zeros_like(i_X_pre)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Parent array mapping: i -> λ(i).
    # Exception: λ(0) must not be used, it's initialized to -1.
    λ = model.parent

    # ====================
    # Propagate kinematics
    # ====================

    PropagateKinematicsCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax]
    propagate_kinematics_carry = (i_X_λi, i_X_0)

    def propagate_kinematics(
        carry: PropagateKinematicsCarry, i: jtp.Int
    ) -> Tuple[PropagateKinematicsCarry, None]:
        i_X_λi, i_X_0 = carry

        # For each body (i), compute the parent (λi) to body (i) adjoint matrix
        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        # Compute the base (0) to body (i) adjoint matrix.
        # This works fine since we traverse the kinematic tree following the link
        # indices assigned with BFS.
        i_X_0_i = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

        return (i_X_λi, i_X_0), None

    (i_X_λi, i_X_0), _ = jax.lax.scan(
        f=propagate_kinematics,
        init=propagate_kinematics_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    # ============================
    # Compute doubly-left Jacobian
    # ============================

    J = jnp.zeros(shape=(6, 6 + model.dofs()))

    Jb = i_X_0[body_index]
    J = J.at[0:6, 0:6].set(Jb)

    # To make JIT happy, we operate on a boolean version of κ(i).
    # Checking if j ∈ κ(i) is equivalent to: κ_bool(j) is True.
    κ_bool = model.support_body_array_bool(body_index=body_index)

    def compute_jacobian(J: jtp.MatrixJax, i: jtp.Int) -> Tuple[jtp.MatrixJax, None]:
        def update_jacobian(J: jtp.MatrixJax, i: jtp.Int) -> jtp.MatrixJax:
            ii = i - 1
            Js_i = i_X_0[body_index] @ Adjoint.inverse(i_X_0[i]) @ S[i]
            J = J.at[0:6, 6 + ii].set(Js_i.squeeze())

            return J

        J = jax.lax.select(pred=κ_bool[i], on_true=update_jacobian(J, i), on_false=J)
        return J, None

    J, _ = jax.lax.scan(
        f=compute_jacobian, init=J, xs=np.arange(start=1, stop=model.NB)
    )

    return J
