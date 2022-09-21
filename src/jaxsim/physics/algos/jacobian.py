from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def jacobian(model: PhysicsModel, body_index: jtp.Int, q: jtp.Vector) -> jtp.Matrix:

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

        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

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

    ComputeJacobianCarry = jtp.MatrixJax
    compute_jacobian_carry = J

    def compute_jacobian(
        carry: ComputeJacobianCarry, i: jtp.Int
    ) -> Tuple[ComputeJacobianCarry, None]:
        def update_jacobian(
            carry: Tuple[ComputeJacobianCarry, jtp.Int]
        ) -> ComputeJacobianCarry:

            J, i = carry

            ii = i - 1

            Js_i = i_X_0[body_index] @ jnp.linalg.inv(i_X_0[i]) @ S[i]
            J = J.at[0:6, 6 + ii].set(Js_i.squeeze())

            return J

        carry = jax.lax.cond(
            pred=(jnp.any(i == model.support_body_array(body_index=body_index))),
            true_fun=update_jacobian,
            false_fun=lambda carry_i: carry_i[0],
            operand=(carry, i),
        )

        return carry, None

    J, _ = jax.lax.scan(
        f=compute_jacobian,
        init=compute_jacobian_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    return J
