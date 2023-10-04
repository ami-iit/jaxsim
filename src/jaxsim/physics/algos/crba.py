from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def crba(model: PhysicsModel, q: jtp.Vector) -> jtp.Matrix:
    """
    Compute the Composite Rigid-Body Inertia Matrix (CRBA) for an articulated body or robot given joint positions.

    Args:
        model (PhysicsModel): The physics model of the articulated body or robot.
        q (jtp.Vector): Joint positions (Generalized coordinates).

    Returns:
        jtp.Matrix: The Composite Rigid-Body Inertia Matrix (CRBA) of the articulated body or robot.
    """

    _, q, _, _, _, _ = utils.process_inputs(
        physics_model=model, xfb=None, q=q, qd=None, tau=None, f_ext=None
    )

    Xtree = model.tree_transforms
    Mc = model.spatial_inertias
    S = model.motion_subspaces(q=q)
    Xj = model.joint_transforms(q=q)

    Xup = jnp.zeros_like(Xtree)
    i_X_0 = jnp.zeros_like(Xtree)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Parent array mapping: i -> λ(i).
    # Exception: λ(0) must not be used, it's initialized to -1.
    λ = model.parent

    # ====================
    # Propagate kinematics
    # ====================

    ForwardPassCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax]
    forward_pass_carry = (Xup, i_X_0)

    def propagate_kinematics(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> Tuple[ForwardPassCarry, None]:
        Xup, i_X_0 = carry

        Xup_i = Xj[i] @ Xtree[i]
        Xup = Xup.at[i].set(Xup_i)

        i_X_0_i = Xup[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

        return (Xup, i_X_0), None

    (Xup, i_X_0), _ = jax.lax.scan(
        f=propagate_kinematics,
        init=forward_pass_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    # ===================
    # Compute mass matrix
    # ===================

    M = jnp.zeros(shape=(6 + model.dofs(), 6 + model.dofs()))

    BackwardPassCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax]
    backward_pass_carry = (Mc, M)

    def backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> Tuple[BackwardPassCarry, None]:
        ii = i - 1
        Mc, M = carry

        Mc_λi = Mc[λ[i]] + Xup[i].T @ Mc[i] @ Xup[i]
        Mc = Mc.at[λ[i]].set(Mc_λi)

        Fi = Mc[i] @ S[i]
        M_ii = S[i].T @ Fi
        M = M.at[ii + 6, ii + 6].set(M_ii.squeeze())

        j = i

        CarryInnerFn = Tuple[jtp.Int, jtp.MatrixJax, jtp.MatrixJax]
        carry_inner_fn = (j, Fi, M)

        def while_loop_body(carry: CarryInnerFn) -> CarryInnerFn:
            j, Fi, M = carry

            Fi = Xup[j].T @ Fi
            j = λ[j]
            jj = j - 1

            M_ij = Fi.T @ S[j]

            M = M.at[ii + 6, jj + 6].set(M_ij.squeeze())
            M = M.at[jj + 6, ii + 6].set(M_ij.squeeze())

            return j, Fi, M

        # The following functions are part of a (rather messy) workaround for computing
        # a while loop using a for loop with fixed number of iterations.
        def inner_fn(carry: CarryInnerFn, k: jtp.Int) -> Tuple[CarryInnerFn, None]:
            def compute_inner(carry: CarryInnerFn) -> Tuple[CarryInnerFn, None]:
                j, Fi, M = carry
                out = jax.lax.cond(
                    pred=(λ[j] > 0),
                    true_fun=while_loop_body,
                    false_fun=lambda carry: carry,
                    operand=carry,
                )
                return out, None

            j, Fi, M = carry
            return jax.lax.cond(
                pred=(k == j),
                true_fun=compute_inner,
                false_fun=lambda carry: (carry, None),
                operand=carry,
            )

        (j, Fi, M), _ = jax.lax.scan(
            f=inner_fn,
            init=carry_inner_fn,
            xs=np.flip(np.arange(start=1, stop=model.NB)),
        )

        Fi = i_X_0[j].T @ Fi

        M = M.at[0:6, ii + 6].set(Fi.squeeze())
        M = M.at[ii + 6, 0:6].set(Fi.squeeze())

        return (Mc, M), None

    # This scan performs the backward pass to compute Mbj, Mjb and Mjj, that
    # also includes a fake while loop implemented with a scan and two cond.
    (Mc, M), _ = jax.lax.scan(
        f=backward_pass,
        init=backward_pass_carry,
        xs=np.flip(np.arange(start=1, stop=model.NB)),
    )

    # Store the locked 6D rigid-body inertia matrix Mbb ∈ ℝ⁶ˣ⁶
    M = M.at[0:6, 0:6].set(Mc[0])

    return M
