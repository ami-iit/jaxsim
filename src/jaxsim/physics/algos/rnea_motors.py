from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.cross import Cross
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def rnea(
    model: PhysicsModel,
    xfb: jtp.Vector,
    q: jtp.Vector,
    qd: jtp.Vector,
    qdd: jtp.Vector,
    a0fb: jtp.Vector = jnp.zeros(6),
    f_ext: jtp.Matrix | None = None,
) -> Tuple[jtp.Vector, jtp.Vector]:
    """
    Recursive Newton-Euler Algorithm (RNEA) algorithm for inverse dynamics.
    """

    xfb, q, qd, qdd, _, f_ext = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=qd, qdd=qdd, f_ext=f_ext
    )

    a0fb = a0fb.squeeze()
    gravity = model.gravity.squeeze()

    if a0fb.shape[0] != 6:
        raise ValueError(a0fb.shape)

    M = model.spatial_inertias
    pre_X_λi = model.tree_transforms
    i_X_pre = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    i_X_λi = jnp.zeros_like(pre_X_λi)

    Γ = jnp.array([*model._joint_motor_gear_ratio.values()])
    IM = jnp.array([*model._joint_motor_inertia.values()])
    K_v = jnp.array([*model._joint_motor_viscous_friction.values()])
    K̅ᵥ = jnp.diag(Γ.T * jnp.diag(K_v) * Γ)
    m_S = jnp.concatenate([S[:1], S[1:] * Γ[:, None, None]], axis=0)

    i_X_0 = jnp.zeros_like(pre_X_λi)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Parent array mapping: i -> λ(i).
    # Exception: λ(0) must not be used, it's initialized to -1.
    λ = model.parent_array()

    v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    a = jnp.array([jnp.zeros([6, 1])] * model.NB)
    f = jnp.array([jnp.zeros([6, 1])] * model.NB)

    v_m = jnp.array([jnp.zeros([6, 1])] * model.NB)
    a_m = jnp.array([jnp.zeros([6, 1])] * model.NB)
    f_m = jnp.array([jnp.zeros([6, 1])] * model.NB)

    # 6D transform of base velocity
    B_X_W = Adjoint.from_quaternion_and_translation(
        quaternion=xfb[0:4],
        translation=xfb[4:7],
        inverse=True,
        normalize_quaternion=True,
    )
    i_X_λi = i_X_λi.at[0].set(B_X_W)

    a_0 = -B_X_W @ jnp.vstack(gravity)
    a = a.at[0].set(a_0)

    if model.is_floating_base:
        W_v_WB = jnp.vstack(jnp.hstack([xfb[10:13], xfb[7:10]]))

        v_0 = B_X_W @ W_v_WB
        v = v.at[0].set(v_0)

        a_0 = B_X_W @ (jnp.vstack(a0fb) - jnp.vstack(gravity))
        a = a.at[0].set(a_0)

        f_0 = (
            M[0] @ a[0]
            + Cross.vx_star(v[0]) @ M[0] @ v[0]
            - Adjoint.inverse(B_X_W).T @ jnp.vstack(f_ext[0])
        )
        f = f.at[0].set(f_0)

    ForwardPassCarry = Tuple[
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
    ]
    forward_pass_carry = (i_X_λi, v, v_m, a, a_m, i_X_0, f, f_m)

    def forward_pass(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> Tuple[ForwardPassCarry, None]:
        ii = i - 1
        i_X_λi, v, v_m, a, a_m, i_X_0, f, f_m = carry

        vJ = S[i] * qd[ii]
        vJ_m = m_S[i] * qd[ii]

        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        v_i_m = i_X_λi[i] @ v_m[λ[i]] + vJ_m
        v_m = v_m.at[i].set(v_i_m)

        a_i = i_X_λi[i] @ a[λ[i]] + S[i] * qdd[ii] + Cross.vx(v[i]) @ vJ
        a = a.at[i].set(a_i)

        a_i_m = i_X_λi[i] @ a_m[λ[i]] + m_S[i] * qdd[ii] + Cross.vx(v_m[i]) @ vJ_m
        a_m = a_m.at[i].set(a_i_m)

        i_X_0_i = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)
        i_Xf_W = Adjoint.inverse(i_X_0[i] @ B_X_W).T

        f_i = (
            M[i] @ a[i]
            + Cross.vx_star(v[i]) @ M[i] @ v[i]
            - i_Xf_W @ jnp.vstack(f_ext[i])
        )
        f = f.at[i].set(f_i)

        f_i_m = IM[i] * a_m[i] + Cross.vx_star(v_m[i]) * IM[i] @ v_m[i]
        f_m = f_m.at[i].set(f_i_m)

        return (i_X_λi, v, v_m, a, a_m, i_X_0, f, f_m), None

    (i_X_λi, v, v_m, a, a_m, i_X_0, f, f_m), _ = jax.lax.scan(
        f=forward_pass,
        init=forward_pass_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    tau = jnp.zeros_like(q)

    BackwardPassCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax]
    backward_pass_carry = (tau, f, f_m)

    def backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> Tuple[BackwardPassCarry, None]:
        ii = i - 1
        tau, f, f_m = carry

        value = S[i].T @ f[i] + m_S[i].T @ f_m[i]  # + K̅ᵥ[i] * qd[ii]
        tau = tau.at[ii].set(value.squeeze())

        def update_f(ffm: Tuple[jtp.MatrixJax, jtp.MatrixJax]) -> jtp.MatrixJax:
            f, f_m = ffm
            f_λi = f[λ[i]] + i_X_λi[i].T @ f[i]
            f = f.at[λ[i]].set(f_λi)

            f_m_λi = f_m[λ[i]] + i_X_λi[i].T @ f_m[i]
            f_m = f_m.at[λ[i]].set(f_m_λi)
            return f, f_m

        f, f_m = jax.lax.cond(
            pred=jnp.array([λ[i] != 0, model.is_floating_base]).any(),
            true_fun=update_f,
            false_fun=lambda f: f,
            operand=(f, f_m),
        )

        return (tau, f, f_m), None

    (tau, f, f_m), _ = jax.lax.scan(
        f=backward_pass,
        init=backward_pass_carry,
        xs=np.flip(np.arange(start=1, stop=model.NB)),
    )

    # Handle 1 DoF models
    tau = jnp.atleast_1d(tau.squeeze())
    tau = jnp.vstack(tau) if tau.size > 0 else jnp.empty(shape=(0, 1))

    # Express the base 6D force in the world frame
    W_f0 = B_X_W.T @ jnp.vstack(f[0])

    return W_f0, tau
