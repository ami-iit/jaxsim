from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.cross import Cross
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def aba(
    model: PhysicsModel,
    xfb: jtp.Vector,
    q: jtp.Vector,
    qd: jtp.Vector,
    tau: jtp.Vector,
    f_ext: jtp.Matrix | None = None,
) -> Tuple[jtp.Vector, jtp.Vector]:
    """
    Articulated Body Algorithm (ABA) algorithm with motor dynamics for forward dynamics.
    """

    x_fb, q, qd, _, tau, f_ext = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=qd, tau=tau, f_ext=f_ext
    )

    # Extract data from the physics model
    pre_X_λi = model.tree_transforms
    M = model.spatial_inertias
    i_X_pre = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    λ = model.parent_array()

    # Extract motor parameters from the physics model
    Γ = jnp.array([*model._joint_motor_gear_ratio.values()])
    IM = jnp.array(
        [jnp.eye(6) * m for m in [*model._joint_motor_inertia.values()]] * model.NB
    )
    K̅ᵥ = Γ.T * jnp.array([*model._joint_motor_viscous_friction.values()]) * Γ
    m_S = jnp.concatenate([S[:1], S[1:] * Γ[:, None, None]], axis=0)

    # Initialize buffers
    v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    MA = jnp.array([jnp.zeros([6, 6])] * model.NB)
    pA = jnp.array([jnp.zeros([6, 1])] * model.NB)
    c = jnp.array([jnp.zeros([6, 1])] * model.NB)
    i_X_λi = jnp.zeros_like(i_X_pre)

    m_v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    m_c = jnp.array([jnp.zeros([6, 1])] * model.NB)
    pR = jnp.array([jnp.zeros([6, 1])] * model.NB)

    # Base pose B_X_W and velocity
    base_quat = jnp.vstack(x_fb[0:4])
    base_pos = jnp.vstack(x_fb[4:7])
    base_vel = jnp.vstack(jnp.hstack([x_fb[10:13], x_fb[7:10]]))

    # 6D transform of base velocity
    B_X_W = Adjoint.from_quaternion_and_translation(
        quaternion=base_quat,
        translation=base_pos,
        inverse=True,
        normalize_quaternion=True,
    )
    i_X_λi = i_X_λi.at[0].set(B_X_W)

    # Transforms link -> base
    i_X_0 = jnp.zeros_like(pre_X_λi)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Initialize base quantities
    if model.is_floating_base:
        # Base velocity v₀
        v_0 = B_X_W @ base_vel
        v = v.at[0].set(v_0)

        # AB inertia (Mᴬ) and AB bias forces (pᴬ)
        MA_0 = M[0]
        MA = MA.at[0].set(MA_0)
        pA_0 = Cross.vx_star(v[0]) @ MA_0 @ v[0] - Adjoint.inverse(
            B_X_W
        ).T @ jnp.vstack(f_ext[0])
        pA = pA.at[0].set(pA_0)

    Pass1Carry = Tuple[
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
    ]

    pass_1_carry = (i_X_λi, v, c, m_v, m_c, MA, pA, pR, i_X_0)

    # Pass 1
    def loop_body_pass1(carry: Pass1Carry, i: jtp.Int) -> Tuple[Pass1Carry, None]:
        ii = i - 1
        i_X_λi, v, c, m_v, m_c, MA, pA, pR, i_X_0 = carry

        # Compute parent-to-child transform
        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        # Propagate link velocity
        vJ = S[i] * qd[ii] * (qd.size != 0)
        m_vJ = m_S[i] * qd[ii] * (qd.size != 0)

        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        m_v_i = i_X_λi[i] @ v[λ[i]] + m_vJ
        m_v = m_v.at[i].set(m_v_i)

        c_i = Cross.vx(v[i]) @ vJ
        c = c.at[i].set(c_i)
        m_c_i = Cross.vx(m_v[i]) @ m_vJ
        m_c = m_c.at[i].set(m_c_i)

        # Initialize articulated-body inertia
        MA_i = jnp.array(M[i])
        MA = MA.at[i].set(MA_i)

        # Initialize articulated-body bias forces
        i_X_0_i = i_X_λi[i] @ i_X_0[model.parent[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)
        i_Xf_W = Adjoint.inverse(i_X_0[i] @ B_X_W).T

        pA_i = Cross.vx_star(v[i]) @ M[i] @ v[i] - i_Xf_W @ jnp.vstack(f_ext[i])
        pA = pA.at[i].set(pA_i)

        pR_i = Cross.vx_star(m_v[i]) @ IM[i] @ m_v[i] - K̅ᵥ[i] * m_v[i]
        pR = pR.at[i].set(pR_i)

        return (i_X_λi, v, c, m_v, m_c, MA, pA, pR, i_X_0), None

    (i_X_λi, v, c, m_v, m_c, MA, pA, pR, i_X_0), _ = jax.lax.scan(
        f=loop_body_pass1,
        init=pass_1_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    U = jnp.zeros_like(S)
    m_U = jnp.zeros_like(S)
    d = jnp.zeros(shape=(model.NB, 1))
    u = jnp.zeros(shape=(model.NB, 1))
    m_u = jnp.zeros(shape=(model.NB, 1))

    Pass2Carry = Tuple[
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
        jtp.MatrixJax,
    ]

    pass_2_carry = (U, m_U, d, u, m_u, MA, pA)

    # Pass 2
    def loop_body_pass2(carry: Pass2Carry, i: jtp.Int) -> Tuple[Pass2Carry, None]:
        ii = i - 1
        U, m_U, d, u, m_u, MA, pA = carry

        # Compute intermediate results
        u_i = tau[ii] - S[i].T @ pA[i] if tau.size != 0 else -S[i].T @ pA[i]
        u = u.at[i].set(u_i.squeeze())

        has_motors = jnp.allclose(Γ[i], 1.0)

        m_u_i = (
            tau[ii] / Γ[i] * has_motors - m_S[i].T @ pR[i]
            if tau.size != 0
            else -m_S[i].T @ pR[i]
        )
        m_u = m_u.at[i].set(m_u_i.squeeze())

        U_i = MA[i] @ S[i]
        U = U.at[i].set(U_i)

        m_U_i = IM[i] @ m_S[i]
        m_U = m_U.at[i].set(m_U_i)

        d_i = S[i].T @ MA[i] @ S[i] + m_S[i].T @ IM[i] @ m_S[i]
        d = d.at[i].set(d_i.squeeze())

        # Compute the articulated-body inertia and bias forces of this link
        Ma = MA[i] + IM[i] - U[i] / d[i] @ U[i].T - m_U[i] / d[i] @ m_U[i].T
        pa = (
            pA[i]
            + pR[i]
            + Ma[i] @ c[i]
            + IM[i] @ m_c[i]
            + U[i] / d[i] * u[i]
            + m_U[i] / d[i] * m_u[i]
        )

        # Propagate them to the parent, handling the base link
        def propagate(
            MA_pA: Tuple[jtp.MatrixJax, jtp.MatrixJax]
        ) -> Tuple[jtp.MatrixJax, jtp.MatrixJax]:
            MA, pA = MA_pA

            MA_λi = MA[λ[i]] + i_X_λi[i].T @ Ma @ i_X_λi[i]
            MA = MA.at[λ[i]].set(MA_λi)

            pA_λi = pA[λ[i]] + i_X_λi[i].T @ pa
            pA = pA.at[λ[i]].set(pA_λi)

            return MA, pA

        MA, pA = jax.lax.cond(
            pred=jnp.array([λ[i] != 0, model.is_floating_base]).any(),
            true_fun=propagate,
            false_fun=lambda MA_pA: MA_pA,
            operand=(MA, pA),
        )

        return (U, m_U, d, u, m_u, MA, pA), None

    (U, m_U, d, u, m_u, MA, pA), _ = jax.lax.scan(
        f=loop_body_pass2,
        init=pass_2_carry,
        xs=np.flip(np.arange(start=1, stop=model.NB)),
    )

    if model.is_floating_base:
        a0 = jnp.linalg.solve(-MA[0], pA[0])
    else:
        a0 = -B_X_W @ jnp.vstack(model.gravity)

    a = jnp.zeros_like(S)
    a = a.at[0].set(a0)
    qdd = jnp.zeros_like(q)

    Pass3Carry = Tuple[jtp.MatrixJax, jtp.VectorJax]
    pass_3_carry = (a, qdd)

    # Pass 3
    def loop_body_pass3(carry: Pass3Carry, i: jtp.Int) -> Tuple[Pass3Carry, None]:
        ii = i - 1
        a, qdd = carry

        # Propagate link accelerations
        a_i = i_X_λi[i] @ a[λ[i]] + c[i]

        # Compute joint accelerations
        qdd_ii = (u[i] + m_u[i] - (U[i].T + m_U[i].T) @ a_i) / d[i]
        qdd = qdd.at[ii].set(qdd_ii.squeeze()) if qdd.size != 0 else qdd

        a_i = a_i + S[i] * qdd[ii] if qdd.size != 0 else a_i
        a = a.at[i].set(a_i)

        return (a, qdd), None

    (a, qdd), _ = jax.lax.scan(
        f=loop_body_pass3,
        init=pass_3_carry,
        xs=np.arange(1, model.NB),
    )

    # Handle 1 DoF models
    qdd = jnp.atleast_1d(qdd.squeeze())
    qdd = jnp.vstack(qdd) if qdd.size > 0 else jnp.empty(shape=(0, 1))

    # Get the resulting base acceleration (w/o gravity) in body-fixed representation
    B_a_WB = a[0]

    # Convert the base acceleration to inertial-fixed representation, and add gravity
    W_a_WB = jnp.vstack(
        jnp.linalg.solve(B_X_W, B_a_WB) + jnp.vstack(model.gravity)
        if model.is_floating_base
        else jnp.zeros(6)
    )

    return W_a_WB, qdd
