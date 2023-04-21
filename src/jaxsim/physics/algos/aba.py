from typing import Tuple

import jax
import jax.experimental.loops
import jax.numpy as jnp

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
    f_ext: jtp.Matrix = None,
) -> Tuple[jtp.Vector, jtp.Vector]:
    x_fb, q, qd, _, tau, f_ext = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=qd, tau=tau, f_ext=f_ext
    )

    # Extract data from the physics model
    pre_X_λi = model.tree_transforms
    M = model.spatial_inertias
    i_X_pre = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    λ = model.parent_array()

    # Initialize buffers
    v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    MA = jnp.array([jnp.zeros([6, 6])] * model.NB)
    pA = jnp.array([jnp.zeros([6, 1])] * model.NB)
    c = jnp.array([jnp.zeros([6, 1])] * model.NB)
    i_X_λi = jnp.zeros_like(i_X_pre)

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
        pA_0 = Cross.vx_star(v[0]) @ MA_0 @ v[0] - jnp.linalg.inv(B_X_W).T @ jnp.vstack(
            f_ext[0]
        )
        pA = pA.at[0].set(pA_0)

    with jax.experimental.loops.Scope() as s:  # Pass 1
        s.qd = qd
        s.f_ext = f_ext

        s.i_X_λi = i_X_λi
        s.v = v
        s.c = c
        s.MA = MA
        s.pA = pA

        for i in range(1, model.NB):
            ii = i - 1

            # Compute parent-to-child transform
            i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
            s.i_X_λi = s.i_X_λi.at[i].set(i_X_λi_i)

            # Propagate link velocity
            vJ = S[i] * s.qd[ii] if s.qd.size != 0 else S[i] * 0

            v_i = s.i_X_λi[i] @ s.v[λ[i]] + vJ
            s.v = s.v.at[i].set(v_i)

            c_i = Cross.vx(s.v[i]) @ vJ
            s.c = s.c.at[i].set(c_i)

            # Initialize articulated-body inertia
            MA_i = jnp.array(M[i])
            s.MA = s.MA.at[i].set(MA_i)

            # Initialize articulated-body bias forces
            i_X_0_i = s.i_X_λi[i] @ i_X_0[model.parent[i]]
            i_X_0 = i_X_0.at[i].set(i_X_0_i)
            i_X_W = jnp.linalg.inv(i_X_0[i] @ B_X_W).T

            pA_i = Cross.vx_star(s.v[i]) @ M[i] @ s.v[i] - i_X_W @ jnp.vstack(
                s.f_ext[i]
            )
            s.pA = s.pA.at[i].set(pA_i)

        i_X_λi = s.i_X_λi
        c = s.c
        MA = s.MA
        pA = s.pA

    U = jnp.zeros_like(S)
    d = jnp.zeros(shape=(model.NB, 1))
    u = jnp.zeros(shape=(model.NB, 1))

    with jax.experimental.loops.Scope() as s:  # Pass 2
        s.tau = tau

        s.U = U
        s.d = d
        s.u = u
        s.MA = MA
        s.pA = pA

        for i in s.range(model.NB - 1, 0, -1):
            ii = i - 1

            # Compute intermediate results
            U_i = s.MA[i] @ S[i]
            s.U = s.U.at[i].set(U_i)

            d_i = S[i].T @ s.U[i]
            s.d = s.d.at[i].set(d_i.squeeze())

            u_i = s.tau[ii] - S[i].T @ s.pA[i] if s.tau.size != 0 else -S[i].T @ s.pA[i]
            s.u = s.u.at[i].set(u_i.squeeze())

            # Compute the articulated-body inertia and bias forces of this link
            Ma = s.MA[i] - s.U[i] / s.d[i] @ s.U[i].T
            pa = s.pA[i] + Ma @ c[i] + s.U[i] * s.u[i] / s.d[i]

            # Propagate them to the parent, handling the base link
            def propagate(MA_pA):
                MA, pA = MA_pA

                MA_λi = MA[λ[i]] + i_X_λi[i].T @ Ma @ i_X_λi[i]
                MA = MA.at[λ[i]].set(MA_λi)

                pA_λi = pA[λ[i]] + i_X_λi[i].T @ pa
                pA = pA.at[λ[i]].set(pA_λi)

                return MA, pA

            s.MA, s.pA = jax.lax.cond(
                pred=jnp.array([λ[i] != 0, model.is_floating_base]).any(),
                true_fun=propagate,
                false_fun=lambda MA_pA: MA_pA,
                operand=(s.MA, s.pA),
            )

        U = s.U
        d = s.d
        u = s.u
        MA = s.MA
        pA = s.pA

    if model.is_floating_base:
        a0 = jnp.linalg.solve(-MA[0], pA[0])
    else:
        a0 = -B_X_W @ jnp.vstack(model.gravity)

    a = jnp.zeros_like(S)
    a = a.at[0].set(a0)
    qdd = jnp.zeros_like(q)

    with jax.experimental.loops.Scope() as s:  # Pass 3
        s.a = a
        s.qdd = qdd

        for i in s.range(1, model.NB):
            ii = i - 1

            # Propagate link accelerations
            a_i = i_X_λi[i] @ s.a[λ[i]] + c[i]

            # Compute joint accelerations
            qdd_ii = (u[i] - U[i].T @ a_i) / d[i]
            s.qdd = s.qdd.at[ii].set(qdd_ii.squeeze()) if qdd.size != 0 else s.qdd

            a_i = a_i + S[i] * s.qdd[ii] if qdd.size != 0 else a_i
            s.a = s.a.at[i].set(a_i)

        a = s.a
        qdd = s.qdd

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
