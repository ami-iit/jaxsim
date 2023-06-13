from typing import Tuple

import jax
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

    # Pass 1
    def loop_body_pass1(carry, i):
        qd, f_ext, i_X_λi, v, c, MA, pA, i_X_0 = carry  # Add i_X_0 as an argument

        # Compute parent-to-child transform
        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        # Propagate link velocity
        vJ = S[i] * qd[i - 1] if qd.size != 0 else S[i] * 0

        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        c_i = Cross.vx(v[i]) @ vJ
        c = c.at[i].set(c_i)

        # Initialize articulated-body inertia
        MA_i = jnp.array(M[i])
        MA = MA.at[i].set(MA_i)

        # Initialize articulated-body bias forces
        i_X_0_i = i_X_λi[i] @ i_X_0[model.parent[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)
        i_X_W = jnp.linalg.inv(i_X_0[i] @ B_X_W).T

        pA_i = Cross.vx_star(v[i]) @ M[i] @ v[i] - i_X_W @ jnp.vstack(f_ext[i])
        pA = pA.at[i].set(pA_i)

        return (qd, f_ext, i_X_λi, v, c, MA, pA, i_X_0), ()

    carry = (qd, f_ext, i_X_λi, v, c, MA, pA, i_X_0)
    (qd, f_ext, i_X_λi, v, c, MA, pA, i_X_0), _ = jax.lax.scan(
        loop_body_pass1, carry, jnp.arange(1, model.NB + 1)
    )

    U = jnp.zeros_like(S)
    d = jnp.zeros(shape=(model.NB, 1))
    u = jnp.zeros(shape=(model.NB, 1))

    # Pass 2
    def loop_body_pass2(carry, i):
        tau, U, d, u, MA, pA = carry

        # Compute intermediate results
        U_i = MA[i] @ S[i]
        U = U.at[i].set(U_i)

        d_i = S[i].T @ U[i]
        d = d.at[i].set(d_i.squeeze())

        u_i = tau[i - 1] - S[i].T @ pA[i] if tau.size != 0 else -S[i].T @ pA[i]
        u = u.at[i].set(u_i.squeeze())

        # Compute the articulated-body inertia and bias forces of this link
        Ma = MA[i] - U[i] / d[i] @ U[i].T
        pa = pA[i] + Ma @ c[i] + U[i] * u[i] / d[i]

        # Propagate them to the parent, handling the base link
        def propagate(MA_pA):
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

        return (tau, U, d, u, MA, pA), ()

    carry = (tau, U, d, u, MA, pA)
    (tau, U, d, u, MA, pA), _ = jax.lax.scan(
        loop_body_pass2, carry, jnp.arange(model.NB - 1, 0, -1)
    )

    if model.is_floating_base:
        a0 = jnp.linalg.solve(-MA[0], pA[0])
    else:
        a0 = -B_X_W @ jnp.vstack(model.gravity)

    a = jnp.zeros_like(S)
    a = a.at[0].set(a0)
    qdd = jnp.zeros_like(q)

    # Pass 3
    def loop_body_pass3(carry, i):
        a, qdd = carry

        # Propagate link accelerations
        a_i = i_X_λi[i] @ a[λ[i]] + c[i]

        # Compute joint accelerations
        qdd_ii = (u[i] - U[i].T @ a_i) / d[i]
        qdd = qdd.at[i - 1].set(qdd_ii.squeeze()) if qdd.size != 0 else qdd

        a_i = a_i + S[i] * qdd[i - 1] if qdd.size != 0 else a_i
        a = a.at[i].set(a_i)

        return (a, qdd), ()

    carry = (a, qdd)
    (a_, qdd), () = jax.lax.scan(loop_body_pass3, carry, jnp.arange(1, model.NB + 1))

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
