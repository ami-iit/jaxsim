from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.cross import Cross
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def coriolis(
    model: PhysicsModel,
    q: jnp.ndarray,
    qd: jnp.ndarray,
    xfb: jtp.Vector,
) -> Tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    """
    Coriolis matrix
    """

    (
        x_fb,
        q,
        qd,
        _,
        _,
        _,
    ) = utils.process_inputs(
        physics_model=model,
        xfb=xfb,
        q=q,
        qd=qd,
    )

    # Extract data from the physics model
    pre_X_λi = model.tree_transforms
    M = model.spatial_inertias
    i_X_pre = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    λ = model.parent_array()

    # Initialize buffers
    v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    Sd = jnp.array([jnp.zeros([6, 1])] * model.NB)
    BC = jnp.array([jnp.zeros([6, 6])] * model.NB)
    IC = jnp.zeros_like(M)

    i_X_λi = jnp.zeros_like(i_X_pre)

    # Base pose B_X_W and velocity
    base_quat = jnp.vstack(x_fb[0:4])
    base_pos = jnp.vstack(x_fb[4:7])

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

    Pass1Carry = Tuple[
        jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax
    ]

    def loop_pass_1(carry: Pass1Carry, i: jtp.Int) -> Tuple[Pass1Carry, None]:
        i_X_λi, v, Sd, BC, IC = carry
        vJ = S[i] * qd[i]
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        Sd_i = Cross.vx(v[i]) @ S[i]
        Sd = Sd.at[i].set(Sd_i)

        IC = IC.at[i].set(M[i])
        BC_i = (
            Cross.vx_star(v[i]) @ Cross.vx(IC[i] @ v[i]) - IC[i] @ Cross.vx(v[i])
        ) / 2
        BC = BC.at[i].set(BC_i)

        return (i_X_λi, v, Sd, BC, IC), None

    (i_X_λi, v, Sd, BC, IC), _ = jax.lax.scan(
        f=loop_pass_1,
        init=(i_X_λi, v, Sd, BC, IC),
        xs=np.arange(1, model.NB + 1),
    )

    C = jnp.zeros([model.NB, model.NB])
    H = jnp.zeros([model.NB, model.NB])
    Hd = jnp.zeros([model.NB, model.NB])

    Pass2Carry = Tuple[
        jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax
    ]

    def loop_pass_2(carry: Pass2Carry, j: jtp.Int) -> Tuple[Pass2Carry, None]:
        jj = λ[j] - 1

        C, H, Hd, IC, BC = carry

        F_1 = IC[j] @ Sd[j] + BC[j] @ S[j]
        F_2 = IC[j] @ S[j]
        F_3 = BC[j].T @ S[j]

        C = C.at[jj, jj].set((S[j].T @ F_1).squeeze())
        H = H.at[jj, jj].set((S[j].T @ F_2).squeeze())
        Hd = Hd.at[jj, jj].set((Sd[j].T @ F_2 + S[j].T @ F_3).squeeze())

        F_1 = i_X_λi[j] @ F_1
        F_2 = i_X_λi[j] @ F_2
        F_3 = i_X_λi[j] @ F_3

        InnerLoopCarry = Tuple[
            jtp.MatrixJax,
            jtp.MatrixJax,
            jtp.MatrixJax,
            jtp.MatrixJax,
            jtp.MatrixJax,
            jtp.MatrixJax,
            jtp.MatrixJax,
        ]

        def inner_loop_body(carry: InnerLoopCarry) -> Tuple[InnerLoopCarry]:
            C, H, Hd, F_1, F_2, F_3, i = carry
            ii = λ[i] - 1

            C = C.at[ii, jj].set((S[i].T @ F_1).squeeze())
            C = C.at[jj, ii].set((S[i].T @ F_1).squeeze())

            H = H.at[ii, ii].set((S[i].T @ F_2).squeeze())
            Hd = Hd.at[ii].set((Sd[i].T @ F_2 + S[i].T @ F_3).squeeze())

            F_1 = F_1 + i_X_λi[i] @ F_1
            F_2 = F_2 + i_X_λi[i] @ F_2
            F_3 = F_3 + i_X_λi[i] @ F_3

            i = λ[i]
            return C, H, Hd, F_1, F_2, F_3, i

        (C, H, Hd, F_1, F_2, F_3, _) = jax.lax.while_loop(
            body_fun=inner_loop_body,
            cond_fun=lambda idx: idx[-1] > 0,
            init_val=(C, H, Hd, F_1, F_2, F_3, 0),
        )

        def propagate(
            IC_BC: Tuple[jtp.MatrixJax, jtp.MatrixJax]
        ) -> Tuple[jtp.MatrixJax, jtp.MatrixJax]:
            IC, BC = IC_BC

            IC = IC.at[λ[j]].set(IC[λ[j]] + i_X_λi[j] @ IC[j] @ i_X_λi[j].T)
            BC = BC.at[λ[j]].set(BC[λ[j]] + i_X_λi[j] @ BC[j] @ i_X_λi[j].T)

            return IC, BC

        IC, BC = jax.lax.cond(
            pred=jnp.array([λ[j] != 0, model.is_floating_base]).any(),
            true_fun=propagate,
            false_fun=lambda IC_BC: IC_BC,
            operand=(IC, BC),
        )

        return (C, H, Hd, IC, BC), None

    (C, H, Hd, IC, BC), _ = jax.lax.scan(
        f=loop_pass_2,
        init=(C, H, Hd, IC, BC),
        xs=np.flip(np.arange(1, model.NB + 1)),
    )

    return H, Hd, C
