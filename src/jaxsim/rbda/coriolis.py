import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Cross, StandardGravity, Transform

from . import utils


def coriolis(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
    base_linear_velocity: jtp.VectorLike,
    base_angular_velocity: jtp.VectorLike,
    joint_velocities: jtp.VectorLike,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    standard_gravity: jtp.FloatLike = StandardGravity,
) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    """
    Coriolis matrix
    """

    W_p_B, W_Q_B, s, _, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        joint_velocities=joint_velocities,
        standard_gravity=standard_gravity,
    )

    W_H_B = Transform.from_quaternion_and_translation(
        quaternion=W_Q_B,
        translation=W_p_B,
    )

    # Extract data from the physics model
    pre_X_λi = model.tree_transforms
    M = js.model.link_spatial_inertia_matrices(model=model)
    i_X_pre, S = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=s, base_transform=W_H_B.as_matrix()
    )
    λ = model.kin_dyn_parameters.parent_array

    # Initialize buffers
    v = jnp.array([jnp.zeros([6, 1])] * model.number_of_links())
    Ṡ = jnp.array([jnp.zeros([6, 1])] * model.number_of_links())
    BC = jnp.array([jnp.zeros([6, 6])] * model.number_of_links())
    IC = jnp.zeros_like(M)

    i_X_λi = jnp.zeros_like(i_X_pre)

    # 6D transform of base velocity
    B_X_W = Adjoint.from_quaternion_and_translation(
        quaternion=W_Q_B,
        translation=W_p_B,
        inverse=True,
        normalize_quaternion=True,
    )
    i_X_λi = i_X_λi.at[0].set(B_X_W)

    # Transforms link -> base
    i_X_0 = jnp.zeros_like(pre_X_λi)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    Pass1Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]

    def loop_pass_1(carry: Pass1Carry, i: jtp.Int) -> tuple[Pass1Carry, None]:
        i_X_λi, v, Ṡ, BC, IC = carry
        vJ = S[i] * ṡ[i]
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        Ṡ_i = Cross.vx(v[i]) @ S[i]
        Ṡ = Ṡ.at[i].set(Ṡ_i)

        IC = IC.at[i].set(M[i])
        BC_i = (
            Cross.vx_star(v[i]) @ Cross.vx(IC[i] @ v[i]) - IC[i] @ Cross.vx(v[i])
        ) / 2
        BC = BC.at[i].set(BC_i)

        return (i_X_λi, v, Ṡ, BC, IC), None

    (i_X_λi, v, Ṡ, BC, IC), _ = (
        jax.lax.scan(
            f=loop_pass_1,
            init=(i_X_λi, v, Ṡ, BC, IC),
            xs=jnp.arange(1, model.number_of_links() + 1),
        )
        if model.number_of_links() > 1
        else [(i_X_λi, v, Ṡ, BC, IC), None]
    )

    C = jnp.zeros([model.number_of_links(), model.number_of_links()])
    M = jnp.zeros([model.number_of_links(), model.number_of_links()])
    Ṁ = jnp.zeros([model.number_of_links(), model.number_of_links()])

    Pass2Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]

    def loop_pass_2(carry: Pass2Carry, j: jtp.Int) -> tuple[Pass2Carry, None]:
        jj = λ[j] - 1

        C, M, Ṁ, IC, BC = carry

        F_1 = IC[j] @ Ṡ[j] + BC[j] @ S[j]
        F_2 = IC[j] @ S[j]
        F_3 = BC[j].T @ S[j]

        C = C.at[jj, jj].set((S[j].T @ F_1).squeeze())
        M = M.at[jj, jj].set((S[j].T @ F_2).squeeze())
        Ṁ = Ṁ.at[jj, jj].set((Ṡ[j].T @ F_2 + S[j].T @ F_3).squeeze())

        F_1 = i_X_λi[j] @ F_1
        F_2 = i_X_λi[j] @ F_2
        F_3 = i_X_λi[j] @ F_3

        InnerLoopCarry = tuple[
            jtp.Matrix,
            jtp.Matrix,
            jtp.Matrix,
            jtp.Matrix,
            jtp.Matrix,
            jtp.Matrix,
            jtp.Matrix,
        ]

        def inner_loop_body(carry: InnerLoopCarry) -> tuple[InnerLoopCarry]:
            C, M, Ṁ, F_1, F_2, F_3, i = carry
            ii = λ[i] - 1

            C = C.at[ii, jj].set((S[i].T @ F_1).squeeze())
            C = C.at[jj, ii].set((S[i].T @ F_1).squeeze())

            M = M.at[ii, ii].set((S[i].T @ F_2).squeeze())
            Ṁ = Ṁ.at[ii].set((Ṡ[i].T @ F_2 + S[i].T @ F_3).squeeze())

            F_1 = F_1 + i_X_λi[i] @ F_1
            F_2 = F_2 + i_X_λi[i] @ F_2
            F_3 = F_3 + i_X_λi[i] @ F_3

            i = λ[i]
            return C, M, Ṁ, F_1, F_2, F_3, i

        (C, M, Ṁ, F_1, F_2, F_3, _) = jax.lax.while_loop(
            body_fun=inner_loop_body,
            cond_fun=lambda idx: idx[-1] > 0,
            init_val=(C, M, Ṁ, F_1, F_2, F_3, 0),
        )

        def propagate(
            IC_BC: tuple[jtp.Matrix, jtp.Matrix]
        ) -> tuple[jtp.Matrix, jtp.Matrix]:
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

        return (C, M, Ṁ, IC, BC), None

    (C, M, Ṁ, IC, BC), _ = (
        jax.lax.scan(
            f=loop_pass_2,
            init=(C, M, Ṁ, IC, BC),
            xs=jnp.flip(jnp.arange(1, model.number_of_links() + 1)),
        )
        if model.number_of_links() > 1
        else [(C, M, Ṁ, IC, BC), None]
    )

    return M, Ṁ, C
