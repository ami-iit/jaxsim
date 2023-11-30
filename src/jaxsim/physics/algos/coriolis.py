import jax
import jax.numpy as jnp

import jaxsim
from jaxsim.math.cross import Cross
from jaxsim.physics.model.physics_model import PhysicsModel


def coriolis(model: PhysicsModel, q: jnp.ndarray, qd: jnp.ndarray) -> None:
    """
    Coriolis matrix
    """

    # Extract data from the physics model
    pre_X_λi = model.tree_transforms
    M = model.spatial_inertias
    i_X_pre = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    λ = model.parent_array()

    # Initialize buffers
    v = jnp.array([jnp.zeros([6, 1])] * model.NB)
    Sd = jnp.array([jnp.zeros([6, 1])] * model.NB)
    BC = jnp.array([jnp.zeros([6, 1])] * model.NB)
    IC = jnp.array([jnp.zeros([6, 1])] * model.NB)
    Ic = jnp.zeros([6, 6])
    Bc = jnp.zeros([6, 6])

    Pass1Carry = Tuple[
        jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax
    ]

    def loop_pass_1(carry: Pass1Carry, i: jtp.Int) -> Tuple[Pass1Carry, None]:
        vJ = S[i] * qd[i]
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        Sd_i = Cross.vx(v[i]) @ S[i]

        IC = IC.at[i].set(MC[i])
        BC_i = (
            Cross.vx_star(v[i]) @ Cross.vx_star_bar(IC[i] @ v[i])
            - IC[i] @ Cross.vx(v[i])
        ) / 2

        return (i_X_λi, v, Sd, BC, IC), None

    (i_X_λi, v, Sd, BC, IC), _ = jax.lax.scan(
        loop_pass_1,
        (i_X_λi, v, Sd, BC, IC),
        jnp.arange(1, model.NB + 1),
    )

    F_1 = jnp.zeros([6, 6])
    F_2 = jnp.zeros([6, 6])
    F_3 = jnp.zeros([6, 6])

    Pass2Carry = Tuple[
        jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax
    ]

    def loop_pass_2(carry: Pass2Carry, i: jtp.Int) -> Tuple[Pass2Carry, None]:
        ii = i - 1
        i_X_λi, v, Sd, BC, IC = carry

        # Compute parent-to-child transform
        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        # Propagate link velocity
        vJ = S[i] * qd[ii] * (qd.size != 0)
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        Sd_i = Cross.vx(v[i]) @ S[i]

        IC = IC.at[i].set(MC[i])
        BC_i = (
            Cross.vx_star(v[i]) @ Cross.vx_star_bar(IC[i] @ v[i])
            - IC[i] @ Cross.vx(v[i])
        ) / 2

        InnerLoopCarry = Tuple[jtp.MatrixJax, jtp.MatrixJax, jtp.MatrixJax]

        def inner_loop_body(
            carry: InnerLoopCarry, i: jtp.Int
        ) -> Tuple[InnerLoopCarry, None]:
            F_1 = i_X_λi[i] @ F_1
            F_2 = i_X_λi[i] @ F_2
            F_3 = i_X_λi[i] @ F_3

            C_ij = S[i].T @ F_1
            C_ji = (Sd[i].T @ F_2) + (S[i].T @ F_3).T

            H_ij = S[i].T @ F_2
            H_ji = H_ij.T

            Hd_ij = Sd[i].T @ F_2 + S[i].T @ (F_1 + F_3)
            Hd_ji = Hd_ij.T

            i = λ[i]
            return (F_1, F_2, F_3), None

        jax.lax.while_loop(
            body_fun=inner_loop_body,
            cond_fun=i > 0,
            init_val=0,
        )

        Ic = Ic + i_X_λi[i] @ IC[i] @ i_X_λi[i].T
        Bc = Bc + i_X_λi[i] @ BC[i] @ i_X_λi[i].T
        return (i_X_λi, v, Sd, BC, IC), None

    (i_X_λi, v, Sd, BC, IC), _ = jax.lax.scan(
        loop_pass_2,
        (i_X_λi, v, Sd, BC, IC),
        jnp.arange(1, model.NB + 1),
    )

    return Ic, Bc


# if __name__ == "__main__":
#     import jax.numpy as jnp
#     import jaxsim
#     from jaxsim.high_level.model import Model
#     from pathlib import Path

#     urdf_path = Path(
#         "/home/flferretti/git/element_rl-for-codesign/assets/model/Hopper.sdf"
#     )

#     model = Model.build_from_model_description(model_description=urdf_path)

#     with jax.disable_jit():
#         H, H_dot, C = model.coriolis_matrix()
