import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp

from . import utils


def mass_inverse(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the inverse of the mass matrix using the Articulated Body Algorithm (ABA).

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.

    Returns:
        The inverse of the free-floating mass matrix.

    Note:
        The algorithm expects a quaternion with unit norm.
    """

    W_p_B, W_Q_B, s, W_v_WB, _, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
    )

    W_v_WB = jnp.atleast_2d(W_v_WB).T

    # Get the 6D spatial inertia matrices of all links.
    M = js.model.link_spatial_inertia_matrices(model=model)

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the base transform.
    W_H_B = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=W_Q_B),
        translation=W_p_B,
    )

    # Compute the parent-to-child adjoints and the motion subspaces of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi, S = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=s, base_transform=W_H_B.as_matrix()
    )

    # Allocate buffers.
    MA = jnp.zeros(shape=(model.number_of_links(), 6, 6))

    # Allocate the buffer of transforms link -> base.
    i_X_0 = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Initialize base quantities.
    if model.floating_base():
        # Initialize the articulated-body inertia (Mᴬ) of base link.
        MA_0 = M[0]
        MA = MA.at[0].set(MA_0)

    # ======
    # Pass 1
    # ======

    Pass1Carry = tuple[jtp.Matrix, jtp.Matrix]
    pass_1_carry: Pass1Carry = (MA, i_X_0)

    # Propagate kinematics and initialize AB inertia and AB bias forces.
    def loop_body_pass1(carry: Pass1Carry, i: jtp.Int) -> tuple[Pass1Carry, None]:

        MA, i_X_0 = carry

        # Initialize the articulated-body inertia.
        MA_i = jnp.array(M[i])
        MA = MA.at[i].set(MA_i)

        # Compute the link-to-base transform.
        i_Xi_0 = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_Xi_0)

        return (MA, i_X_0), None

    (MA, i_X_0), _ = (
        jax.lax.scan(
            f=loop_body_pass1,
            init=pass_1_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(MA, i_X_0), None]
    )

    # ======
    # Pass 2
    # ======

    U = jnp.zeros_like(S)
    d = jnp.zeros(shape=(model.number_of_links(), 1))
    M_inv = jnp.zeros(
        shape=(
            model.number_of_joints() + 6 * model.floating_base(),
            model.number_of_joints() + 6 * model.floating_base(),
        )
    )

    if model.number_of_joints() == 0:
        M_inv_0 = jnp.linalg.solve(MA[0], jnp.eye(6))
        M_inv = M_inv.at[:].set(M_inv_0)

    F = jnp.zeros(
        shape=(
            model.number_of_joints(),
            6,
            model.number_of_joints() + 6 * model.floating_base(),
        )
    )

    Pass2Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]
    pass_2_carry: Pass2Carry = (U, d, M_inv, MA, F)

    def loop_body_pass2(carry: Pass2Carry, i: jtp.Int) -> tuple[Pass2Carry, None]:

        ii = i - 1
        (U, d, M_inv, MA, F) = carry

        U_i = MA[i] @ S[i]
        U = U.at[i].set(U_i)

        d_i = S[i].T @ U[i]
        d = d.at[i].set(d_i.squeeze())

        M_inv_ii = -S[i].T @ F[i] / d[i]

        M_inv = M_inv.at[ii].set(M_inv_ii.squeeze())
        M_inv = M_inv.at[ii, ii].set(M_inv[i, i] + 1 / d[i].squeeze())

        # Propagate them to the parent, handling the base link.
        def propagate(
            MA_F: tuple[jtp.Matrix, jtp.Matrix]
        ) -> tuple[jtp.Matrix, jtp.Matrix]:
            MA, F = MA_F

            # Compute the articulated-body inertia and bias force of this link.
            Fa = F[i] + U[i] @ M_inv[ii][jnp.newaxis, :]
            Ma = MA[i] - U[i] / d[i] @ U[i].T

            F = F.at[λ[i]].set(F[λ[i]] + i_X_λi[i].T @ Fa)
            MA = MA.at[λ[i]].set(MA[λ[i]] + i_X_λi[i].T @ Ma @ i_X_λi[i])

            return MA, F

        MA, F = jax.lax.cond(
            pred=jnp.logical_or(λ[i] != 0, model.floating_base()),
            true_fun=propagate,
            false_fun=lambda MA_F: MA_F,
            operand=(MA, F),
        )

        return (U, d, M_inv, MA, F), None

    with jax.disable_jit(True):
        (U, d, M_inv, MA, F), _ = (
            jax.lax.scan(
                f=loop_body_pass2,
                init=pass_2_carry,
                xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
            )
            if model.number_of_links() > 1
            else [(U, d, M_inv, MA, F), None]
        )

    # ======
    # Pass 3
    # ======

    P = jnp.zeros_like(F)

    Pass3Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix]
    pass_3_carry = (U, M_inv, P)

    def loop_body_pass3(carry: Pass3Carry, i: jtp.Int) -> tuple[Pass3Carry, None]:

        ii = i - 1
        U, M_inv, P = carry

        P_i = S[i] @ jnp.atleast_2d(M_inv[ii])
        P = P.at[i].set(P_i)

        def propagate_M_P(M_inv_P: tuple[jtp.Matrix, jtp.Array]) -> jtp.Vector:
            M_inv, P = M_inv_P

            M_inv = M_inv.at[ii].set(
                M_inv[ii]
                - (U[i].T.squeeze() @ i_X_λi[i].T @ P[λ[i]]).T / d[i].squeeze(),
            )

            P = P.at[i].set(i_X_λi[i].T @ P[λ[i]])

            return M_inv, P

        M_inv, P = jax.lax.cond(
            pred=jnp.logical_or(λ[i] != 0, model.floating_base()),
            true_fun=propagate_M_P,
            false_fun=lambda M_inv_P: (M_inv, P),
            operand=(M_inv, P),
        )

        return (U, M_inv, P), None

    with jax.disable_jit(True):
        (U, M_inv, P), _ = (
            jax.lax.scan(
                f=loop_body_pass3,
                init=pass_3_carry,
                xs=jnp.arange(1, model.number_of_links()),
            )
            if model.number_of_links() > 1
            else [(U, M_inv, P), None]
        )

    # ==============
    # Adjust outputs
    # ==============
    M_inv = M_inv.squeeze()

    # Mirror the upper triangle to the lower triangle.
    M_inv = jnp.triu(M_inv) + jnp.triu(M_inv, k=1).T

    return M_inv
