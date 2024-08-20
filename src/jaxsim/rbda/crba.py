import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp

from . import utils


def crba(model: js.model.JaxSimModel, *, joint_positions: jtp.Vector) -> jtp.Matrix:
    """
    Compute the free-floating mass matrix using the Composite Rigid-Body Algorithm (CRBA).

    Args:
        model: The model to consider.
        joint_positions: The positions of the joints.

    Returns:
        The free-floating mass matrix of the model in body-fixed representation.
    """

    _, _, s, _, _, _, _, _, _, _ = utils.process_inputs(
        model=model, joint_positions=joint_positions
    )

    # Get the 6D spatial inertia matrices of all links.
    Mc = js.model.link_spatial_inertia_matrices(model=model)

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the parent-to-child adjoints and the motion subspaces of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi, S = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=s, base_transform=jnp.eye(4)
    )

    # Allocate the buffer of transforms link -> base.
    i_X_0 = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # ====================
    # Propagate kinematics
    # ====================

    ForwardPassCarry = tuple[jtp.Matrix]
    forward_pass_carry: ForwardPassCarry = (i_X_0,)

    def propagate_kinematics(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> tuple[ForwardPassCarry, None]:

        (i_X_0,) = carry

        i_X_0_i = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

        return (i_X_0,), None

    (i_X_0,), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=forward_pass_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(i_X_0,), None]
    )

    # ===================
    # Compute mass matrix
    # ===================

    M = jnp.zeros(shape=(6 + model.dofs(), 6 + model.dofs()))

    BackwardPassCarry = tuple[jtp.Matrix, jtp.Matrix]
    backward_pass_carry: BackwardPassCarry = (Mc, M)

    def backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> tuple[BackwardPassCarry, None]:

        ii = i - 1
        Mc, M = carry

        Mc_λi = Mc[λ[i]] + i_X_λi[i].T @ Mc[i] @ i_X_λi[i]
        Mc = Mc.at[λ[i]].set(Mc_λi)

        Fi = Mc[i] @ S[i]
        M_ii = S[i].T @ Fi
        M = M.at[ii + 6, ii + 6].set(M_ii.squeeze())

        j = i

        CarryInnerFn = tuple[jtp.Int, jtp.Matrix, jtp.Matrix]
        carry_inner_fn = (j, Fi, M)

        def while_loop_body(carry: CarryInnerFn) -> CarryInnerFn:
            j, Fi, M = carry

            Fi = i_X_λi[j].T @ Fi
            j = λ[j]
            jj = j - 1

            M_ij = Fi.T @ S[j]

            M = M.at[ii + 6, jj + 6].set(M_ij.squeeze())
            M = M.at[jj + 6, ii + 6].set(M_ij.squeeze())

            return j, Fi, M

        # The following functions are part of a (rather messy) workaround for computing
        # a while loop using a for loop with fixed number of iterations.
        def inner_fn(carry: CarryInnerFn, k: jtp.Int) -> tuple[CarryInnerFn, None]:
            def compute_inner(carry: CarryInnerFn) -> tuple[CarryInnerFn, None]:
                j, _, _ = carry
                out = jax.lax.cond(
                    pred=(λ[j] > 0),
                    true_fun=while_loop_body,
                    false_fun=lambda carry: carry,
                    operand=carry,
                )
                return out, None

            j, _, _ = carry
            return jax.lax.cond(
                pred=(k == j),
                true_fun=compute_inner,
                false_fun=lambda carry: (carry, None),
                operand=carry,
            )

        (j, Fi, M), _ = (
            jax.lax.scan(
                f=inner_fn,
                init=carry_inner_fn,
                xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
            )
            if model.number_of_links() > 1
            else [(j, Fi, M), None]
        )

        Fi = i_X_0[j].T @ Fi

        M = M.at[0:6, ii + 6].set(Fi.squeeze())
        M = M.at[ii + 6, 0:6].set(Fi.squeeze())

        return (Mc, M), None

    # This scan performs the backward pass to compute Mbj, Mjb and Mjj, that
    # also includes a fake while loop implemented with a scan and two cond.
    (Mc, M), _ = (
        jax.lax.scan(
            f=backward_pass,
            init=backward_pass_carry,
            xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
        )
        if model.number_of_links() > 1
        else [(Mc, M), None]
    )

    # Store the locked 6D rigid-body inertia matrix Mbb ∈ ℝ⁶ˣ⁶.
    M = M.at[0:6, 0:6].set(Mc[0])

    return M
