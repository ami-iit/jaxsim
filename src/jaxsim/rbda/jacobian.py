import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Cross

from . import utils


def jacobian(
    model: js.model.JaxSimModel,
    *,
    link_index: jtp.Int,
    joint_positions: jtp.VectorLike,
) -> jtp.Matrix:
    """
    Compute the free-floating Jacobian of a link.

    Args:
        model: The model to consider.
        link_index: The index of the link for which to compute the Jacobian matrix.
        joint_positions: The positions of the joints.

    Returns:
        The free-floating left-trivialized Jacobian of the link :math:`{}^L J_{W,L/B}`.
    """

    _, _, s, _, _, _, _, _, _, _ = utils.process_inputs(
        model=model, joint_positions=joint_positions
    )

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = model.kin_dyn_parameters.joint_transforms(
        joint_positions=s, base_transform=jnp.eye(4)
    )

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate the buffer of transforms link -> base.
    i_X_0 = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # ====================
    # Propagate kinematics
    # ====================

    PropagateKinematicsCarry = tuple[jtp.Matrix]
    propagate_kinematics_carry: PropagateKinematicsCarry = (i_X_0,)

    def propagate_kinematics(
        carry: PropagateKinematicsCarry, i: jtp.Int
    ) -> tuple[PropagateKinematicsCarry, None]:

        (i_X_0,) = carry

        # Compute the base (0) to link (i) adjoint matrix.
        # This works fine since we traverse the kinematic tree following the link
        # indices assigned with BFS.
        i_X_0_i = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

        return (i_X_0,), None

    (i_X_0,), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=propagate_kinematics_carry,
            xs=np.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(i_X_0,), None]
    )

    # ============================
    # Compute doubly-left Jacobian
    # ============================

    J = jnp.zeros(shape=(6, 6 + model.dofs()))

    Jb = i_X_0[link_index]
    J = J.at[0:6, 0:6].set(Jb)

    # To make JIT happy, we operate on a boolean version of κ(i).
    # Checking if j ∈ κ(i) is equivalent to: κ_bool(j) is True.
    κ_bool = model.kin_dyn_parameters.support_body_array_bool[link_index]

    def compute_jacobian(J: jtp.Matrix, i: jtp.Int) -> tuple[jtp.Matrix, None]:

        def update_jacobian(J: jtp.Matrix, i: jtp.Int) -> jtp.Matrix:

            ii = i - 1

            Js_i = i_X_0[link_index] @ Adjoint.inverse(i_X_0[i]) @ S[i]
            J = J.at[0:6, 6 + ii].set(Js_i.squeeze())

            return J

        J = jax.lax.select(
            pred=κ_bool[i],
            on_true=update_jacobian(J, i),
            on_false=J,
        )

        return J, None

    L_J_WL_B, _ = (
        jax.lax.scan(
            f=compute_jacobian,
            init=J,
            xs=np.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [J, None]
    )

    return L_J_WL_B


@jax.jit
def jacobian_full_doubly_left(
    model: js.model.JaxSimModel,
    *,
    joint_positions: jtp.VectorLike,
) -> tuple[jtp.Matrix, jtp.Array]:
    r"""
    Compute the doubly-left full free-floating Jacobian of a model.

    The full Jacobian is a 6x(6+n) matrix with all the columns filled.
    It is useful to run the algorithm once, and then extract the link Jacobian by
    filtering the columns of the full Jacobian using the support parent array
    :math:`\kappa(i)` of the link.

    Args:
        model: The model to consider.
        joint_positions: The positions of the joints.

    Returns:
        The doubly-left full free-floating Jacobian of a model.
    """

    _, _, s, _, _, _, _, _, _, _ = utils.process_inputs(
        model=model, joint_positions=joint_positions
    )

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = model.kin_dyn_parameters.joint_transforms(
        joint_positions=s, base_transform=jnp.eye(4)
    )

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate the buffer of transforms base -> link.
    B_X_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    B_X_i = B_X_i.at[0].set(jnp.eye(6))

    # =================================
    # Compute doubly-left full Jacobian
    # =================================

    # Allocate the Jacobian matrix.
    # The Jbb section of the doubly-left Jacobian is an identity matrix.
    J = jnp.zeros(shape=(6, 6 + model.dofs()))
    J = J.at[0:6, 0:6].set(jnp.eye(6))

    ComputeFullJacobianCarry = tuple[jtp.Matrix, jtp.Matrix]
    compute_full_jacobian_carry: ComputeFullJacobianCarry = (B_X_i, J)

    def compute_full_jacobian(
        carry: ComputeFullJacobianCarry, i: jtp.Int
    ) -> tuple[ComputeFullJacobianCarry, None]:

        ii = i - 1
        B_X_i, J = carry

        # Compute the base (0) to link (i) adjoint matrix.
        B_Xi_i = B_X_i[λ[i]] @ Adjoint.inverse(i_X_λi[i])
        B_X_i = B_X_i.at[i].set(B_Xi_i)

        # Compute the ii-th column of the B_S_BL(s) matrix.
        B_Sii_BL = B_Xi_i @ S[i]
        J = J.at[0:6, 6 + ii].set(B_Sii_BL.squeeze())

        return (B_X_i, J), None

    (B_X_i, J), _ = (
        jax.lax.scan(
            f=compute_full_jacobian,
            init=compute_full_jacobian_carry,
            xs=np.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(B_X_i, J), None]
    )

    # Convert adjoints to SE(3) transforms.
    # Returning them here prevents calling FK in case the output representation
    # of the Jacobian needs to be changed.
    B_H_L = jax.vmap(Adjoint.to_transform)(B_X_i)

    # Adjust shape of doubly-left free-floating full Jacobian.
    B_J_full_WL_B = J.squeeze().astype(float)

    return B_J_full_WL_B, B_H_L


def jacobian_derivative_full_doubly_left(
    model: js.model.JaxSimModel,
    *,
    joint_positions: jtp.VectorLike,
    joint_velocities: jtp.VectorLike,
) -> tuple[jtp.Matrix, jtp.Array]:
    r"""
    Compute the derivative of the doubly-left full free-floating Jacobian of a model.

    The derivative of the full Jacobian is a 6x(6+n) matrix with all the columns filled.
    It is useful to run the algorithm once, and then extract the link Jacobian
    derivative by filtering the columns of the full Jacobian using the support
    parent array :math:`\kappa(i)` of the link.

    Args:
        model: The model to consider.
        joint_positions: The positions of the joints.
        joint_velocities: The velocities of the joints.

    Returns:
        The derivative of the doubly-left full free-floating Jacobian of a model.
    """

    _, _, s, _, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model, joint_positions=joint_positions, joint_velocities=joint_velocities
    )

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = model.kin_dyn_parameters.joint_transforms(
        joint_positions=s, base_transform=jnp.eye(4)
    )

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate the buffer of 6D transform base -> link.
    B_X_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    B_X_i = B_X_i.at[0].set(jnp.eye(6))

    # Allocate the buffer of 6D transform derivatives base -> link.
    B_Ẋ_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))

    # Allocate the buffer of the 6D link velocity in body-fixed representation.
    B_v_Bi = jnp.zeros(shape=(model.number_of_links(), 6))

    # Helper to compute the time derivative of the adjoint matrix.
    def A_Ẋ_B(A_X_B: jtp.Matrix, B_v_AB: jtp.Vector) -> jtp.Matrix:
        return A_X_B @ Cross.vx(B_v_AB).squeeze()

    # ============================================
    # Compute doubly-left full Jacobian derivative
    # ============================================

    # Allocate the Jacobian matrix.
    J̇ = jnp.zeros(shape=(6, 6 + model.dofs()))

    ComputeFullJacobianDerivativeCarry = tuple[
        jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix
    ]

    compute_full_jacobian_derivative_carry: ComputeFullJacobianDerivativeCarry = (
        B_v_Bi,
        B_X_i,
        B_Ẋ_i,
        J̇,
    )

    def compute_full_jacobian_derivative(
        carry: ComputeFullJacobianDerivativeCarry, i: jtp.Int
    ) -> tuple[ComputeFullJacobianDerivativeCarry, None]:

        ii = i - 1
        B_v_Bi, B_X_i, B_Ẋ_i, J̇ = carry

        # Compute the base (0) to link (i) adjoint matrix.
        B_Xi_i = B_X_i[λ[i]] @ Adjoint.inverse(i_X_λi[i])
        B_X_i = B_X_i.at[i].set(B_Xi_i)

        # Compute the body-fixed velocity of the link.
        B_vi_Bi = B_v_Bi[λ[i]] + B_X_i[i] @ S[i].squeeze() * ṡ[ii]
        B_v_Bi = B_v_Bi.at[i].set(B_vi_Bi)

        # Compute the base (0) to link (i) adjoint matrix derivative.
        i_Xi_B = Adjoint.inverse(B_Xi_i)
        B_Ẋi_i = A_Ẋ_B(A_X_B=B_Xi_i, B_v_AB=i_Xi_B @ B_vi_Bi)
        B_Ẋ_i = B_Ẋ_i.at[i].set(B_Ẋi_i)

        # Compute the ii-th column of the B_Ṡ_BL(s) matrix.
        B_Ṡii_BL = B_Ẋ_i[i] @ S[i]
        J̇ = J̇.at[0:6, 6 + ii].set(B_Ṡii_BL.squeeze())

        return (B_v_Bi, B_X_i, B_Ẋ_i, J̇), None

    (_, B_X_i, B_Ẋ_i, J̇), _ = (
        jax.lax.scan(
            f=compute_full_jacobian_derivative,
            init=compute_full_jacobian_derivative_carry,
            xs=np.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(_, B_X_i, B_Ẋ_i, J̇), None]
    )

    # Convert adjoints to SE(3) transforms.
    # Returning them here prevents calling FK in case the output representation
    # of the Jacobian needs to be changed.
    B_H_L = jax.vmap(Adjoint.to_transform)(B_X_i)

    # Adjust shape of doubly-left free-floating full Jacobian derivative.
    B_J̇_full_WL_B = J̇.squeeze().astype(float)

    return B_J̇_full_WL_B, B_H_L
