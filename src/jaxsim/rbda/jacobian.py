import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint

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
        The doubly-left free-floating Jacobian of the link.
    """

    _, _, s, _, _, _, _, _, _, _ = utils.process_inputs(
        model=model, joint_positions=joint_positions
    )

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

    PropagateKinematicsCarry = tuple[jtp.MatrixJax]
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

    (i_X_0,), _ = jax.lax.scan(
        f=propagate_kinematics,
        init=propagate_kinematics_carry,
        xs=np.arange(start=1, stop=model.number_of_links()),
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

    def compute_jacobian(J: jtp.MatrixJax, i: jtp.Int) -> tuple[jtp.MatrixJax, None]:

        def update_jacobian(J: jtp.MatrixJax, i: jtp.Int) -> jtp.MatrixJax:

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

    W_J_WL_W, _ = jax.lax.scan(
        f=compute_jacobian,
        init=J,
        xs=np.arange(start=1, stop=model.number_of_links()),
    )

    return W_J_WL_W
