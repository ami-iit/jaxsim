import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def crba(model: PhysicsModel, q: jtp.Vector) -> jtp.Matrix:

    _, q, _, _, _, _ = utils.process_inputs(
        physics_model=model, xfb=None, q=q, qd=None, tau=None, f_ext=None
    )

    Xtree = model.tree_transforms
    Mc = model.spatial_inertias
    S = model.motion_subspaces(q=q)
    Xj = model.joint_transforms(q=q)

    Xup = jnp.zeros_like(Xtree)
    i_X_0 = jnp.zeros_like(Xtree)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    for i in np.arange(start=1, stop=model.NB):

        Xup_i = Xj[i] @ Xtree[i]
        Xup = Xup.at[i].set(Xup_i)

        i_X_0_i = Xup[i] @ i_X_0[model.parent[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

    M = jnp.zeros(shape=(6 + model.dofs(), 6 + model.dofs()))

    for i in reversed(np.arange(start=1, stop=model.NB)):

        Mc_λi = Mc[model.parent[i]] + Xup[i].T @ Mc[i] @ Xup[i]
        Mc = Mc.at[model.parent[i]].set(Mc_λi)

    for i in reversed(np.arange(start=1, stop=model.NB)):

        ii = i - 1

        Fi = Mc[i] @ S[i]
        M_ii = S[i].T @ Fi
        M = M.at[ii + 6, ii + 6].set(M_ii.squeeze())

        j = i

        while model._parent_array_dict[j] > 0:

            Fi = Xup[j].T @ Fi
            j = model._parent_array_dict[j]
            jj = j - 1

            M_ij = Fi.T @ S[j]

            M = M.at[ii + 6, jj + 6].set(M_ij.squeeze())
            M = M.at[jj + 6, ii + 6].set(M_ij.squeeze())

        Fi = i_X_0[j].T @ Fi

        M = M.at[0:6, ii + 6].set(Fi.squeeze())
        M = M.at[ii + 6, 0:6].set(Fi.squeeze())

    M = M.at[0:6, 0:6].set(Mc[0])

    return M
