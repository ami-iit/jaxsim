import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def jacobian(
    model: PhysicsModel,
    body_index: int,
    q: jtp.Vector,
) -> jtp.Matrix:

    _, q, _, _, _, _ = utils.process_inputs(physics_model=model, q=q)

    S = model.motion_subspaces(q=q)
    i_X_pre = model.joint_transforms(q=q)
    pre_X_λi = model.tree_transforms
    i_X_λi = jnp.zeros_like(i_X_pre)

    i_X_0 = jnp.zeros_like(i_X_pre)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    for i in np.arange(start=1, stop=model.NB):

        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        i_X_0_i = i_X_λi[i] @ i_X_0[model.parent[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

    J = jnp.zeros(shape=(6, 6 + model.dofs()))

    Jb = i_X_0[body_index]
    J = J.at[0:6, 0:6].set(Jb)

    for i in reversed(model.support_body_array(body_index=body_index)):

        ii = i - 1

        if i == 0:
            break

        Js_i = i_X_0[body_index] @ jnp.linalg.inv(i_X_0[i]) @ S[i]
        J = J.at[0:6, 6 + ii].set(Js_i.squeeze())

    return J
