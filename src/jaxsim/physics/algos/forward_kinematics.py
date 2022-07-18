import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.plucker import Plucker
from jaxsim.math.quaternion import Quaternion
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def forward_kinematics_model(
    model: PhysicsModel, q: jtp.Vector, xfb: jtp.Vector
) -> jtp.Matrix:

    x_fb, q, _, _, _, _ = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=None, tau=None, f_ext=None
    )

    qn = jnp.vstack(x_fb[0:4])
    r = jnp.vstack(x_fb[4:7])
    W_X_0 = jnp.linalg.inv(Plucker.from_rot_and_trans(Quaternion.to_dcm(qn), r))

    W_X_i = jnp.zeros(shape=[model.NB, 6, 6])
    W_X_i = W_X_i.at[0].set(W_X_0)

    i_X_pre = model.joint_transforms(q=q)
    pre_X_λi = model.tree_transforms
    i_X_λi = jnp.zeros_like(i_X_pre)

    for i in np.arange(start=1, stop=model.NB):

        i_X_λi_i = i_X_pre[i] @ pre_X_λi[i]
        i_X_λi = i_X_λi.at[i].set(i_X_λi_i)

        W_X_i_i = W_X_i[model.parent[i]] @ jnp.linalg.inv(i_X_λi[i])
        W_X_i = W_X_i.at[i].set(W_X_i_i)

    return jnp.stack([Plucker.to_transform(X) for X in list(W_X_i)])


def forward_kinematics(
    model: PhysicsModel, body_index: int, q: jtp.Vector, xfb: jtp.Vector
) -> jtp.Matrix:

    return forward_kinematics_model(model=model, q=q, xfb=xfb)[body_index]
