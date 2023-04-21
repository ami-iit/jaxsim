from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.cross import Cross
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


def rnea(
    model: PhysicsModel,
    xfb: jtp.Vector,
    q: jtp.Vector,
    qd: jtp.Vector,
    qdd: jtp.Vector,
    a0fb: jtp.Vector = jnp.zeros(6),
    f_ext: jtp.Matrix = None,
) -> Tuple[jtp.Vector, jtp.Vector]:
    xfb, q, qd, qdd, _, f_ext = utils.process_inputs(
        physics_model=model, xfb=xfb, q=q, qd=qd, qdd=qdd, f_ext=f_ext
    )

    a0fb = a0fb.squeeze()
    gravity = model.gravity.squeeze()

    if a0fb.shape[0] != 6:
        raise ValueError(a0fb.shape)

    I = model.spatial_inertias
    Xtree = model.tree_transforms
    Xj = model.joint_transforms(q=q)
    S = model.motion_subspaces(q=q)
    Xup = jnp.zeros_like(Xtree)

    i_X_0 = jnp.zeros_like(Xtree)
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    v: Dict[int, jtp.VectorJax] = dict()
    a: Dict[int, jtp.VectorJax] = dict()
    f: Dict[int, jtp.VectorJax] = dict()

    # 6D transform of base velocity
    Xup_0 = B_X_W = Adjoint.from_quaternion_and_translation(
        quaternion=xfb[0:4],
        translation=xfb[4:7],
        inverse=True,
        normalize_quaternion=True,
    )
    Xup = Xup.at[0].set(Xup_0)

    v[0] = jnp.zeros(shape=(6, 1))
    f[0] = jnp.zeros(shape=(6, 1))
    a[0] = -B_X_W @ jnp.vstack(gravity)

    if model.is_floating_base:
        W_v_WB = jnp.vstack(jnp.hstack([xfb[10:13], xfb[7:10]]))
        v[0] = B_X_W @ W_v_WB

        a[0] = B_X_W @ (jnp.vstack(a0fb) - jnp.vstack(gravity))
        f[0] = (
            I[0] @ a[0]
            + Cross.vx_star(v[0]) @ I[0] @ v[0]
            - jnp.linalg.inv(B_X_W).T @ jnp.vstack(f_ext[0])
        )

    for i in np.arange(start=1, stop=model.NB):
        ii = i - 1

        vJ = S[i] * qd[ii]
        Xup_i = Xj[i] @ Xtree[i]
        Xup = Xup.at[i].set(Xup_i)

        λi = model._parent_array_dict[i]
        v[i] = Xup[i] @ v[λi] + vJ
        a[i] = Xup[i] @ a[λi] + S[i] * qdd[ii] + Cross.vx(v[i]) @ vJ

        i_X_0_i = Xup[i] @ i_X_0[λi]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)
        i_X_W = jnp.linalg.inv(i_X_0[i] @ B_X_W).T

        f[i] = (
            I[i] @ a[i]
            + Cross.vx_star(v[i]) @ I[i] @ v[i]
            - i_X_W @ jnp.vstack(f_ext[i])
        )

    tau = jnp.zeros_like(q)

    for i in reversed(np.arange(start=1, stop=model.NB)):
        ii = i - 1

        value = S[i].T @ f[i]
        tau = tau.at[ii].set(value.squeeze())

        λi = model._parent_array_dict[i]

        if λi != 0 or model.is_floating_base:
            f[λi] = f[λi] + Xup[i].T @ f[i]

    # Handle 1 DoF models
    tau = jnp.atleast_1d(tau.squeeze())
    tau = jnp.vstack(tau) if tau.size > 0 else jnp.empty(shape=(0, 1))

    W_f0 = B_X_W.T @ jnp.vstack(f[0])

    return W_f0, tau
