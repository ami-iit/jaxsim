from typing import Optional, Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel


def process_inputs(
    physics_model: PhysicsModel,
    xfb: jtp.Vector = None,
    q: jtp.Vector = None,
    qd: jtp.Vector = None,
    qdd: jtp.Vector = None,
    tau: jtp.Vector = None,
    f_ext: jtp.Matrix = None,
) -> Tuple[jtp.Vector, jtp.Vector, jtp.Vector, jtp.Vector, jtp.Vector, jtp.Matrix]:
    # Remove extra dimensions
    q = q.squeeze() if q is not None else None
    qd = qd.squeeze() if qd is not None else None
    qdd = qdd.squeeze() if qdd is not None else None
    tau = tau.squeeze() if tau is not None else None
    xfb = xfb.squeeze() if xfb is not None else None
    f_ext = f_ext.squeeze() if f_ext is not None else None

    def fix_one_dof(vector: jtp.Vector) -> Optional[jtp.Vector]:
        if vector is None:
            return None

        return jnp.array([vector]) if vector.shape == () else vector

    # Fix case with just 1 DoF
    q = fix_one_dof(q)
    qd = fix_one_dof(qd)
    qdd = fix_one_dof(qdd)
    tau = fix_one_dof(tau)

    # Build matrix of external forces if not given
    f_ext = f_ext if f_ext is not None else jnp.zeros(shape=(physics_model.NB, 6))

    # Fix case with just 1 body
    f_ext = f_ext if physics_model.NB != 1 else f_ext.squeeze()[jnp.newaxis, ...]

    # Validate dimensions
    dofs = physics_model.dofs()

    if xfb is not None and xfb.shape[0] != 13:
        raise ValueError(xfb.shape)
    if q is not None and q.shape[0] != dofs:
        raise ValueError(q.shape, dofs)
    if qd is not None and qd.shape[0] != dofs:
        raise ValueError(qd.shape, dofs)
    if tau is not None and tau.shape[0] != dofs:
        raise ValueError(tau.shape, dofs)
    if f_ext is not None and f_ext.shape != (physics_model.NB, 6):
        raise ValueError(f_ext.shape, (physics_model.NB, 6))

    return xfb, q, qd, qdd, tau, f_ext
