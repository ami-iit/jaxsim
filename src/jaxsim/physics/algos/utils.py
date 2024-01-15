from typing import Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel


def process_inputs(
    physics_model: PhysicsModel,
    xfb: jtp.Vector | None = None,
    q: jtp.Vector | None = None,
    qd: jtp.Vector | None = None,
    qdd: jtp.Vector | None = None,
    tau: jtp.Vector | None = None,
    f_ext: jtp.Matrix | None = None,
) -> Tuple[jtp.Vector, jtp.Vector, jtp.Vector, jtp.Vector, jtp.Vector, jtp.Matrix]:
    """
    Adjust the inputs to the physics model.

    Args:
        physics_model: The physics model.
        xfb: The variables of the base link.
        q: The generalized coordinates.
        qd: The generalized velocities.
        qdd: The generalized accelerations.
        tau: The generalized forces.
        f_ext: The link external forces.

    Returns:
        The adjusted inputs.
    """

    # Remove extra dimensions
    q = q.squeeze() if q is not None else jnp.zeros(physics_model.dofs())
    qd = qd.squeeze() if qd is not None else jnp.zeros(physics_model.dofs())
    qdd = qdd.squeeze() if qdd is not None else jnp.zeros(physics_model.dofs())
    tau = tau.squeeze() if tau is not None else jnp.zeros(physics_model.dofs())
    xfb = xfb.squeeze() if xfb is not None else jnp.zeros(13).at[0].set(1)
    f_ext = (
        f_ext.squeeze()
        if f_ext is not None
        else jnp.zeros(shape=(physics_model.NB, 6)).squeeze()
    )

    # Fix case with just 1 DoF
    q = jnp.atleast_1d(q)
    qd = jnp.atleast_1d(qd)
    qdd = jnp.atleast_1d(qdd)
    tau = jnp.atleast_1d(tau)

    # Fix case with just 1 body
    f_ext = jnp.atleast_2d(f_ext)

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
