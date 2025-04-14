import dataclasses

import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class ActuationParams(JaxsimDataclass):
    """
    Parameters class for the actuation model.
    """

    torque_max: jtp.Float = dataclasses.field(default=3000.0)  # (Nm)
    omega_th: jtp.Float = dataclasses.field(default=30.0)  # (rad/s)
    omega_max: jtp.Float = dataclasses.field(default=100.0)  # (rad/s)
