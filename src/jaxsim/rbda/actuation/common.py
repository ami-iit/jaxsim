import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


class ActuationParams(JaxsimDataclass):
    """
    Parameters class for the actuation model.
    """

    τ_max: jtp.Float = 3000.0  # Max torque (Nm)
    ω_th: jtp.Float = 30.0  # Threshold speed (rad/s)
    ω_max: jtp.Float = 100.0  # Max speed for torque drop-off (rad/s)
