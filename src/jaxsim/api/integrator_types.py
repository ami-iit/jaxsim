from enum import auto, Enum


class IntegratorType(Enum):
    """The type of integrator to use for the dynamics."""

    SEMI_IMPLICIT = auto()
    HEUN2 = auto()
