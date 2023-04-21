import enum


class VelRepr(enum.IntEnum):
    """
    Enumeration of all supported 6D velocity representations.
    """

    Body = enum.auto()
    Mixed = enum.auto()
    Inertial = enum.auto()
