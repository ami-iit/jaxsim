import dataclasses
import enum
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .link import LinkDescription


class JointType(enum.IntEnum):
    F = enum.auto()  # Fixed
    R = enum.auto()  # Revolute
    P = enum.auto()  # Prismatic

    # Revolute joints, single axis
    Rx = enum.auto()
    Ry = enum.auto()
    Rz = enum.auto()

    # Prismatic joints, single axis
    Px = enum.auto()
    Py = enum.auto()
    Pz = enum.auto()


@dataclasses.dataclass
class JointDescriptor:
    code: JointType

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclasses.dataclass
class JointGenericAxis(JointDescriptor):
    axis: npt.NDArray

    def __post_init__(self):
        if np.allclose(self.axis, 0.0):
            raise ValueError(self.axis)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.axis, other.axis)

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclasses.dataclass
class JointDescription:
    name: str
    axis: npt.NDArray
    pose: npt.NDArray
    jtype: Union[JointType, JointDescriptor]
    child: LinkDescription = dataclasses.dataclass(repr=False)
    parent: LinkDescription = dataclasses.dataclass(repr=False)

    index: Optional[int] = None

    friction_static: float = 0.0
    friction_viscous: float = 0.0

    position_limit_damper: float = 0.0
    position_limit_spring: float = 0.0

    position_limit: Tuple[float, float] = (0.0, 0.0)
    initial_position: Union[float, npt.NDArray] = 0.0

    def __post_init__(self):
        if self.axis is not None:
            norm_of_axis = np.linalg.norm(self.axis)
            self.axis = self.axis / norm_of_axis

    def __hash__(self) -> int:
        return hash(self.__repr__())
