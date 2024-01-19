import dataclasses
import enum
from typing import Tuple, Union

import jax_dataclasses
import numpy as np
import numpy.typing as npt

from jaxsim.utils import JaxsimDataclass, Mutability

from .link import LinkDescription


class JointType(enum.IntEnum):
    """
    Enumeration of joint types for robot joints.

    Args:
        F: Fixed joint (no movement).
        R: Revolute joint (rotation).
        P: Prismatic joint (translation).
        Rx: Revolute joint with rotation about the X-axis.
        Ry: Revolute joint with rotation about the Y-axis.
        Rz: Revolute joint with rotation about the Z-axis.
        Px: Prismatic joint with translation along the X-axis.
        Py: Prismatic joint with translation along the Y-axis.
        Pz: Prismatic joint with translation along the Z-axis.
    """

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
    """
    Description of a joint type with a specific code.

    Args:
        code (JointType): The code representing the joint type.

    """

    code: JointType

    def __hash__(self) -> int:
        return hash(self.__repr__())


@dataclasses.dataclass
class JointGenericAxis(JointDescriptor):
    """
    Description of a joint type with a generic axis.

    Attributes:
        axis (npt.NDArray): The axis of rotation or translation for the joint.

    """

    axis: npt.NDArray

    def __post_init__(self):
        if np.allclose(self.axis, 0.0):
            raise ValueError(self.axis)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(self.axis, other.axis)

    def __hash__(self) -> int:
        return hash(self.__repr__())


@jax_dataclasses.pytree_dataclass
class JointDescription(JaxsimDataclass):
    """
    In-memory description of a robot link.

    Attributes:
        name (str): The name of the joint.
        axis (npt.NDArray): The axis of rotation or translation for the joint.
        pose (npt.NDArray): The pose transformation matrix of the joint.
        jtype (Union[JointType, JointDescriptor]): The type of the joint.
        child (LinkDescription): The child link attached to the joint.
        parent (LinkDescription): The parent link attached to the joint.
        index (Optional[int]): An optional index for the joint.
        friction_static (float): The static friction coefficient for the joint.
        friction_viscous (float): The viscous friction coefficient for the joint.
        position_limit_damper (float): The damper coefficient for position limits.
        position_limit_spring (float): The spring coefficient for position limits.
        position_limit (Tuple[float, float]): The position limits for the joint.
        initial_position (Union[float, npt.NDArray]): The initial position of the joint.

    """

    name: jax_dataclasses.Static[str]
    axis: npt.NDArray
    pose: npt.NDArray
    jtype: jax_dataclasses.Static[Union[JointType, JointDescriptor]]
    child: LinkDescription = dataclasses.dataclass(repr=False)
    parent: LinkDescription = dataclasses.dataclass(repr=False)

    index: int | None = None

    friction_static: float = 0.0
    friction_viscous: float = 0.0

    position_limit_damper: float = 0.0
    position_limit_spring: float = 0.0

    position_limit: Tuple[float, float] = (0.0, 0.0)
    initial_position: Union[float, npt.NDArray] = 0.0

    motor_inertia: float = 0.0
    motor_viscous_friction: float = 0.0
    motor_gear_ratio: float = 1.0

    def __post_init__(self):
        if self.axis is not None:
            with self.mutable_context(
                mutability=Mutability.MUTABLE, restore_after_exception=False
            ):
                norm_of_axis = np.linalg.norm(self.axis)
                self.axis = self.axis / norm_of_axis

    def __hash__(self) -> int:
        return hash(self.__repr__())
