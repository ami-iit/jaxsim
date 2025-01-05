from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .link import LinkDescription


@dataclasses.dataclass(frozen=True)
class JointType:
    """
    Enumeration of joint types.
    """

    Fixed: ClassVar[int] = 0
    Revolute: ClassVar[int] = 1
    Prismatic: ClassVar[int] = 2


@dataclasses.dataclass(eq=False)
class JointDescription:
    """
    In-memory description of a robot joint.

    Attributes:
        name: The name of the joint.
        axis: The axis of rotation or translation for the joint.
        pose: The pose transformation matrix of the joint.
        jtype: The type of the joint.
        child: The child link attached to the joint.
        parent: The parent link attached to the joint.
        index: An optional index for the joint.
        friction_static: The static friction coefficient for the joint.
        friction_viscous: The viscous friction coefficient for the joint.
        position_limit_damper: The damper coefficient for position limits.
        position_limit_spring: The spring coefficient for position limits.
        position_limit: The position limits for the joint.
        initial_position: The initial position of the joint.
    """

    name: str
    _axis: tuple[float]
    _pose: tuple[float]
    jtype: int
    child: LinkDescription = dataclasses.field(
        default_factory=LinkDescription, repr=False
    )
    parent: LinkDescription = dataclasses.field(
        default_factory=LinkDescription, repr=False
    )

    index: int | None = None

    friction_static: float = 0.0
    friction_viscous: float = 0.0

    position_limit_damper: float = 0.0
    position_limit_spring: float = 0.0

    position_limit: tuple[float, float] = (0.0, 0.0)
    _initial_position: float | tuple[float] = 0.0

    motor_inertia: float = 0.0
    motor_viscous_friction: float = 0.0
    motor_gear_ratio: float = 1.0

    def __post_init__(self) -> None:

        if self._axis is not None:

            self._axis = self.axis / np.linalg.norm(self.axis)

    @property
    def axis(self) -> npt.NDArray:
        """
        Get the axis of the joint.

        Returns:
            npt.NDArray: The axis of the joint.
        """

        return np.array(self._axis)

    @axis.setter
    def axis(self, value: npt.NDArray) -> None:
        """
        Set the axis of the joint.

        Args:
            value (npt.NDArray): The new axis of the joint.
        """

        norm_of_axis = np.linalg.norm(value)
        self._axis = tuple((value / norm_of_axis).tolist())

    @property
    def pose(self) -> npt.NDArray:
        """
        Get the pose of the joint.

        Returns:
            The pose of the joint.
        """

        return np.array(self._pose, dtype=float)

    @pose.setter
    def pose(self, value: npt.NDArray) -> None:
        """
        Set the pose of the joint.

        Args:
            value: The new pose of the joint.
        """

        self._pose = tuple(np.array(value).tolist())

    @property
    def initial_position(self) -> float | npt.NDArray:
        """
        Get the initial position of the joint.

        Returns:
            The initial position of the joint.
        """

        return np.array(self._initial_position, dtype=float)

    @initial_position.setter
    def initial_position(self, value: float | npt.NDArray) -> None:
        """
        Set the initial position of the joint.

        Args:
            value: The new initial position of the joint.
        """

        self._initial_position = tuple(np.array(value).tolist())
