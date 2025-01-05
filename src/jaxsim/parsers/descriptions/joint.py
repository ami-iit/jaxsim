from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax_dataclasses
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


@jax_dataclasses.pytree_dataclass
class JointGenericAxis:
    """
    A joint requiring the specification of a 3D axis.
    """

    # The axis of rotation or translation of the joint (must have norm 1).
    axis: npt.NDArray


@dataclasses.dataclass(eq=False, unsafe_hash=False)
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
    axis: npt.NDArray
    pose: npt.NDArray
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
    initial_position: float | npt.NDArray = 0.0

    motor_inertia: float = 0.0
    motor_viscous_friction: float = 0.0
    motor_gear_ratio: float = 1.0

    def __post_init__(self) -> None:

        if self.axis is not None:

            norm_of_axis = np.linalg.norm(self.axis)
            super().__setattr__("axis", self.axis / norm_of_axis)
