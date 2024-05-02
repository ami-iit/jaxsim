from __future__ import annotations

import dataclasses
import enum
from typing import Tuple, Union

import jax_dataclasses
import numpy as np
import numpy.typing as npt

import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass, Mutability

from .link import LinkDescription


@enum.unique
class JointType(enum.IntEnum):
    """
    Type of supported joints.
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        # Start auto Enum value from 0 instead of 1
        return count

    #: Fixed joint.
    F = enum.auto()

    #: Revolute joint (1 DoF around axis).
    R = enum.auto()

    #: Prismatic joint (1 DoF along axis).
    P = enum.auto()


@jax_dataclasses.pytree_dataclass
class JointDescriptor:
    """
    Base class for joint types requiring to store additional metadata.
    """

    #: The joint type.
    joint_type: JointType


@jax_dataclasses.pytree_dataclass
class JointGenericAxis(JointDescriptor):
    """
    A joint requiring the specification of a 3D axis.
    """

    #: The axis of rotation or translation of the joint (must have norm 1).
    axis: jtp.Vector

    def __hash__(self) -> int:
        return hash((self.joint_type, tuple(np.array(self.axis).tolist())))

    def __eq__(self, other: JointGenericAxis) -> bool:
        if not isinstance(other, JointGenericAxis):
            return False

        return hash(self) == hash(other)


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
