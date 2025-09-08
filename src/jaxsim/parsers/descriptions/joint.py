from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax_dataclasses
import numpy as np

import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass, Mutability

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
    axis: jtp.Vector

    def __hash__(self) -> int:

        return hash(tuple(self.axis.tolist()))

    def __eq__(self, other: JointGenericAxis) -> bool:

        if not isinstance(other, JointGenericAxis):
            return False

        return hash(self) == hash(other)


@jax_dataclasses.pytree_dataclass(eq=False, unsafe_hash=False)
class JointDescription(JaxsimDataclass):
    """
    In-memory description of a robot link.

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

    name: jax_dataclasses.Static[str]
    axis: jtp.Vector
    pose: jtp.Matrix
    jtype: jax_dataclasses.Static[jtp.IntLike]
    child: LinkDescription = dataclasses.dataclass(repr=False)
    parent: LinkDescription = dataclasses.dataclass(repr=False)

    index: jtp.IntLike | None = None

    friction_static: jtp.FloatLike = 0.0
    friction_viscous: jtp.FloatLike = 0.0

    position_limit_damper: jtp.FloatLike = 0.0
    position_limit_spring: jtp.FloatLike = 0.0

    position_limit: tuple[jtp.FloatLike, jtp.FloatLike] = (0.0, 0.0)
    initial_position: jtp.FloatLike | jtp.VectorLike = 0.0

    motor_inertia: jtp.FloatLike = 0.0
    motor_viscous_friction: jtp.FloatLike = 0.0
    motor_gear_ratio: jtp.FloatLike = 1.0

    def __post_init__(self) -> None:

        if self.axis is not None:

            with self.mutable_context(
                mutability=Mutability.MUTABLE, restore_after_exception=False
            ):
                norm_of_axis = np.linalg.norm(self.axis)
                self.axis = (
                    self.axis / norm_of_axis if norm_of_axis != 0.0 else self.axis
                )

    def __eq__(self, other: JointDescription) -> bool:

        if not isinstance(other, JointDescription):
            return False

        return hash(self) == hash(other)

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                hash(self.name),
                HashedNumpyArray.hash_of_array(self.axis),
                HashedNumpyArray.hash_of_array(self.pose),
                hash(int(self.jtype)),
                hash(self.child),
                hash(self.parent),
                hash(int(self.index)) if self.index is not None else 0,
                HashedNumpyArray.hash_of_array(self.friction_static),
                HashedNumpyArray.hash_of_array(self.friction_viscous),
                HashedNumpyArray.hash_of_array(self.position_limit_damper),
                HashedNumpyArray.hash_of_array(self.position_limit_spring),
                HashedNumpyArray.hash_of_array(self.position_limit),
                HashedNumpyArray.hash_of_array(self.initial_position),
                HashedNumpyArray.hash_of_array(self.motor_inertia),
                HashedNumpyArray.hash_of_array(self.motor_viscous_friction),
                HashedNumpyArray.hash_of_array(self.motor_gear_ratio),
            ),
        )
