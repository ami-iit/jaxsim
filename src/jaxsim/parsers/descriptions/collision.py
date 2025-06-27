from __future__ import annotations

import dataclasses
from abc import ABC

import jaxsim.typing as jtp

@dataclasses.dataclass
class CollisionShape(ABC):
    """
    Base class for collision shapes.

    This class serves as a base for specific collision shapes like BoxCollision and SphereCollision.
    It is not intended to be instantiated directly.
    """

    center: jtp.VectorLike
    parent_link: str
    size: jtp.VectorLike

@dataclasses.dataclass
class BoxCollision(CollisionShape):
    """
    Represents a box-shaped collision shape.

    Attributes:
        center: The center of the box in the local frame of the collision shape.
    """

    @property
    def x(self) -> float:
        return self.size[0]

    @property
    def y(self) -> float:
        return self.size[1]

    @property
    def z(self) -> float:
        return self.size[2]

    def __hash__(self) -> int:
        return hash(
            (
                hash(super()),
                hash(tuple(self.center.tolist())),
            )
        )

    def __eq__(self, other: BoxCollision) -> bool:

        if not isinstance(other, BoxCollision):
            return False

        return hash(self) == hash(other)


@dataclasses.dataclass
class SphereCollision(CollisionShape):
    """
    Represents a spherical collision shape.

    Attributes:
        center: The center of the sphere in the local frame of the collision shape.
    """

    def radius(self) -> float:
        return self.size[0]

    def __hash__(self) -> int:
        return hash(
            (
                hash(super()),
                hash(tuple(self.center.tolist())),
            )
        )

    def __eq__(self, other: SphereCollision) -> bool:

        if not isinstance(other, SphereCollision):
            return False

        return hash(self) == hash(other)