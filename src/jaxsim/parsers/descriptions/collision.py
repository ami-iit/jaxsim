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
    size: jtp.VectorLike
    parent_link: str

    def __hash__(self) -> int:
        return hash(
            (
                hash(tuple(self.center.tolist())),
                hash(tuple(self.size.tolist())),
                hash(self.parent_link),
            )
        )

    def __eq__(self, other: CollisionShape) -> bool:

        if not isinstance(other, CollisionShape):
            return False

        return hash(self) == hash(other)


@dataclasses.dataclass
class BoxCollision(CollisionShape):
    """
    Represents a box-shaped collision shape.
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

    @x.setter
    def x(self, value: float) -> None:
        self.size[0] = value

    @y.setter
    def y(self, value: float) -> None:
        self.size[1] = value

    @z.setter
    def z(self, value: float) -> None:
        self.size[2] = value


@dataclasses.dataclass
class SphereCollision(CollisionShape):
    """
    Represents a spherical collision shape.
    """

    @property
    def radius(self) -> float:
        return self.size[0]


@dataclasses.dataclass
class CylinderCollision(CollisionShape):
    """
    Represents a cylindrical collision shape.
    """

    @property
    def radius(self) -> float:
        return self.size[0]

    @property
    def height(self) -> float:
        return self.size[1]
