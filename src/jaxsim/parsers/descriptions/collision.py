from __future__ import annotations

import dataclasses
from abc import ABC

import numpy as np

import jaxsim.typing as jtp


@dataclasses.dataclass
class CollisionShape(ABC):
    """
    Base class for collision shapes.

    This class serves as a base for specific collision shapes like BoxCollision and SphereCollision.
    It is not intended to be instantiated directly.
    """

    size: jtp.VectorLike
    parent_link: str
    transform: jtp.MatrixLike = dataclasses.field(default_factory=lambda: np.eye(4))

    def __hash__(self) -> int:
        return hash(
            (
                hash(tuple(self.size.tolist())),
                hash(self.parent_link),
                hash(tuple(self.transform.flatten().tolist())),
            )
        )

    def __eq__(self, other: CollisionShape) -> bool:

        if not isinstance(other, CollisionShape):
            return False

        return hash(self) == hash(other)

    @property
    def center(self) -> jtp.Vector:
        """Extract the translation from the transformation matrix."""
        return self.transform[:3, 3]

    @property
    def orientation(self) -> jtp.Matrix:
        """Extract the rotation matrix from the transformation matrix."""
        return self.transform[:3, :3]


@dataclasses.dataclass
class BoxCollision(CollisionShape):
    """
    Represents a box-shaped collision shape.
    """


@dataclasses.dataclass
class SphereCollision(CollisionShape):
    """
    Represents a spherical collision shape.
    """


@dataclasses.dataclass
class CylinderCollision(CollisionShape):
    """
    Represents a cylindrical collision shape.
    """
