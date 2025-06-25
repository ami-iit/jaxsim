from __future__ import annotations

import dataclasses


import jaxsim.typing as jtp


@dataclasses.dataclass
class BoxCollision:
    """
    Represents a box-shaped collision shape.

    Attributes:
        center: The center of the box in the local frame of the collision shape.
    """

    center: jtp.VectorLike
    x: float
    y: float
    z: float

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
class SphereCollision:
    """
    Represents a spherical collision shape.

    Attributes:
        center: The center of the sphere in the local frame of the collision shape.
    """

    center: jtp.VectorLike
    radius: float
    parent_link: str = ""

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