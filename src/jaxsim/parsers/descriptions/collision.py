from __future__ import annotations

import abc
import dataclasses

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import jaxsim.typing as jtp
from jaxsim import logging

from .link import LinkDescription


@dataclasses.dataclass
class CollidablePoint:
    """
    Represents a collidable point associated with a parent link.

    Attributes:
        parent_link: The parent link to which the collidable point is attached.
        position: The position of the collidable point relative to the parent link.
        enabled: A flag indicating whether the collidable point is enabled for collision detection.

    """

    parent_link: LinkDescription
    position: npt.NDArray = dataclasses.field(default_factory=lambda: np.zeros(3))
    enabled: bool = True

    def change_link(
        self, new_link: LinkDescription, new_H_old: npt.NDArray
    ) -> CollidablePoint:
        """
        Move the collidable point to a new parent link.

        Args:
            new_link (LinkDescription): The new parent link to which the collidable point is moved.
            new_H_old (npt.NDArray): The transformation matrix from the new link's frame to the old link's frame.

        Returns:
            CollidablePoint: A new collidable point associated with the new parent link.
        """

        msg = f"Moving collidable point: {self.parent_link.name} -> {new_link.name}"
        logging.debug(msg=msg)

        return CollidablePoint(
            parent_link=new_link,
            position=(new_H_old @ jnp.hstack([self.position, 1.0])).squeeze()[0:3],
            enabled=self.enabled,
        )

    def __hash__(self) -> int:

        return hash(
            (
                hash(self.parent_link),
                hash(tuple(self.position.tolist())),
                hash(self.enabled),
            )
        )

    def __eq__(self, other: CollidablePoint) -> bool:

        if not isinstance(other, CollidablePoint):
            return False

        return hash(self) == hash(other)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + f"parent_link={self.parent_link.name}"
            + f", position={self.position}"
            + f", enabled={self.enabled}"
            + ")"
        )


@dataclasses.dataclass
class CollisionShape(abc.ABC):
    """
    Abstract base class for representing collision shapes.

    Attributes:
        collidable_points: A list of collidable points associated with the collision shape.

    """

    collidable_points: tuple[CollidablePoint]

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            + "collidable_points=[\n    "
            + ",\n    ".join(str(cp) for cp in self.collidable_points)
            + "\n])"
        )


@dataclasses.dataclass
class BoxCollision(CollisionShape):
    """
    Represents a box-shaped collision shape.

    Attributes:
        center: The center of the box in the local frame of the collision shape.

    """

    center: jtp.VectorLike

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

    center: jtp.VectorLike

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
