import abc
import dataclasses
from typing import List

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import jaxsim.logging as logging

from .link import LinkDescription


@dataclasses.dataclass
class CollidablePoint:
    parent_link: LinkDescription
    position: npt.NDArray = dataclasses.field(default_factory=lambda: np.zeros(3))
    enabled: bool = True

    def change_link(
        self, new_link: LinkDescription, new_H_old: npt.NDArray
    ) -> "CollidablePoint":
        msg = f"Moving collidable point: {self.parent_link.name} -> {new_link.name}"
        logging.debug(msg=msg)

        return CollidablePoint(
            parent_link=new_link,
            position=(new_H_old @ jnp.hstack([self.position, 1.0])).squeeze()[0:3],
            enabled=self.enabled,
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            + f"parent_link={self.parent_link.name}"
            + f", position={self.position}"
            + f", enabled={self.enabled}"
            + ")"
        )


@dataclasses.dataclass
class CollisionShape(abc.ABC):
    collidable_points: List[CollidablePoint]

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            + "collidable_points=[\n    "
            + ",\n    ".join(str(cp) for cp in self.collidable_points)
            + "\n])"
        )


@dataclasses.dataclass
class BoxCollision(CollisionShape):
    center: npt.NDArray


@dataclasses.dataclass
class SphereCollision(CollisionShape):
    center: npt.NDArray
