import dataclasses
from typing import ClassVar

from jax_dataclasses._copy_and_mutate import _Mutability as Mutability

from .jaxsim_dataclass import JaxsimDataclass
from .tracing import not_tracing, tracing
from .wrappers import HashedNumpyArray, HashlessObject


# TODO (flferretti): Definitely not the best place for this
@dataclasses.dataclass(frozen=True)
class CollidableShapeType:
    """
    Enum representing the types of collidable shapes.
    """

    Unsupported: ClassVar[int] = -1
    Box: ClassVar[int] = 0
    Cylinder: ClassVar[int] = 1
    Sphere: ClassVar[int] = 2
