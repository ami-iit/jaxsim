from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Generic, TypeVar

import jax
import jax_dataclasses
import numpy as np
import numpy.typing as npt

T = TypeVar("T")


@dataclasses.dataclass
class HashlessObject(Generic[T]):
    """
    A class that wraps an object and makes it hashless.

    This is useful for creating particular JAX pytrees.
    For example, to create a pytree with a static leaf that is ignored
    by JAX when it compares two instances to trigger a JIT recompilation.
    """

    obj: T

    def get(self: HashlessObject[T]) -> T:
        return self.obj

    def __hash__(self) -> int:

        return 0

    def __eq__(self, other: HashlessObject[T]) -> bool:

        if not isinstance(other, HashlessObject) and isinstance(
            other.get(), type(self.get())
        ):
            return False

        return hash(self) == hash(other)


@dataclasses.dataclass
class CustomHashedObject(Generic[T]):
    """
    A class that wraps an object and computes its hash with a custom hash function.
    """

    obj: T

    hash_function: Callable[[T], int] = hash

    def get(self: CustomHashedObject[T]) -> T:
        return self.obj

    def __hash__(self) -> int:

        return self.hash_function(self.obj)

    def __eq__(self, other: CustomHashedObject[T]) -> bool:

        if not isinstance(other, CustomHashedObject) and isinstance(
            other.get(), type(self.get())
        ):
            return False

        return hash(self) == hash(other)
