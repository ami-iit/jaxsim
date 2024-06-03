from __future__ import annotations

import dataclasses
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


@jax_dataclasses.pytree_dataclass
class HashedNumpyArray:
    """
    A class that wraps a numpy array and makes it hashable.

    This is useful for creating particular JAX pytrees.
    For example, to create a pytree with a plain NumPy or JAX NumPy array as static leaf.

    Note:
        Calculating with the wrapper class the hash of a very large array can be
        very expensive. If the array is large and only the equality operator is needed,
        set `large_array=True` to use a faster comparison method.
    """

    array: jax.Array | npt.NDArray

    large_array: jax_dataclasses.Static[bool] = dataclasses.field(
        default=False, repr=False, compare=False, hash=False
    )

    def get(self) -> jax.Array | npt.NDArray:
        return self.array

    def __hash__(self) -> int:

        return hash(tuple(np.atleast_1d(self.array).flatten().tolist()))

    def __eq__(self, other: HashedNumpyArray) -> bool:

        if not isinstance(other, HashedNumpyArray):
            return False

        if self.large_array:
            return np.array_equal(self.array, other.array)

        return hash(self) == hash(other)
