import dataclasses
from typing import Type

import jax
import jax.numpy as jnp
import jax_dataclasses

from . import JaxsimDataclass, Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class Vmappable(JaxsimDataclass):
    """Abstract class with utilities for vmappable pytrees."""

    batch_size: jax_dataclasses.Static[int] = dataclasses.field(
        default=int(0), repr=False, compare=False, hash=False, kw_only=True
    )

    @property
    def vectorized(self) -> bool:
        """Marks this pytree as vectorized."""

        return self.batch_size > 0

    @classmethod
    def build_from_list(cls: Type[Self], list_of_obj: list[Self]) -> Self:
        """
        Build a vectorized pytree from a list of pytree of the same type.

        Args:
            list_of_obj: The list of pytrees to vectorize.

        Returns:
            The vectorized pytree having as leaves the stacked leaves of the input list.
        """

        if set(type(el) for el in list_of_obj) != {cls}:
            msg = "The input list must contain only objects of type '{}'"
            raise ValueError(msg.format(cls.__name__))

        # Create a pytree by stacking all the leafs of the input list
        data_vec: Vmappable = jax.tree_map(
            lambda *leafs: jnp.array(leafs), *list_of_obj
        )

        # Store the batch dimension
        with data_vec.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            data_vec.batch_size = len(list_of_obj)

        # Detect the most common mutability in the input list
        mutabilities = [e._mutability() for e in list_of_obj]
        mutability = max(set(mutabilities), key=mutabilities.count)

        # Update the mutability of the vectorized pytree
        data_vec._set_mutability(mutability)

        return data_vec

    def vectorize(self: Self, batch_size: int) -> Self:
        """
        Return a vectorized version of this pytree.

        Args:
            batch_size: The batch size.

        Returns:
            A vectorized version of this pytree obtained by stacking the leaves of the
            original pytree along a new batch dimension (the first one).
        """

        if self.vectorized:
            raise RuntimeError("Cannot vectorize an already vectorized object")

        if batch_size == 0:
            return self.copy()

        # TODO validate if mutability is maintained

        return self.__class__.build_from_list(list_of_obj=[self] * batch_size)

    def extract_element(self: Self, index: int) -> Self:
        """
        Extract the i-th element from a vectorized pytree.

        Args:
            index: The index of the element to extract.

        Returns:
            A non vectorized pytree obtained by extracting the i-th element from the
            vectorized pytree.
        """

        if index < 0:
            raise ValueError("The index of the desired element cannot be negative")

        if index == 0 and self.batch_size == 0:
            return self.copy()

        if not self.vectorized:
            raise RuntimeError("Cannot extract elements from a non-vectorized object")

        if index >= self.batch_size:
            raise ValueError("The index must be smaller than the batch size")

        # Get the i-th pytree by extracting the i-th element from the vectorized pytree
        data = jax.tree_map(lambda leaf: leaf[index], self)

        # Update the batch size of the extracted scalar pytree
        with data.mutable_context(mutability=Mutability.MUTABLE):
            data.batch_size = 0

        return data
