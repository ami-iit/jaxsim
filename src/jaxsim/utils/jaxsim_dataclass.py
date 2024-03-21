import abc
import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Callable, ClassVar, Type

import jax.flatten_util
import jax_dataclasses

import jaxsim.typing as jtp

from . import Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxsimDataclass(abc.ABC):
    """"""

    # This attribute is set by jax_dataclasses
    __mutability__: ClassVar[Mutability] = Mutability.FROZEN

    @contextlib.contextmanager
    def editable(self: Self, validate: bool = True) -> Iterator[Self]:
        """"""

        mutability = (
            Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
        )

        with self.copy().mutable_context(mutability=mutability) as obj:
            yield obj

    @contextlib.contextmanager
    def mutable_context(
        self: Self, mutability: Mutability, restore_after_exception: bool = True
    ) -> Iterator[Self]:
        """"""

        if restore_after_exception:
            self_copy = self.copy()

        original_mutability = self._mutability()

        original_dtypes = JaxsimDataclass.get_leaf_dtypes(tree=self)
        original_shapes = JaxsimDataclass.get_leaf_shapes(tree=self)
        original_weak_types = JaxsimDataclass.get_leaf_weak_types(tree=self)
        original_structure = jax.tree_util.tree_structure(tree=self)

        def restore_self() -> None:
            self._set_mutability(mutability=Mutability.MUTABLE_NO_VALIDATION)
            for f in dataclasses.fields(self_copy):
                setattr(self, f.name, getattr(self_copy, f.name))

        try:
            self._set_mutability(mutability)
            yield self

            if mutability is not Mutability.MUTABLE_NO_VALIDATION:
                new_structure = jax.tree_util.tree_structure(tree=self)
                if original_structure != new_structure:
                    msg = "Pytree structure has changed from {} to {}"
                    raise ValueError(msg.format(original_structure, new_structure))

                new_shapes = JaxsimDataclass.get_leaf_shapes(tree=self)
                if original_shapes != new_shapes:
                    msg = "Leaves shapes have changed from {} to {}"
                    raise ValueError(msg.format(original_shapes, new_shapes))

                new_dtypes = JaxsimDataclass.get_leaf_dtypes(tree=self)
                if original_dtypes != new_dtypes:
                    msg = "Leaves dtypes have changed from {} to {}"
                    raise ValueError(msg.format(original_dtypes, new_dtypes))

                new_weak_types = JaxsimDataclass.get_leaf_weak_types(tree=self)
                if original_weak_types != new_weak_types:
                    msg = "Leaves weak types have changed from {} to {}"
                    raise ValueError(msg.format(original_weak_types, new_weak_types))

        except Exception as e:
            if restore_after_exception:
                restore_self()
            self._set_mutability(original_mutability)
            raise e

        finally:
            self._set_mutability(original_mutability)

    @staticmethod
    def get_leaf_shapes(tree: jtp.PyTree) -> tuple[tuple[int, ...]]:
        return tuple(  # noqa
            leaf.shape
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "shape")
        )

    @staticmethod
    def get_leaf_dtypes(tree: jtp.PyTree) -> tuple:
        return tuple(
            leaf.dtype
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "dtype")
        )

    @staticmethod
    def get_leaf_weak_types(tree: jtp.PyTree) -> tuple[bool, ...]:
        return tuple(
            leaf.weak_type
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "weak_type")
        )

    def is_mutable(self, validate: bool = False) -> bool:
        """"""

        return (
            self.__mutability__ is Mutability.MUTABLE
            if validate
            else self.__mutability__ is Mutability.MUTABLE_NO_VALIDATION
        )

    def set_mutability(self, mutable: bool = True, validate: bool = False) -> None:
        if not mutable:
            mutability = Mutability.FROZEN
        else:
            mutability = (
                Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
            )

        self._set_mutability(mutability=mutability)

    def _mutability(self) -> Mutability:
        return self.__mutability__

    def _set_mutability(self, mutability: Mutability) -> None:
        jax_dataclasses._copy_and_mutate._mark_mutable(
            self, mutable=mutability, visited=set()
        )

    def mutable(self: Self, mutable: bool = True, validate: bool = False) -> Self:
        self.set_mutability(mutable=mutable, validate=validate)
        return self

    def copy(self: Self) -> Self:
        obj = jax.tree_util.tree_map(lambda leaf: leaf, self)
        obj._set_mutability(mutability=self._mutability())
        return obj

    def replace(self: Self, validate: bool = True, **kwargs) -> Self:
        with self.editable(validate=validate) as obj:
            _ = [obj.__setattr__(k, v) for k, v in kwargs.items()]

        obj._set_mutability(mutability=self._mutability())
        return obj

    def flatten(self) -> jtp.VectorJax:
        return self.flatten_fn()(self)

    @classmethod
    def flatten_fn(cls: Type[Self]) -> Callable[[Self], jtp.VectorJax]:
        return lambda pytree: jax.flatten_util.ravel_pytree(pytree)[0]

    def unflatten_fn(self: Self) -> Callable[[jtp.VectorJax], Self]:
        return jax.flatten_util.ravel_pytree(self)[1]
