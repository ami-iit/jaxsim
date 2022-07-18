import abc
import contextlib
import copy
from typing import ContextManager, TypeVar

import jax.abstract_arrays
import jax.flatten_util
import jax.interpreters.partial_eval
import jax_dataclasses
from jax_dataclasses._copy_and_mutate import _Mutability as Mutability

import jaxsim.typing as jtp

T = TypeVar("T")


def tracing(var) -> bool:

    return isinstance(
        var,
        (
            jax.abstract_arrays.ShapedArray,
            jax.interpreters.partial_eval.DynamicJaxprTracer,
        ),
    )


class JaxsimDataclass(abc.ABC):
    @contextlib.contextmanager
    def editable(self: T, validate: bool = True) -> ContextManager[T]:

        with jax_dataclasses.copy_and_mutate(self, validate=validate) as self_rw:
            yield self_rw

        self_rw._set_mutability(self._mutability())

    def is_mutable(self: T, validate: bool = False) -> bool:

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

    def mutable(self: T, mutable: bool = True, validate: bool = False) -> T:

        self.set_mutability(mutable=mutable, validate=validate)
        return self

    def copy(self: T) -> T:

        obj = jax.tree_map(lambda leaf: leaf, self)
        obj._set_mutability(mutability=self._mutability())
        return obj

    def replace(self: T, validate: bool = True, **kwargs) -> T:

        with self.editable(validate=validate) as obj:

            _ = [obj.__setattr__(k, copy.copy(v)) for k, v in kwargs.items()]

        obj._set_mutability(mutability=self._mutability())
        return obj

    def flatten(self: T) -> jtp.VectorJax:

        return jax.flatten_util.ravel_pytree(self)[0]
