import abc
import contextlib
import copy
from typing import ContextManager

import jax.abstract_arrays
import jax.flatten_util
import jax.interpreters.partial_eval
import jax_dataclasses

import jaxsim.typing as jtp

from . import Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class JaxsimDataclass(abc.ABC):
    """"""

    @contextlib.contextmanager
    def editable(self: Self, validate: bool = True) -> ContextManager[Self]:
        """"""

        mutability = (
            Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
        )

        with JaxsimDataclass.mutable_context(self.copy(), mutability=mutability) as obj:
            yield obj

    @contextlib.contextmanager
    def mutable_context(self: Self, mutability: Mutability) -> ContextManager[Self]:
        """"""

        original_mutability = self._mutability()

        try:
            self._set_mutability(mutability)
            yield self
        except Exception as e:
            self._set_mutability(original_mutability)
            raise e
        finally:
            self._set_mutability(original_mutability)

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
            _ = [obj.__setattr__(k, copy.copy(v)) for k, v in kwargs.items()]

        obj._set_mutability(mutability=self._mutability())
        return obj

    def flatten(self) -> jtp.VectorJax:
        return jax.flatten_util.ravel_pytree(self)[0]
