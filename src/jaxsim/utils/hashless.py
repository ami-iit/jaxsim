from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class HashlessObject(Generic[T]):

    obj: T

    def get(self: HashlessObject[T]) -> T:
        return self.obj

    def __hash__(self) -> int:
        return 0
