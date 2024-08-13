from collections.abc import Hashable
from typing import Any, TypeVar

import jax

# =========
# JAX types
# =========

Array = jax.Array
Scalar = Array
Vector = Array
Matrix = Array

Int = Scalar
Bool = Scalar
Float = Scalar

PyTree: object = (
    dict[Hashable, TypeVar("PyTree")]
    | list[TypeVar("PyTree")]
    | tuple[TypeVar("PyTree")]
    | None
    | jax.Array
    | Any
)

# =======================
# Mixed JAX / NumPy types
# =======================

ArrayLike = jax.typing.ArrayLike | tuple
ScalarLike = int | float | Scalar | ArrayLike
VectorLike = Vector | ArrayLike | tuple
MatrixLike = Matrix | ArrayLike

IntLike = int | Int | jax.typing.ArrayLike
BoolLike = bool | Bool | jax.typing.ArrayLike
FloatLike = float | Float | jax.typing.ArrayLike
