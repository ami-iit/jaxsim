from collections.abc import Hashable
from typing import Any

import jax

# =========
# JAX types
# =========

ScalarJax = jax.Array
IntJax = ScalarJax
BoolJax = ScalarJax
FloatJax = ScalarJax

ArrayJax = jax.Array
VectorJax = ArrayJax
MatrixJax = ArrayJax

PyTree = (
    dict[Hashable, "PyTree"] | list["PyTree"] | tuple["PyTree"] | None | jax.Array | Any
)

# =======================
# Mixed JAX / NumPy types
# =======================

Array = jax.typing.ArrayLike
Scalar = Array
Vector = Array
Matrix = Array

Int = int | IntJax
Bool = bool | ArrayJax
Float = float | FloatJax

ScalarLike = Scalar | int | float
ArrayLike = Array
VectorLike = Vector
MatrixLike = Matrix
IntLike = Int
BoolLike = Bool
FloatLike = Float
