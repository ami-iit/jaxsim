from typing import Any, Hashable

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

PyTree = dict[Hashable, "PyTree"] | list["PyTree"] | tuple["PyTree"] | None | Any

# =======================
# Mixed JAX / NumPy types
# =======================

Array = jax.typing.ArrayLike
Vector = Array
Matrix = Array

Int = int | IntJax
Bool = bool | ArrayJax
Float = float | FloatJax

ArrayLike = Array
VectorLike = Vector
MatrixLike = Matrix
IntLike = Int
BoolLike = Bool
FloatLike = Float
