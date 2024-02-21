from typing import ClassVar

import jax
import jax.numpy as jnp
import jax_dataclasses

from .common import ExplicitRungeKutta

# ================================
# Explicit Runge-Kutta integrators
# ================================


@jax_dataclasses.pytree_dataclass
class ForwardEuler(ExplicitRungeKutta):

    A: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [
            [0],
        ]
    ).astype(float)

    b: ClassVar[jax.typing.ArrayLike] = (
        jnp.array(
            [
                [1],
            ]
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [0],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (1,)


class Heun(ExplicitRungeKutta):

    A: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [
            [0, 0],
            [1 / 2, 0],
        ]
    ).astype(float)

    b: ClassVar[jax.typing.ArrayLike] = (
        jnp.atleast_2d(
            jnp.array([1 / 2, 1 / 2]),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [0, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (2,)


class RungeKutta4(ExplicitRungeKutta):

    A: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0],
        ]
    ).astype(float)

    b: ClassVar[jax.typing.ArrayLike] = (
        jnp.atleast_2d(
            jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jax.typing.ArrayLike] = jnp.array(
        [0, 1 / 2, 1 / 2, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (4,)
