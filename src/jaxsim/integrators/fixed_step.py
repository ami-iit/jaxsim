from typing import ClassVar, Generic

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp

from .common import ExplicitRungeKutta, ExplicitRungeKuttaSO3Mixin, PyTreeType

ODEStateDerivative = js.ode_data.ODEState

# =====================================================
# Explicit Runge-Kutta integrators operating on PyTrees
# =====================================================


@jax_dataclasses.pytree_dataclass
class ForwardEuler(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

    A: ClassVar[jtp.Matrix] = jnp.atleast_2d(0).astype(float)

    b: ClassVar[jtp.Matrix] = jnp.atleast_2d(1).astype(float).transpose()

    c: ClassVar[jtp.Vector] = jnp.atleast_1d(0).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (1,)


@jax_dataclasses.pytree_dataclass
class Heun2(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

    A: ClassVar[jtp.Matrix] = jnp.array(
        [
            [0, 0],
            [1, 0],
        ]
    ).astype(float)

    b: ClassVar[jtp.Matrix] = (
        jnp.atleast_2d(
            jnp.array([1 / 2, 1 / 2]),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jtp.Vector] = jnp.array(
        [0, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (2,)


@jax_dataclasses.pytree_dataclass
class RungeKutta4(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

    A: ClassVar[jtp.Matrix] = jnp.array(
        [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0],
        ]
    ).astype(float)

    b: ClassVar[jtp.Matrix] = (
        jnp.atleast_2d(
            jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        )
        .astype(float)
        .transpose()
    )

    c: ClassVar[jtp.Vector] = jnp.array(
        [0, 1 / 2, 1 / 2, 1],
    ).astype(float)

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (4,)


# ===============================================================================
# Explicit Runge-Kutta integrators operating on ODEState and integrating on SO(3)
# ===============================================================================


@jax_dataclasses.pytree_dataclass
class ForwardEulerSO3(ExplicitRungeKuttaSO3Mixin, ForwardEuler[js.ode_data.ODEState]):
    pass


@jax_dataclasses.pytree_dataclass
class Heun2SO3(ExplicitRungeKuttaSO3Mixin, Heun2[js.ode_data.ODEState]):
    pass


@jax_dataclasses.pytree_dataclass
class RungeKutta4SO3(ExplicitRungeKuttaSO3Mixin, RungeKutta4[js.ode_data.ODEState]):
    pass
