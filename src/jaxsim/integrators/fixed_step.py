import dataclasses
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
    """
    Forward Euler integrator.
    """

    A: jtp.Matrix = dataclasses.field(
        default_factory=lambda: jnp.atleast_2d(0).astype(float), compare=False
    )
    b: jtp.Matrix = dataclasses.field(
        default_factory=lambda: jnp.atleast_2d(1).astype(float), compare=False
    )

    c: jtp.Vector = dataclasses.field(
        default_factory=lambda: jnp.atleast_1d(0).astype(float), compare=False
    )

    row_index_of_solution: int = 0
    order_of_bT_rows: tuple[int, ...] = (1,)
    index_of_fsal: jtp.IntLike | None = None
    fsal_enabled_if_supported: bool = False


@jax_dataclasses.pytree_dataclass
class Heun2(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):
    """
    Heun's second-order integrator.
    """

    A: jtp.Matrix = dataclasses.field(
        default_factory=lambda: jnp.array(
            [
                [0, 0],
                [1, 0],
            ]
        ).astype(float),
        compare=False,
    )

    b: jtp.Matrix = dataclasses.field(
        default_factory=lambda: (
            jnp.atleast_2d(
                jnp.array([1 / 2, 1 / 2]),
            )
            .astype(float)
            .transpose()
        ),
        compare=False,
    )

    c: jtp.Vector = dataclasses.field(
        default_factory=lambda: jnp.array(
            [0, 1],
        ).astype(float),
        compare=False,
    )

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (2,)
    index_of_fsal: jtp.IntLike | None = None
    fsal_enabled_if_supported: bool = False


@jax_dataclasses.pytree_dataclass
class RungeKutta4(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):
    """
    Fourth-order Runge-Kutta integrator.
    """

    A: jtp.Matrix = dataclasses.field(
        default_factory=lambda: jnp.array(
            [
                [0, 0, 0, 0],
                [1 / 2, 0, 0, 0],
                [0, 1 / 2, 0, 0],
                [0, 0, 1, 0],
            ]
        ).astype(float),
        compare=False,
    )

    b: jtp.Matrix = dataclasses.field(
        default_factory=lambda: (
            jnp.atleast_2d(
                jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
            )
            .astype(float)
            .transpose()
        ),
        compare=False,
    )

    c: jtp.Vector = dataclasses.field(
        default_factory=lambda: jnp.array(
            [0, 1 / 2, 1 / 2, 1],
        ).astype(float),
        compare=False,
    )

    row_index_of_solution: ClassVar[int] = 0
    order_of_bT_rows: ClassVar[tuple[int, ...]] = (4,)
    index_of_fsal: jtp.IntLike | None = None
    fsal_enabled_if_supported: bool = False


# ===============================================================================
# Explicit Runge-Kutta integrators operating on ODEState and integrating on SO(3)
# ===============================================================================


@jax_dataclasses.pytree_dataclass
class ForwardEulerSO3(ExplicitRungeKuttaSO3Mixin, ForwardEuler[js.ode_data.ODEState]):
    """
    Forward Euler integrator for SO(3) states.
    """

    pass


@jax_dataclasses.pytree_dataclass
class Heun2SO3(ExplicitRungeKuttaSO3Mixin, Heun2[js.ode_data.ODEState]):
    """
    Heun's second-order integrator for SO(3) states.
    """

    pass


@jax_dataclasses.pytree_dataclass
class RungeKutta4SO3(ExplicitRungeKuttaSO3Mixin, RungeKutta4[js.ode_data.ODEState]):
    """
    Fourth-order Runge-Kutta integrator for SO(3) states.
    """

    pass
