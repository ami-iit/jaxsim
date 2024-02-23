from typing import ClassVar, Generic

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie

from jaxsim.simulation.ode_data import ODEState

from .common import ExplicitRungeKutta, PyTreeType, Time, TimeStep

ODEStateDerivative = ODEState


# =====================================================
# Explicit Runge-Kutta integrators operating on PyTrees
# =====================================================


@jax_dataclasses.pytree_dataclass
class ForwardEuler(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

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


@jax_dataclasses.pytree_dataclass
class Heun(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

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


@jax_dataclasses.pytree_dataclass
class RungeKutta4(ExplicitRungeKutta[PyTreeType], Generic[PyTreeType]):

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


# ===============================================================================
# Explicit Runge-Kutta integrators operating on ODEState and integrating on SO(3)
# ===============================================================================


class ExplicitRungeKuttaSO3Mixin:
    """
    Mixin class to apply over explicit RK integrators defined on
    `PyTreeType = ODEState` to integrate the quaternion on SO(3).
    """

    @classmethod
    def post_process_state(
        cls, x0: ODEState, t0: Time, xf: ODEState, dt: TimeStep
    ) -> ODEState:

        # Indices to convert quaternions between serializations.
        to_xyzw = jnp.array([1, 2, 3, 0])
        to_wxyz = jnp.array([3, 0, 1, 2])

        # Get the initial quaternion.
        W_Q_B_t0 = jaxlie.SO3.from_quaternion_xyzw(
            xyzw=x0.physics_model.base_quaternion[to_xyzw]
        )

        # Get the final angular velocity.
        # This is already computed by averaging the kᵢ in RK-based schemes.
        # Therefore, by using the ω at tf, we obtain a RK scheme operating
        # on the SO(3) manifold.
        W_ω_WB_tf = xf.physics_model.base_angular_velocity

        # Integrate the quaternion on SO(3).
        # Note that we left-multiply with the exponential map since the angular
        # velocity is expressed in the inertial frame.
        W_Q_B_tf = jaxlie.SO3.exp(tangent=dt * W_ω_WB_tf) @ W_Q_B_t0

        # Replace the quaternion in the final state.
        return xf.replace(
            physics_model=xf.physics_model.replace(
                base_quaternion=W_Q_B_tf.as_quaternion_xyzw()[to_wxyz]
            ),
            validate=True,
        )


@jax_dataclasses.pytree_dataclass
class ForwardEulerSO3(ExplicitRungeKuttaSO3Mixin, Heun[ODEState]):
    pass


@jax_dataclasses.pytree_dataclass
class HeunSO3(ExplicitRungeKuttaSO3Mixin, Heun[ODEState]):
    pass


@jax_dataclasses.pytree_dataclass
class RungeKutta4SO3(ExplicitRungeKuttaSO3Mixin, RungeKutta4[ODEState]):
    pass
