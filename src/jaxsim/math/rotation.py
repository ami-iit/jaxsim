from typing import Tuple

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Rotation:
    @staticmethod
    def x(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta)
        s = jnp.sin(theta)

        return jnp.array(
            [
                [1, 0, 0],
                [0, c, s],
                [0, -s, c],
            ]
        )

    @staticmethod
    def y(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta)
        s = jnp.sin(theta)

        return jnp.array(
            [
                [c, 0, -s],
                [0, 1, 0],
                [s, 0, c],
            ]
        )

    @staticmethod
    def z(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta)
        s = jnp.sin(theta)

        return jnp.array(
            [
                [c, s, 0],
                [-s, c, 0],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def from_axis_angle(vector: jtp.Vector) -> jtp.Matrix:

        theta = jnp.linalg.norm(vector)

        def theta_is_zero(theta_and_v: Tuple[float, jtp.Vector]) -> jtp.Matrix:

            return jnp.eye(3)

        def theta_is_not_zero(theta_and_v: Tuple[float, jtp.Vector]) -> jtp.Matrix:

            theta, v = theta_and_v

            s = jnp.sin(theta)
            c = jnp.cos(theta)

            c1 = 2 * jnp.sin(theta / 2.0) ** 2

            u = v / theta
            u = jnp.vstack(u.squeeze())

            R = c * jnp.eye(3) - s * Skew.wedge(u) + c1 * u @ u.T

            return R

        return jax.lax.cond(
            pred=(theta == 0.0),
            true_fun=theta_is_zero,
            false_fun=theta_is_not_zero,
            operand=(theta, vector),
        )
