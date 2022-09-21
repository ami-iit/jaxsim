import jax.numpy as jnp

import jaxsim.typing as jtp


class Adjoint:
    @staticmethod
    def rotate_x(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta).squeeze()
        s = jnp.sin(theta).squeeze()

        return jnp.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, c, s, 0, 0, 0],
                [0, -s, c, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, c, s],
                [0, 0, 0, 0, -s, c],
            ]
        )

    @staticmethod
    def rotate_y(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta).squeeze()
        s = jnp.sin(theta).squeeze()

        return jnp.array(
            [
                [c, 0, -s, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [s, 0, c, 0, 0, 0],
                [0, 0, 0, c, 0, -s],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, s, 0, c],
            ]
        )

    @staticmethod
    def rotate_z(theta: float) -> jtp.Matrix:

        c = jnp.cos(theta).squeeze()
        s = jnp.sin(theta).squeeze()

        return jnp.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def translate(direction: jtp.Vector) -> jtp.Matrix:

        x, y, z = direction

        return jnp.array(
            [
                [1, 0, 0, 0, z, -y],
                [0, 1, 0, -z, 0, x],
                [0, 0, 1, y, -x, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
