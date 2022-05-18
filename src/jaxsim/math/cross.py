import jax.numpy as jnp

import jaxsim.typing as jtp


class Cross:
    @staticmethod
    def vx(velocity_sixd: jtp.Vector) -> jtp.Matrix:

        v = velocity_sixd.squeeze()

        v_cross = jnp.array(
            [
                [0, -v[2], v[1], 0, 0, 0],
                [v[2], 0, -v[0], 0, 0, 0],
                [-v[1], v[0], 0, 0, 0, 0],
                [0, -v[5], v[4], 0, -v[2], v[1]],
                [v[5], 0, -v[3], v[2], 0, -v[0]],
                [-v[4], v[3], 0, -v[1], v[0], 0],
            ]
        )

        return v_cross

    @staticmethod
    def vx_star(velocity_sixd: jtp.Vector) -> jtp.Matrix:

        v_cross_star = -Cross.vx(velocity_sixd).T
        return v_cross_star
