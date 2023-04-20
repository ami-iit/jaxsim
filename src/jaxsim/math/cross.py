import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Cross:
    @staticmethod
    def vx(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        v, ω = jnp.split(velocity_sixd.squeeze(), 2)

        v_cross = jnp.block(
            [
                [Skew.wedge(vector=ω), Skew.wedge(vector=v)],
                [jnp.zeros(shape=(3, 3)), Skew.wedge(vector=ω)],
            ]
        )

        return v_cross

    @staticmethod
    def vx_star(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        v_cross_star = -Cross.vx(velocity_sixd).T
        return v_cross_star
