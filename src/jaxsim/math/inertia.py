from typing import Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Inertia:
    @staticmethod
    def to_sixd(mass: jtp.Float, com: jtp.Vector, I: jtp.Matrix) -> jtp.Matrix:
        if I.shape != (3, 3):
            raise ValueError(I, I.shape)

        c = Skew.wedge(vector=com)

        M = jnp.block(
            [
                [mass * jnp.eye(3), mass * c.T],
                [mass * c, I + mass * c @ c.T],
            ]
        )

        return M

    @staticmethod
    def to_params(M: jtp.Matrix) -> Tuple[jtp.Float, jtp.Vector, jtp.Matrix]:
        m = jnp.diag(M[0:3, 0:3]).sum() / 3

        mC = M[3:6, 0:3]
        c = Skew.vee(mC) / m
        I = M[3:6, 3:6] - (mC @ mC.T / m)

        return m, c, I
