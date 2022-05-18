from typing import Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Inertia:
    @staticmethod
    def to_sixd(mass: float, com: jtp.Vector, I: jtp.Matrix) -> jtp.Matrix:

        if I.shape != (3, 3):
            raise ValueError(I, I.shape)

        C = Skew.wedge(vector=com)

        M = jnp.vstack(
            [
                jnp.hstack([I + mass * C @ C.T, mass * C]),
                jnp.hstack([mass * C.T, mass * jnp.eye(3)]),
            ]
        )

        return M

    @staticmethod
    def to_params(M: jtp.Matrix) -> Tuple[float, jtp.Vector, jtp.Matrix]:

        m = M[5, 5]

        mC = M[0:3, 3:6]
        c = Skew.vee(mC) / m
        I = M[0:3, 0:3] - mC @ mC.T / m

        return m, c, I
