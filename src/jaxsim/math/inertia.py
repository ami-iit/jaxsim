import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Inertia:
    @staticmethod
    def to_sixd(mass: jtp.Float, com: jtp.Vector, I: jtp.Matrix) -> jtp.Matrix:
        """
        Convert mass, center of mass, and inertia matrix to a 6x6 inertia matrix.

        Args:
            mass (jtp.Float): The mass of the body.
            com (jtp.Vector): The center of mass position (3D).
            I (jtp.Matrix): The 3x3 inertia matrix.

        Returns:
            jtp.Matrix: The 6x6 inertia matrix.

        Raises:
            ValueError: If the shape of the inertia matrix I is not (3, 3).
        """
        if I.shape != (3, 3):
            raise ValueError(I, I.shape)

        c = Skew.wedge(vector=com)

        M = jnp.vstack(
            [
                jnp.block([mass * jnp.eye(3), mass * c.T]),
                jnp.block([mass * c, I + mass * c @ c.T]),
            ]
        )

        return M

    @staticmethod
    def to_params(M: jtp.Matrix) -> tuple[jtp.Float, jtp.Vector, jtp.Matrix]:
        """
        Convert a 6x6 inertia matrix to mass, center of mass, and inertia matrix.

        Args:
            M (jtp.Matrix): The 6x6 inertia matrix.

        Returns:
            tuple[jtp.Float, jtp.Vector, jtp.Matrix]: A tuple containing mass, center of mass (3D), and inertia matrix (3x3).

        Raises:
            ValueError: If the input matrix M has an unexpected shape.
        """
        m = jnp.diag(M[0:3, 0:3]).sum() / 3

        mC = M[3:6, 0:3]
        c = Skew.vee(mC) / m
        I = M[3:6, 3:6] - (mC @ mC.T / m)

        return m, c, I
