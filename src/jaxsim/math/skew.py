import jax.numpy as jnp

import jaxsim.typing as jtp


class Skew:
    """
    A utility class for skew-symmetric matrix operations.
    """

    @staticmethod
    def wedge(vector: jtp.Vector) -> jtp.Matrix:
        """
        Compute the skew-symmetric matrix (wedge operator) of a 3D vector.

        Args:
            vector (jtp.Vector): A 3D vector.

        Returns:
            jtp.Matrix: The skew-symmetric matrix corresponding to the input vector.

        """
        vector = vector.squeeze()
        x, y, z = vector
        skew = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return skew

    @staticmethod
    def vee(matrix: jtp.Matrix) -> jtp.Vector:
        """
        Extract the 3D vector from a skew-symmetric matrix (vee operator).

        Args:
            matrix (jtp.Matrix): A 3x3 skew-symmetric matrix.

        Returns:
            jtp.Vector: The 3D vector extracted from the input matrix.

        """
        vector = 0.5 * jnp.vstack(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        return vector
