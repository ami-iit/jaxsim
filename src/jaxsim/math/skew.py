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
            vector: A 3D vector.

        Returns:
            The skew-symmetric matrix corresponding to the input vector.

        """

        vector = vector.reshape(-1, 3)

        x, y, z = jnp.split(vector, 3, axis=-1)

        skew = jnp.stack(
            [
                jnp.concatenate([jnp.zeros_like(x), -z, y], axis=-1),
                jnp.concatenate([z, jnp.zeros_like(x), -x], axis=-1),
                jnp.concatenate([-y, x, jnp.zeros_like(x)], axis=-1),
            ],
            axis=-2,
        ).squeeze()

        return skew

    @staticmethod
    def vee(matrix: jtp.Matrix) -> jtp.Vector:
        """
        Extract the 3D vector from a skew-symmetric matrix (vee operator).

        Args:
            matrix: A 3x3 skew-symmetric matrix.

        Returns:
            The 3D vector extracted from the input matrix.

        """
        vector = 0.5 * jnp.vstack(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        return vector
