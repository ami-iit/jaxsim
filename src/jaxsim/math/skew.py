import jax.numpy as jnp

import jaxsim.typing as jtp


class Skew:
    @staticmethod
    def wedge(vector: jtp.Vector) -> jtp.Matrix:
        vector = vector.squeeze()

        x, y, z = vector
        skew = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        return skew

    @staticmethod
    def vee(matrix: jtp.Matrix) -> jtp.Vector:
        # Note: if the input is not already skew-symmetric, this method returns
        #       the values of the skew-symmetric component
        vector = 0.5 * jnp.vstack(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )

        return vector
