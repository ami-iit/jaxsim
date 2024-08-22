import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

from .skew import Skew


class Rotation:
    @staticmethod
    def x(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the X-axis.

        Args:
            theta (jtp.Float): Rotation angle in radians.

        Returns:
            jtp.Matrix: 3D rotation matrix.
        """
        return jaxlie.SO3.from_x_radians(theta=theta).as_matrix()

    @staticmethod
    def y(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the Y-axis.

        Args:
            theta (jtp.Float): Rotation angle in radians.

        Returns:
            jtp.Matrix: 3D rotation matrix.
        """
        return jaxlie.SO3.from_y_radians(theta=theta).as_matrix()

    @staticmethod
    def z(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the Z-axis.

        Args:
            theta (jtp.Float): Rotation angle in radians.

        Returns:
            jtp.Matrix: 3D rotation matrix.
        """
        return jaxlie.SO3.from_z_radians(theta=theta).as_matrix()

    @staticmethod
    def from_axis_angle(vector: jtp.Vector) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix from an axis-angle representation.

        Args:
            vector (jtp.Vector): Axis-angle representation as a 3D vector.

        Returns:
            jtp.Matrix: 3D rotation matrix.

        """
        vector = vector.squeeze()
        theta = jnp.linalg.norm(vector)

        def theta_is_not_zero(theta_and_v: tuple[jtp.Float, jtp.Vector]) -> jtp.Matrix:
            theta, v = theta_and_v

            s = jnp.sin(theta)
            c = jnp.cos(theta)

            c1 = 2 * jnp.sin(theta / 2.0) ** 2

            u = v / theta
            u = jnp.vstack(u.squeeze())

            R = c * jnp.eye(3) - s * Skew.wedge(u) + c1 * u @ u.T

            return R.transpose()

        return jax.lax.cond(
            pred=(theta == 0.0),
            true_fun=lambda operand: jnp.eye(3),
            false_fun=theta_is_not_zero,
            operand=(theta, vector),
        )
