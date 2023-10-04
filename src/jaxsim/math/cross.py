import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Cross:
    @staticmethod
    def vx(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        """
        Compute the cross product matrix for 6D velocities.

        Args:
            velocity_sixd (jtp.Vector): A 6D velocity vector [v, ω].

        Returns:
            jtp.Matrix: The cross product matrix (6x6).

        Raises:
            ValueError: If the input vector does not have a size of 6.
        """
        v, ω = jnp.split(velocity_sixd.squeeze(), 2)

        v_cross = jnp.vstack(
            [
                jnp.block([Skew.wedge(vector=ω), Skew.wedge(vector=v)]),
                jnp.block([jnp.zeros(shape=(3, 3)), Skew.wedge(vector=ω)]),
            ]
        )

        return v_cross

    @staticmethod
    def vx_star(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        """
        Compute the negative transpose of the cross product matrix for 6D velocities.

        Args:
            velocity_sixd (jtp.Vector): A 6D velocity vector [v, ω].

        Returns:
            jtp.Matrix: The negative transpose of the cross product matrix (6x6).

        Raises:
            ValueError: If the input vector does not have a size of 6.
        """
        v_cross_star = -Cross.vx(velocity_sixd).T
        return v_cross_star
