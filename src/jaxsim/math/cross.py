import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Cross:
    """
    A utility class for cross product matrix operations.
    """

    @staticmethod
    def vx(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        """
        Compute the cross product matrix for 6D velocities.

        Args:
            velocity_sixd: A 6D velocity vector [v, ω].

        Returns:
            The cross product matrix (6x6).

        Raises:
            ValueError: If the input vector does not have a size of 6.
        """
        velocity_sixd = velocity_sixd.reshape(-1, 6)

        v, ω = jnp.split(velocity_sixd, 2, axis=-1)

        v_cross = jnp.concatenate(
            [
                jnp.concatenate(
                    [Skew.wedge(ω), jnp.zeros((ω.shape[0], 3, 3)).squeeze()], axis=-2
                ),
                jnp.concatenate([Skew.wedge(v), Skew.wedge(ω)], axis=-2),
            ],
            axis=-1,
        )

        return v_cross

    @staticmethod
    def vx_star(velocity_sixd: jtp.Vector) -> jtp.Matrix:
        """
        Compute the negative transpose of the cross product matrix for 6D velocities.

        Args:
            velocity_sixd: A 6D velocity vector [v, ω].

        Returns:
            The negative transpose of the cross product matrix (6x6).

        Raises:
            ValueError: If the input vector does not have a size of 6.
        """
        v_cross_star = -Cross.vx(velocity_sixd).T
        return v_cross_star
