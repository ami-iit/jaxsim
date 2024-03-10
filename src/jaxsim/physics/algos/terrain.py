import abc

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp


class Terrain(abc.ABC):
    delta = 0.010

    @abc.abstractmethod
    def height(self, x: float, y: float) -> float:
        pass

    def normal(self, x: float, y: float) -> jtp.Vector:
        """
        Compute the normal vector of the terrain at a specific (x, y) location.

        Args:
            x (float): The x-coordinate of the location.
            y (float): The y-coordinate of the location.

        Returns:
            jtp.Vector: The normal vector of the terrain surface at the specified location.
        """

        # https://stackoverflow.com/a/5282364
        h_xp = self.height(x=x + self.delta, y=y)
        h_xm = self.height(x=x - self.delta, y=y)
        h_yp = self.height(x=x, y=y + self.delta)
        h_ym = self.height(x=x, y=y - self.delta)

        n = jnp.array(
            [(h_xm - h_xp) / (2 * self.delta), (h_ym - h_yp) / (2 * self.delta), 1.0]
        )

        return n / jnp.linalg.norm(n)


@jax_dataclasses.pytree_dataclass
class FlatTerrain(Terrain):
    def height(self, x: float, y: float) -> float:
        return 0.0


@jax_dataclasses.pytree_dataclass
class PlaneTerrain(Terrain):
    plane_normal: list = jax_dataclasses.field(default_factory=lambda: [0, 0, 1.0])

    @staticmethod
    def build(plane_normal: list) -> "PlaneTerrain":
        """
        Create a PlaneTerrain instance with a specified plane normal vector.

        Args:
            plane_normal (list): The normal vector of the terrain plane.

        Returns:
            PlaneTerrain: A PlaneTerrain instance.
        """

        return PlaneTerrain(plane_normal=plane_normal)

    def height(self, x: float, y: float) -> float:
        """
        Compute the height of the terrain at a specific (x, y) location on a plane.

        Args:
            x (float): The x-coordinate of the location.
            y (float): The y-coordinate of the location.

        Returns:
            float: The height of the terrain at the specified location on the plane.
        """

        a, b, c = self.plane_normal
        return -(a * x + b * y) / c
