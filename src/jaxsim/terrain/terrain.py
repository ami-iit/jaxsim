from __future__ import annotations

import abc
import dataclasses

import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim.typing as jtp


class Terrain(abc.ABC):

    delta = 0.010

    @abc.abstractmethod
    def height(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Float:
        pass

    def normal(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Vector:
        """
        Compute the normal vector of the terrain at a specific (x, y) location.

        Args:
            x: The x-coordinate of the location.
            y: The y-coordinate of the location.

        Returns:
            The normal vector of the terrain surface at the specified location.
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

    _height: float = dataclasses.field(default=0.0, kw_only=True)

    @staticmethod
    def build(height: jtp.FloatLike) -> FlatTerrain:

        return FlatTerrain(_height=float(height))

    def height(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Float:

        return jnp.array(self._height, dtype=float)

    def normal(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Vector:

        return jnp.array([0.0, 0.0, 1.0], dtype=float)

    def __hash__(self) -> int:

        return hash(self._height)

    def __eq__(self, other: FlatTerrain) -> bool:

        if not isinstance(other, FlatTerrain):
            return False

        return self._height == other._height


@jax_dataclasses.pytree_dataclass
class PlaneTerrain(FlatTerrain):

    _normal: tuple[float, float, float] = jax_dataclasses.field(
        default=(0.0, 0.0, 1.0), kw_only=True
    )

    @staticmethod
    def build(height: jtp.FloatLike = 0.0, *, normal: jtp.VectorLike) -> PlaneTerrain:
        """
        Create a PlaneTerrain instance with a specified plane normal vector.

        Args:
            normal: The normal vector of the terrain plane.
            height: The height of the plane over the origin.

        Returns:
            PlaneTerrain: A PlaneTerrain instance.
        """

        normal = jnp.array(normal, dtype=float)
        height = jnp.array(height, dtype=float)

        if normal.shape != (3,):
            msg = "Expected a 3D vector for the plane normal, got '{}'."
            raise ValueError(msg.format(normal.shape))

        # Make sure that the plane normal is a unit vector.
        normal = normal / jnp.linalg.norm(normal)

        return PlaneTerrain(
            _height=height.item(),
            _normal=tuple(normal.tolist()),
        )

    def normal(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Vector:
        """
        Compute the normal vector of the terrain at a specific (x, y) location.

        Args:
            x: The x-coordinate of the location.
            y: The y-coordinate of the location.

        Returns:
            The normal vector of the terrain surface at the specified location.
        """

        return jnp.array(self._normal, dtype=float)

    def height(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Float:
        """
        Compute the height of the terrain at a specific (x, y) location on a plane.

        Args:
            x: The x-coordinate of the location.
            y: The y-coordinate of the location.

        Returns:
            The height of the terrain at the specified location on the plane.
        """

        # Equation of the plane:      A x + B y + C z + D = 0
        # Normal vector coordinates:  (A, B, C)
        # The height over the origin: -D/C

        # Get the plane equation coefficients from the terrain normal.
        A, B, C = self._normal

        # Compute the final coefficient D considering the terrain height.
        D = -C * self._height

        # Invert the plane equation to get the height at the given (x, y) coordinates.
        return jnp.array(-(A * x + B * y + D) / C).astype(float)

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                hash(self._height),
                HashedNumpyArray.hash_of_array(
                    array=jnp.array(self._normal, dtype=float)
                ),
            )
        )

    def __eq__(self, other: PlaneTerrain) -> bool:

        if not isinstance(other, PlaneTerrain):
            return False

        if not (
            np.allclose(self._height, other._height)
            and np.allclose(
                np.array(self._normal, dtype=float),
                np.array(other._normal, dtype=float),
            )
        ):
            return False

        return True
