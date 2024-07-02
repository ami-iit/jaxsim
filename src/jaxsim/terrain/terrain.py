from __future__ import annotations

import abc
import dataclasses

import jax.numpy as jnp
import jax_dataclasses

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

    z: float = dataclasses.field(default=0.0, kw_only=True)

    @staticmethod
    def build(height: jtp.FloatLike) -> FlatTerrain:

        return FlatTerrain(z=float(height))

    def height(self, x: jtp.FloatLike, y: jtp.FloatLike) -> jtp.Float:

        return jnp.array(self.z, dtype=float)

    def __hash__(self) -> int:

        return hash(self.z)

    def __eq__(self, other: FlatTerrain) -> bool:

        if not isinstance(other, FlatTerrain):
            return False

        return self.z == other.z


@jax_dataclasses.pytree_dataclass
class PlaneTerrain(FlatTerrain):

    plane_normal: tuple[float, float, float] = jax_dataclasses.field(
        default=(0.0, 0.0, 0.0), kw_only=True
    )

    @staticmethod
    def build(
        plane_normal: jtp.VectorLike, plane_height_over_origin: jtp.FloatLike = 0.0
    ) -> PlaneTerrain:
        """
        Create a PlaneTerrain instance with a specified plane normal vector.

        Args:
            plane_normal: The normal vector of the terrain plane.
            plane_height_over_origin: The height of the plane over the origin.

        Returns:
            PlaneTerrain: A PlaneTerrain instance.
        """

        plane_normal = jnp.array(plane_normal, dtype=float)
        plane_height_over_origin = jnp.array(plane_height_over_origin, dtype=float)

        if plane_normal.shape != (3,):
            msg = "Expected a 3D vector for the plane normal, got '{}'."
            raise ValueError(msg.format(plane_normal.shape))

        # Make sure that the plane normal is a unit vector.
        plane_normal = plane_normal / jnp.linalg.norm(plane_normal)

        return PlaneTerrain(
            z=float(plane_height_over_origin),
            plane_normal=tuple(plane_normal.tolist()),
        )

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
        A, B, C = self.plane_normal

        # Compute the final coefficient D considering the terrain height.
        D = -C * self.z

        # Invert the plane equation to get the height at the given (x, y) coordinates.
        return jnp.array(-(A * x + B * y + D) / C).astype(float)

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                hash(self.z),
                HashedNumpyArray.hash_of_array(
                    array=jnp.array(self.plane_normal, dtype=float)
                ),
            )
        )

    def __eq__(self, other: PlaneTerrain) -> bool:

        if not isinstance(other, PlaneTerrain):
            return False

        if not (
            jnp.allclose(self.z, other.z)
            and jnp.allclose(
                jnp.array(self.plane_normal, dtype=float),
                jnp.array(other.plane_normal, dtype=float),
            )
        ):
            return False

        return True
