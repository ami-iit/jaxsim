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
    plane_normal: jtp.Vector = jax_dataclasses.field(default=jnp.array([0, 0, 1.0]))

    def height(self, x: float, y: float) -> float:
        a, b, c = self.plane_normal
        return -(a * x + b * x) / c
