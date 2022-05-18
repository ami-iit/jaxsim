import abc

import jax.numpy as jnp

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


class FlatTerrain(Terrain):
    def height(self, x: float, y: float) -> float:
        return 0.0
