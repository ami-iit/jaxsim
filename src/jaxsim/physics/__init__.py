import numpy.typing

from . import algos, model


def default_gravity() -> numpy.typing.NDArray:
    import jax.numpy as jnp

    return jnp.array([0, 0, -9.80])


# from . import dyn, models, spatial, threed, utils
