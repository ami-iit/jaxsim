import gymnasium.spaces
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import numpy.typing as npt
from gymnasium.spaces.utils import flatdim, flatten
from gymnasium.vector.utils.spaces import batch_space

import jaxsim.typing as jtp
from jaxsim.utils import not_tracing, tracing

from .space import Space

# TODO: inherit from gymnasium.spaces?


class PyTree(Space):
    """"""

    def __init__(self, low: jtp.PyTree, high: jtp.PyTree):
        """"""

        # ==========================
        # Check low and high pytrees
        # ==========================
        # TODO: make generic (pytrees_with_same_dtype|shape|supported_dtype) and move
        #       to utils

        # supported_dtypes = {
        #     jnp.array(0, dtype=jnp.float32).dtype,
        #     jnp.array(0, dtype=jnp.float64).dtype,
        #     jnp.array(0, dtype=int).dtype,
        #     jnp.array(0, dtype=bool).dtype,
        # }
        #
        # dtypes_supported, _ = jax.flatten_util.ravel_pytree(
        #     jax.tree_util.tree_map(
        #         lambda l1, l2: jnp.array(l1).dtype in supported_dtypes
        #         and jnp.array(l2).dtype in supported_dtypes,
        #         low,
        #         high,
        #     )
        # )
        #
        # if not jnp.alltrue(dtypes_supported):
        # # if not_tracing(low) and not jnp.alltrue(dtypes_supported):
        # # if jnp.where(jnp.array([tracing(low), jnp.alltrue(dtypes_supported)]).any(), False, True):
        # # if np.any([not_tracing(low), jnp.alltrue(dtypes_supported)]):
        # # if not_tracing(low):
        #     raise ValueError(
        #         "Either low or high pytrees have attributes with unsupported dtype"
        #     )
        #
        # shape_match, _ = jax.flatten_util.ravel_pytree(
        #     jax.tree_util.tree_map(
        #         lambda l1, l2: jnp.array(l1).shape == jnp.array(l2).shape, low, high
        #     )
        # )
        #
        # if not jnp.alltrue(shape_match):
        # # if not_tracing(low) and not jnp.alltrue(shape_match):
        #     raise ValueError("Wrong shape of low and high attributes")
        #
        # dtype_match, _ = jax.flatten_util.ravel_pytree(
        #     jax.tree_util.tree_map(
        #         lambda l1, l2: jnp.array(l1).dtype == jnp.array(l2).dtype, low, high
        #     )
        # )
        #
        # if not jnp.alltrue(dtype_match):
        # # if not_tracing(low) and not jnp.alltrue(dtype_match):
        #     raise ValueError("Wrong dtype of low and high attributes")

        # Flatten the pytrees
        low_flat, _ = jax.flatten_util.ravel_pytree(low)
        high_flat, _ = jax.flatten_util.ravel_pytree(high)

        if low_flat.dtype != high_flat.dtype:
            raise ValueError(low_flat.dtype, high_flat.dtype)

        if low_flat.shape != high_flat.shape:
            raise ValueError(low_flat.shape, high_flat.shape)

        # Transform all leafs to array and store them in the object
        self.low = jax.tree_util.tree_map(lambda l: jnp.array(l), low)
        self.high = jax.tree_util.tree_map(lambda l: jnp.array(l), high)
        self.shape = low_flat.shape

    # TODO: what if key is a vector?
    def sample(self, key: jax.random.PRNGKeyArray) -> jtp.PyTree:
        """"""

        def random_array(
            key, shape: tuple, min: jtp.PyTree, max: jtp.PyTree, dtype
        ) -> jtp.Array:
            """Helper to select the right sampling function for the supported dtypes"""

            match dtype:
                case jnp.float32.dtype | jnp.float64.dtype:
                    return jax.random.uniform(
                        key=key,
                        shape=shape,
                        minval=min,
                        maxval=max,
                        dtype=dtype,
                    )
                case jnp.int16.dtype | jnp.int32.dtype | jnp.int64.dtype:
                    return jax.random.randint(
                        key=key,
                        shape=shape,
                        minval=min,
                        maxval=max + 1,
                        dtype=dtype,
                    )
                case jnp.bool_.dtype:
                    return jax.random.randint(
                        key=key,
                        shape=shape,
                        minval=min,
                        maxval=max + 1,
                    ).astype(bool)
                case _:
                    raise ValueError(dtype)

        # Create and flatten a tree having a PRNGKey for each leaf
        key_pytree = jax.tree_util.tree_map(lambda l: jax.random.PRNGKey(0), self.low)
        key_pytree_flat, unflatten_fn = jax.flatten_util.ravel_pytree(key_pytree)

        # Generate a pytree having a subkey in each leaf
        key, *subkey_flat = jax.random.split(key=key, num=key_pytree_flat.size / 2 + 1)
        subkey_pytree = unflatten_fn(jnp.array(subkey_flat).flatten())

        # Generate a pytree sampling leafs according to their dtype and using a
        # different key for each of them
        return jax.tree_util.tree_map(
            lambda low, high, subkey: random_array(
                key=key, shape=low.shape, min=low, max=high, dtype=low.dtype
            ),
            self.low,
            self.high,
            subkey_pytree,
        )

    def contains(self, x: jtp.PyTree) -> bool:
        """"""

        def is_inside_bounds(x, low, high):
            return jax.lax.select(
                pred=jnp.alltrue(
                    jnp.array([jnp.alltrue(x >= low), jnp.alltrue(x <= high)])
                ),
                on_true=True,
                on_false=False,
            )

        contains_all_leaves = jax.tree_util.tree_map(
            lambda low, high, l: is_inside_bounds(x=l, low=low, high=high),
            self.low,
            self.high,
            x,
        )

        contains_all_leaves_flat, _ = jax.flatten_util.ravel_pytree(contains_all_leaves)

        return jnp.alltrue(contains_all_leaves_flat)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""

        return True

    def to_box(self) -> gymnasium.spaces.Box:
        """"""

        # low_flat, _ = jax.flatten_util.ravel_pytree(self.low)
        # high_flat, _ = jax.flatten_util.ravel_pytree(self.high)

        # return gymnasium.spaces.Box(low=np.array(low_flat), high=np.array(high_flat))

        return gymnasium.spaces.Box(
            low=self.flatten_sample(x=self.low), high=self.flatten_sample(x=self.high)
        )

    # TODO: use above
    def flatten_sample(self, x: jtp.PyTree) -> jtp.VectorJax:
        """"""

        x_flat, _ = jax.flatten_util.ravel_pytree(x)
        return x_flat

    def unflatten_sample(self, x: jtp.Vector) -> jtp.PyTree:
        """"""

        _, unflatten_fn = jax.flatten_util.ravel_pytree(self.low)
        return unflatten_fn(x)

    def clip(self, x: jtp.PyTree) -> jtp.PyTree:
        """"""

        return jax.tree_util.tree_map(
            lambda low, high, l: jnp.array(
                jnp.clip(a=l, a_min=low, a_max=high), dtype=low.dtype
            ),
            self.low,
            self.high,
            x,
        )

    # TODO: flatten()
    # TODO: unflatten() from float with proper type casting
    # TODO: normalize?


@flatdim.register(PyTree)
def _flatdim_pytree(space: PyTree) -> int:
    """"""

    low_flat, _ = jax.flatten_util.ravel_pytree(space.low)
    return low_flat.size


@flatten.register(PyTree)
def _flatten_pytree(space: PyTree, x: jtp.PyTree) -> npt.NDArray:
    """"""

    assert x in space
    x_flat, _ = jax.flatten_util.ravel_pytree(x)

    return x_flat


@batch_space.register(PyTree)
def _batch_space_pytree(space: PyTree, n: int = 1) -> PyTree:
    """"""

    low_batched = jax.tree_util.tree_map(lambda l: jnp.stack([l] * n), space.low)
    high_batched = jax.tree_util.tree_map(lambda l: jnp.stack([l] * n), space.high)

    # TODO: np_random
    return PyTree(low=low_batched, high=high_batched)
