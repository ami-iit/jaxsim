import copy
import logging
from typing import Any

import gymnasium as gym
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import numpy.typing as npt
from gymnasium.spaces.utils import flatdim, flatten
from gymnasium.vector.utils.spaces import batch_space

import jaxsim.typing as jtp
from jaxsim.utils import not_tracing


class PyTree(gym.Space[jtp.PyTree]):
    """A generic space operating on JAX PyTree objects."""

    def __init__(
        self,
        low: jtp.PyTree,
        high: jtp.PyTree,
        seed: int | None = None,
        # TODO:
        vectorize: int | None = None,
    ) -> None:
        """"""

        # ====================
        # Handle vectorization
        # ====================

        self.vectorize = vectorize
        self.vectorized = False

        if vectorize is not None and vectorize < 2:
            msg = f"Ignoring 'vectorize={vectorize}' argument since it is < 2"
            logging.warning(msg=msg)

        if vectorize is not None and vectorize >= 2:
            self.vectorized = True
            low = jax.tree_util.tree_map(lambda l: jnp.stack([l] * vectorize), low)
            high = jax.tree_util.tree_map(lambda l: jnp.stack([l] * vectorize), high)

        # ==========================
        # Check low and high pytrees
        # ==========================
        # TODO: make generic (pytrees_with_same_dtype|shape|supported_dtype) and move
        #       to utils

        def check() -> None:
            supported_dtypes = {
                jnp.array(0, dtype=jnp.float32).dtype,
                jnp.array(0, dtype=jnp.float64).dtype,
                jnp.array(0, dtype=int).dtype,
                jnp.array(0, dtype=bool).dtype,
            }

            dtypes_supported = self.flatten_pytree(
                pytree=jax.tree_util.tree_map(
                    lambda l1, l2: jnp.array(l1).dtype in supported_dtypes
                    and jnp.array(l2).dtype in supported_dtypes,
                    low,
                    high,
                )
            )

            if not jnp.alltrue(dtypes_supported):
                raise ValueError(
                    "Either low or high pytrees have attributes with unsupported dtype"
                )

            shape_match = self.flatten_pytree(
                pytree=jax.tree_util.tree_map(
                    lambda l1, l2: jnp.array(l1).shape == jnp.array(l2).shape, low, high
                )
            )

            if not jnp.alltrue(shape_match):
                raise ValueError("Wrong shape of low and high attributes")

            dtype_match = self.flatten_pytree(
                pytree=jax.tree_util.tree_map(
                    lambda l1, l2: jnp.array(l1).dtype == jnp.array(l2).dtype, low, high
                )
            )

            if not jnp.alltrue(dtype_match):
                raise ValueError("Wrong dtype of low and high attributes")

        if not_tracing(var=low):
            check()

        # ===============
        # Build the space
        # ===============

        # Flatten the pytrees
        low_flat = self.flatten_pytree(pytree=low)
        high_flat = self.flatten_pytree(pytree=high)

        if low_flat.dtype != high_flat.dtype:
            raise ValueError(low_flat.dtype, high_flat.dtype)

        if low_flat.shape != high_flat.shape:
            raise ValueError(low_flat.shape, high_flat.shape)

        # Transform all leafs to array and store them in the object
        self.low = jax.tree_util.tree_map(lambda l: jnp.array(l), low)
        self.high = jax.tree_util.tree_map(lambda l: jnp.array(l), high)

        # Initialize the seed if not given
        seed = (
            seed
            if seed is not None
            else np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")
        )

        # Initialize the JAX random key
        self.key = jax.random.PRNGKey(seed=seed)

        # Initialize parent class
        super().__init__(shape=None, dtype=None, seed=int(seed))

    def subkey(self, num: int = 1) -> jax.random.PRNGKeyArray:
        """
        Generate one or multiple sub-keys from the internal key.

        Note:
            The internal key is automatically updated, there's no need to handle
            the environment key externally.

        Args:
            num: Number of keys to generate.

        Returns:
            The generated sub-keys.
        """

        self.key, *sub_keys = jax.random.split(self.key, num=num + 1)
        return jnp.stack(sub_keys).squeeze()

    # TODO: what if key is a vector? -> multiple outputs?
    def sample_with_key(self, key: jax.random.PRNGKeyArray) -> jtp.PyTree:
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

        # Create and flatten a tree having a PRNGKey for each leaf.
        # We do this just to get the number of keys we need to generate, and the
        # function to unflatten the ravelled tree.
        dummy_pytree = jax.tree_util.tree_map(lambda l: jax.random.PRNGKey(0), self.low)
        dummy_pytree_flat, unflatten_fn = jax.flatten_util.ravel_pytree(dummy_pytree)

        # Use the subkey to generate new keys, one for each leaf.
        # Note: the division by 2 is needed because keys are vector of 2 elements.
        subkey_flat = jax.random.split(key=key, num=dummy_pytree_flat.size / 2)

        # Generate a pytree having a different subkey in each leaf
        subkey_pytree = unflatten_fn(jnp.array(subkey_flat).flatten())

        # Generate a pytree by sampling leafs according to their dtype and using a
        # different subkey for each of them
        return jax.tree_util.tree_map(
            lambda low, high, subkey: random_array(
                key=subkey, shape=low.shape, min=low, max=high, dtype=low.dtype
            ),
            self.low,
            self.high,
            subkey_pytree,
        )

    def sample(self, mask: Any | None = None) -> jtp.PyTree:
        """"""

        # Generate a subkey
        subkey = self.subkey(num=1)

        return self.sample_with_key(key=subkey)

    def seed(self, seed: int | None = None) -> list[int]:
        """"""

        seed = (
            seed
            if seed is not None
            else np.random.default_rng().integers(0, 2**32 - 1, dtype="uint32")
        )

        self.key = jax.random.PRNGKey(seed=seed)
        return super().seed(seed=seed)

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

        contains_all_leaves_flat = self.flatten_pytree(pytree=contains_all_leaves)

        return jnp.alltrue(contains_all_leaves_flat)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""

        return True

    def to_box(self) -> gym.spaces.Box:
        """"""

        get_first_element = lambda pytree: jax.tree_util.tree_map(
            lambda l: l[0], pytree
        )

        low = self.low if not self.vectorized else get_first_element(self.low)
        high = self.high if not self.vectorized else get_first_element(self.high)

        low_flat = np.array(self.flatten_pytree(pytree=low))
        high_flat = np.array(self.flatten_pytree(pytree=high))

        if self.vectorized:
            assert self.vectorize >= 2

            repeats = tuple([self.vectorize] + [1] * low_flat.ndim)

            low_flat = np.tile(low_flat, repeats)
            high_flat = np.tile(high_flat, repeats)

        return gym.spaces.Box(
            low=np.array(low_flat, dtype=np.float32),
            high=np.array(high_flat, dtype=np.float32),
            seed=copy.deepcopy(self.np_random),
        )

    def to_dict(self) -> gym.spaces.Dict:
        # if low/high are a dataclass -> convert to dict
        raise NotImplementedError

    @staticmethod
    def flatten_pytree(pytree: jtp.PyTree) -> jtp.VectorJax:
        """"""

        # print("flatten_pytree")
        pytree_flat, _ = jax.flatten_util.ravel_pytree(pytree)
        return pytree_flat

    def flatten_sample(self, pytree: jtp.PyTree) -> jtp.VectorJax:
        """"""

        if not self.vectorized:
            return self.flatten_pytree(pytree=pytree)

        # @jax.jit
        # def flatten_pytree(pytree: jtp.PyTree) -> jtp.ArrayJax:
        #     print("compiling")
        #     return jax.vmap(self.flatten_pytree)(pytree)
        #
        # return flatten_pytree(pytree=pytree)

        # TODO: this trigger recompilation -> do some trick
        # return jax.jit(jax.vmap(self.flatten_pytree))(pytree)
        return PyTree._flatten_sample_vmap(pytree)

    @staticmethod
    @jax.jit
    def _flatten_sample_vmap(pytree: jtp.PyTree) -> jtp.VectorJax:
        return jax.vmap(PyTree.flatten_pytree)(pytree)

    def unflatten_sample(self, x: jtp.Vector) -> jtp.PyTree:
        """"""

        if not self.vectorized:
            _, unflatten_fn = jax.flatten_util.ravel_pytree(self.low)
            return unflatten_fn(x)

        # low_1d = jax.tree_util.tree_map(lambda l: l[0], self.low)
        # low_1d_flat, unflatten_fn = jax.flatten_util.ravel_pytree(low_1d)
        #
        # @jax.jit
        # def unflatten_sample(x: jtp.Vector) -> jtp.PyTree:
        #     return jax.vmap(unflatten_fn)(x)
        #
        # return unflatten_sample(x=x)
        return PyTree._unflatten_sample_vmap(x=x, low=self.low)

    @staticmethod
    @jax.jit
    def _unflatten_sample_vmap(x: jtp.Vector, low: jtp.PyTree) -> jtp.PyTree:
        """"""

        low_1d = jax.tree_util.tree_map(lambda l: l[0], low)
        low_1d_flat, unflatten_fn = jax.flatten_util.ravel_pytree(low_1d)

        return jax.vmap(unflatten_fn)(x)

    def clip(self, x: jtp.PyTree) -> jtp.PyTree:
        """"""

        # TODO: prevent recompilation
        # @jax.jit
        def _clip(pytree: jtp.PyTree) -> jtp.PyTree:
            return jax.tree_util.tree_map(
                lambda low, high, leaf: jnp.array(
                    jnp.clip(a=leaf, a_min=low, a_max=high), dtype=jnp.array(low).dtype
                ),
                self.low,
                self.high,
                pytree,
            )

        return _clip(pytree=x)


@flatdim.register(PyTree)
def _flatdim_pytree(space: PyTree) -> int:
    """"""

    low_flat = space.flatten_sample(pytree=space.low)
    return low_flat.size


@flatten.register(PyTree)
def _flatten_pytree(space: PyTree, x: jtp.PyTree) -> npt.NDArray:
    """"""

    assert x in space
    return space.flatten_sample(pytree=x)


@batch_space.register(PyTree)
def _batch_space_pytree(space: PyTree, n: int = 1) -> PyTree:
    """"""

    return PyTree(low=space.low, high=space.high, vectorize=n)
