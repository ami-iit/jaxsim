import operator
from typing import Generator, List, NamedTuple, Sequence, Tuple, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import numpy.typing as npt
from flax.core.frozen_dict import FrozenDict

import jaxsim.typing as jtp


@jax_dataclasses.pytree_dataclass
class Memory(Sequence):
    states: jtp.MatrixJax
    actions: jtp.MatrixJax
    rewards: jtp.VectorJax
    dones: jtp.VectorJax

    values: jtp.VectorJax
    log_prob_actions: jtp.VectorJax

    infos: FrozenDict[str, jtp.PyTree] = jax_dataclasses.field(default_factory=dict)

    @property
    def flat(self) -> bool:
        # return self.dones.ndim == 2
        # return self.dones.squeeze().ndim == 1
        # return self.dones.squeeze().ndim <= 1
        return self.dones.ndim < 3

    @property
    def number_of_levels(self) -> int:
        if self.flat:
            return 0

        self.check_valid()
        return self.dones.shape[0]

    def truncate_last_trajectory(self) -> "Memory":
        if self.flat:
            dones = self.dones.at[-1].set(True)
            memory = jax_dataclasses.replace(self, dones=dones)  # noqa

        else:
            dones = self.dones.at[:, -1].set(True)
            memory = jax_dataclasses.replace(self, dones=dones)  # noqa

        return memory

    def flatten(self) -> "Memory":
        if self.flat:
            return self

        self.check_valid()
        return jax.tree_map(lambda x: jnp.vstack(x), self)

    def unflatten(self) -> "Memory":
        if not self.flat:
            return self

        # return jax.tree_map(lambda x: jnp.stack([jnp.vstack(x)]), self)

        self.check_valid()

        memory = jax.tree_map(lambda x: x[jnp.newaxis, :], self)
        memory.check_valid()
        return memory

    @staticmethod
    def build(
        states: jtp.MatrixJax,
        actions: jtp.MatrixJax,
        rewards: jtp.VectorJax,
        dones: jtp.VectorJax,
        values: jtp.VectorJax,
        log_prob_actions: jtp.VectorJax,
        infos: FrozenDict[str, jtp.ArrayJax] = FrozenDict(),
    ) -> "Memory":
        memory = Memory(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_prob_actions=log_prob_actions,
            infos=infos,
        )

        # We work on (N, 1) 1D arrays
        # Transform scalars to 1D arrays
        memory = jax.tree_map(lambda x: x.squeeze(), memory)
        # memory = jax.tree_map(lambda x: jnp.vstack(x.squeeze()), memory)
        memory = jax.tree_map(lambda x: jnp.array([x]) if x.ndim == 0 else x, memory)
        # memory = jax.tree_map(lambda x: jnp.vstack(x), memory)  # TODO
        # memory = jax.tree_map(lambda x: x[jnp.newaxis,:] if x.ndim == 1 else x, memory)

        # D, (S, D), (L, S, D)

        # If there's a single sample (S=1), add a new trivial S axis: (D,) -> (1, D)
        # if memory.dones.ndim == 1:
        if memory.dones.size == 1:
            memory = jax.tree_map(lambda x: x[jnp.newaxis, :], memory)

        # memory = jax.tree_map(lambda x: jnp.vstack(x) if x.ndim == 1 else x, memory)
        # TODO: vstack necessary??

        memory.check_valid()
        return memory

    def __len__(self) -> int:
        self.check_valid()
        return self.dones.flatten().size

    def __getitem__(self, item) -> "Memory":
        try:
            item = operator.index(item)
        except TypeError:
            pass

        def get_slice(s: Union[slice, np.ndarray, jnp.ndarray, Tuple]) -> Memory:
            return jax.tree_map(lambda x: x[s], self)
            # squeezed = jax.tree_map(lambda x: jnp.squeeze(x[s]), self)
            # return jax.tree_map(lambda x: jnp.vstack(x[s]), squeezed)

        if isinstance(item, int):
            if item > len(self):
                raise IndexError(
                    f"index {item} is out of bounds for axis 0 with size {len(self)}"
                )

            return get_slice(jnp.s_[item])

        if isinstance(item, slice):
            return get_slice(item)

        if isinstance(item, (np.ndarray, jnp.ndarray)):
            return get_slice(item)

        if isinstance(item, tuple):
            return get_slice(item)

        raise TypeError(item)

    def flat_level(self, level: int) -> "Memory":
        if self.flat:
            return self

        if level > self.number_of_levels - 1:
            raise ValueError(level, self.number_of_levels)

        return jax.tree_map(lambda x: x[level], self)

    def has_nans(self) -> jtp.Bool:
        has_nan = jax.tree_map(lambda l: jnp.isnan(l).any(), self)
        return jax.flatten_util.ravel_pytree(has_nan)[0].any()

    def check_valid(self) -> None:
        # if self.has_nans():
        #     raise ValueError("Found NaN values")

        # L, S, D = (None, None, None)

        if self.dones.ndim < 2 or self.dones.ndim > 3:
            raise ValueError(self.dones.shape)

        # if self.dones.ndim == 2:
        #     L, S, D = None, self.dones.shape[0], self.dones.shape[1]
        #
        # if self.dones.ndim == 3:
        #     L, S, D = self.dones[0].shape, self.dones[1].shape, self.dones.shape[2]
        #
        # if D != 1:
        #     raise ValueError(D, self.dones.shape)

        # return True
        shape_of_leaves = jax.tree_map(
            lambda x: x.shape, jax.tree_util.tree_leaves(self)
        )

        # (L, S, D)
        # (S, D)

        # Check (S, ⋅)  TODO: same as check below with L and S? can be removed?
        if self.flat and len(set([s[0:1] for s in shape_of_leaves])) != 1:
            raise ValueError(shape_of_leaves)

        # Check (L, S, ⋅)  TODO: same as check below with L and S? can be removed?
        if not self.flat and len(set([s[0:2] for s in shape_of_leaves])) != 1:
            raise ValueError(shape_of_leaves)

        # Check all leaves have same shape (L,S,⋅)
        # if len(set([s[:-1] for s in shape_of_leaves])) != 1:
        #     raise ValueError(shape_of_leaves)

        # Get the Level and Samples dimensions
        L = self.dones.shape[0] if not self.flat else None
        S = self.dones.shape[1] if not self.flat else self.dones.shape[0]

        # If flat, check that all leaves have S samples
        if self.flat and set(s[0] for s in shape_of_leaves) != {S}:
            raise ValueError(shape_of_leaves)

        # If not flat, check that all leaves have L levels and S samples
        if not self.flat and set(s[0:2] for s in shape_of_leaves) != {(L, S)}:
            raise ValueError(shape_of_leaves)

    def trajectories(self) -> Generator["Memory", None, None]:
        # In this method we operate on non-flat memory
        memory = self if not self.flat else self.unflatten()

        def trajectory_slices_of_level(
            memory: Memory, level: int = 0
        ) -> List[Tuple[npt.NDArray, npt.NDArray, slice]]:
            idx_of_ones, _ = np.where(memory.dones[level] == 1)

            if idx_of_ones.size < 2:
                return []

            start = idx_of_ones[:-1] + 1
            stop = idx_of_ones[1:] + 1

            return [
                np.s_[level, idx_start:idx_stop]
                for idx_start, idx_stop in zip(start, stop)
            ]

        for level in range(memory.number_of_levels):
            for s in trajectory_slices_of_level(memory=memory, level=level):
                yield memory[s]


class DataLoader:
    def __init__(self, memory: Memory):
        self.memory = memory if memory.flat else memory.flatten()

    def batch_slices_generator(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int = None,
        # key: jax.random.PRNGKeyArray = None,
        allow_partial_batch: bool = False,
    ) -> Generator[jtp.ArrayJax, None, None]:
        # Create the index mask
        # mask_indices = jnp.arange(0, len(self.memory), dtype=int)
        mask_indices = np.arange(0, len(self.memory), dtype=int)

        seed = seed if seed is not None else 0
        # key = key if key is not None else jax.random.PRNGKey(seed=seed)

        # When this function is JIT compiled with shuffle=True, the shuffled indices
        # are always the same, according to the seed
        if shuffle:
            rng = np.random.default_rng(seed)
            mask_indices = rng.permutation(mask_indices)

        # mask_indices = (
        #     mask_indices
        #     if shuffle is False
        #     else jax.random.permutation(key=key, x=mask_indices)
        # )

        def boolean_mask_generator(
            a: npt.NDArray, size: int
        ) -> Generator[jtp.ArrayJax, None, None]:
            if a.ndim != 1:
                raise ValueError(a.ndim)

            if size > a.size:
                raise ValueError(size, a.size)

            idx = 0
            mask = jnp.zeros(a.shape, dtype=bool)

            while idx + size <= a.size:
                batch_slice = np.s_[idx : idx + size]
                # yield mask.at[batch_slice].set(True)

                mask = np.zeros(a.shape, dtype=bool)
                mask[batch_slice] = True
                yield mask[np.array(mask_indices)]

                # low, high = idx, idx + size
                # indices = jnp.arange(start=0, stop=mask.size)
                # batch_slice_low = jnp.where(indices >= low, True, False)
                # batch_slice_high = jnp.where(indices < high, True, False)
                # yield batch_slice_low * batch_slice_high

                idx += size

            if allow_partial_batch:
                # batch_slice = np.s_[idx:]
                batch_slice = jnp.s_[idx:]
                yield mask.at[batch_slice].set(True)

        yield from boolean_mask_generator(a=mask_indices, size=batch_size)
