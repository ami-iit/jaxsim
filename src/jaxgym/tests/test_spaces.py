import jax.numpy as jnp
import jax.random
import jax_dataclasses
import pytest

import jaxsim.typing as jtp
from jaxgym import spaces


def compare(low: jtp.PyTree, high: jtp.PyTree, box: spaces.Box) -> None:
    """"""

    assert box.contains(x=low)
    assert box.contains(x=high)

    key = jax.random.PRNGKey(seed=0)

    for _ in range(10):
        key, subkey = jax.random.split(key=key, num=2)
        sample = box.sample(key=subkey)
        assert box.contains(x=sample)


def test_box_numpy() -> None:
    """"""

    low = jnp.zeros(10)
    high = jnp.ones(10)

    box = spaces.Box(low=low, high=high)

    compare(low=low, high=high, box=box)

    assert box.contains(x=0.5 * jnp.ones_like(low))
    assert not box.contains(x=1.5 * jnp.ones_like(low))
    assert not box.contains(x=-0.5 * jnp.ones_like(low))

    with pytest.raises(ValueError):
        _ = spaces.Box(low=low, high=jnp.ones(low.size + 1))

    with pytest.raises(ValueError):
        _ = spaces.Box(low=low, high=jnp.ones(low.size, dtype=int))


def test_box_pytree() -> None:
    """"""

    @jax_dataclasses.pytree_dataclass
    class SimplePyTree:
        flag: jtp.Bool
        value: jtp.Float
        position: jtp.Vector
        velocity: jtp.Vector

        @staticmethod
        def zero() -> "SimplePyTree":
            return SimplePyTree(
                flag=False,
                value=0,
                position=jnp.zeros(5),
                velocity=jnp.zeros(10),
            )

    zero = SimplePyTree.zero()

    low = SimplePyTree(
        flag=False,
        value=-42.0,
        position=-10.0 * jnp.ones_like(zero.position),
        velocity=0.1 * jnp.ones_like(zero.velocity),
    )

    high = SimplePyTree(
        flag=True,
        value=42.0,
        position=10.0 * jnp.ones_like(zero.position),
        velocity=5.0 * jnp.ones_like(zero.velocity),
    )

    box = spaces.Box(low=low, high=high)

    compare(low=low, high=high, box=box)

    # Wrong dimension of 'position'
    with pytest.raises(ValueError):
        wrong_high = SimplePyTree(
            flag=True,
            value=42.0,
            position=jnp.zeros(6),
            velocity=jnp.zeros(10),
        )
        _ = spaces.Box(low=low, high=wrong_high)

    # Wrong type of 'position' and 'value'
    with pytest.raises(ValueError):
        wrong_high = SimplePyTree(
            flag=True,
            value=int(42),
            position=jnp.zeros(5, dtype=int),
            velocity=jnp.zeros(10),
        )
        _ = spaces.Box(low=low, high=wrong_high)
