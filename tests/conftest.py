import os

import jax
import pytest


def pytest_configure() -> None:
    """Pytest configuration hook."""

    # This is a global variable that is updated by the `prng_key` fixture.
    pytest.prng_key = jax.random.PRNGKey(
        seed=int(os.environ.get("JAXSIM_TEST_SEED", 0))
    )


# ================
# Generic fixtures
# ================


@pytest.fixture(scope="function")
def prng_key() -> jax.Array:
    """
    Fixture to generate a new PRNG key for each test function.

    Returns:
        The new PRNG key passed to the test.

    Note:
        This fixture operates on a global variable initialized in the
        `pytest_configure` hook.
    """

    pytest.prng_key, subkey = jax.random.split(pytest.prng_key, num=2)
    return subkey
