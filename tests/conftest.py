import os

import jax
import pytest
import jaxsim


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


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(jaxsim.VelRepr.Inertial, id="inertial"),
        pytest.param(jaxsim.VelRepr.Body, id="body"),
        pytest.param(jaxsim.VelRepr.Mixed, id="mixed"),
    ],
)
def velocity_representation(request) -> jaxsim.VelRepr:
    """
    Parametrized fixture providing all supported velocity representations.

    Returns:
        A velocity representation.
    """

    return request.param
