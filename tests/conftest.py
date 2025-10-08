import os

os.environ["JAXSIM_ENABLE_EXCEPTIONS"] = "1"

import pathlib
import subprocess

import jax
import pytest
import sdformat

import jaxsim
import jaxsim.api as js
from jaxsim.api.model import IntegratorType


def pytest_addoption(parser):
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Run tests only if GPU is available and utilized",
    )

    parser.addoption(
        "--batch-size",
        action="store",
        default="None",
        help="Batch size for vectorized benchmarks (only applies to benchmark tests)",
    )


def pytest_generate_tests(metafunc):
    if (
        "batch_size" in metafunc.fixturenames
        and (batch_size := metafunc.config.getoption("--batch-size")) != "None"
    ):
        metafunc.parametrize("batch_size", [1, int(batch_size)])


def check_gpu_usage():
    # Set environment variable to prioritize GPU.
    os.environ["JAX_PLATFORM_NAME"] = "gpu"

    # Run a simple JAX operation
    x = jax.device_put(jax.numpy.ones((512, 512)))
    y = jax.device_put(jax.numpy.ones((512, 512)))
    _ = jax.numpy.dot(x, y).block_until_ready()

    # Check GPU memory usage with nvidia-smi.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.exit(
            "Failed to query GPU usage. Ensure nvidia-smi is installed and accessible."
        )

    gpu_memory_usage = [
        int(line.strip().split()[0]) for line in result.stdout.splitlines()
    ]
    if all(usage == 0 for usage in gpu_memory_usage):
        pytest.exit(
            "GPU is available but not utilized during computations. Check your JAX installation."
        )


def pytest_configure(config) -> None:
    """Pytest configuration hook."""

    # This is a global variable that is updated by the `prng_key` fixture.
    pytest.prng_key = jax.random.PRNGKey(
        seed=int(os.environ.get("JAXSIM_TEST_SEED", 0))
    )

    # Check if GPU is available and utilized.
    if config.getoption("--gpu-only"):
        devices = jax.devices()
        if not any(device.platform == "gpu" for device in devices):
            pytest.exit("No GPU devices found. Check your JAX installation.")

        # Ensure GPU is being used during computation
        check_gpu_usage()


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


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(IntegratorType.SemiImplicitEuler, id="semi_implicit_euler"),
        pytest.param(IntegratorType.RungeKutta4, id="runge_kutta_4"),
        pytest.param(IntegratorType.RungeKutta4Fast, id="runge_kutta_4_fast"),
    ],
)
def integrator(request) -> str:
    """
    Fixture providing the integrator to use in the simulation.

    Returns:
        The integrator to use in the simulation.
    """

    return request.param


@pytest.fixture(scope="session")
def batch_size(request) -> int:
    """
    Fixture providing the batch size for vectorized benchmarks.

    Returns:
        The batch size for vectorized benchmarks.
    """

    return 1


# ================================
# Fixtures providing JaxSim models
# ================================

# All the fixtures in this section must have "session" scope.
# In this way, the models are generated only once and shared among all the tests.


# This is not a fixture.
def build_jaxsim_model(
    model_description: str | pathlib.Path | sdformat.Model,
) -> js.model.JaxSimModel:
    """
    Build a JaxSim model from a model description.

    Args:
        model_description: A model description provided by any fixture provider.

    Returns:
        A JaxSim model built from the provided description.
    """

    # Build the JaxSim model.
    model = js.model.JaxSimModel.build_from_model_description(
        model_description=model_description,
    )

    return model


@pytest.fixture(scope="session")
def jaxsim_model_box() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a box.

    Returns:
        The JaxSim model of a box.
    """

    # Use the SDF file for the box model
    model_path = pathlib.Path(__file__).parent / "assets" / "box.sdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_sphere() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a sphere.

    Returns:
        The JaxSim model of a sphere.
    """

    # Use the SDF file for the sphere model
    model_path = pathlib.Path(__file__).parent / "assets" / "sphere.sdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def ergocub_model_description_path() -> pathlib.Path:
    """
    Fixture providing the path to the URDF model description of the ErgoCub robot.

    Returns:
        The path to the URDF model description of the ErgoCub robot.

    """

    try:
        os.environ["ROBOT_DESCRIPTION_COMMIT"] = "v0.7.7"

        import robot_descriptions.ergocub_description

    finally:
        _ = os.environ.pop("ROBOT_DESCRIPTION_COMMIT", None)

    model_urdf_path = pathlib.Path(
        robot_descriptions.ergocub_description.URDF_PATH.replace(
            "ergoCubSN002", "ergoCubSN001"
        )
    )

    return model_urdf_path


@pytest.fixture(scope="session")
def jaxsim_model_ergocub(
    ergocub_model_description_path: pathlib.Path,
) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the ErgoCub robot.

    Returns:
        The JaxSim model of the ErgoCub robot.

    """

    return build_jaxsim_model(model_description=ergocub_model_description_path)


@pytest.fixture(scope="session")
def jaxsim_model_ergocub_reduced(jaxsim_model_ergocub) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the ErgoCub robot with only locomotion joints.

    Returns:
        The JaxSim model of the ErgoCub robot with only locomotion joints.

    """

    model_full = jaxsim_model_ergocub

    # Get the names of the joints to keep.
    reduced_joints = tuple(
        j
        for j in model_full.joint_names()
        if "camera" not in j
        # Remove head and hands.
        and "neck" not in j
        and "wrist" not in j
        and "thumb" not in j
        and "index" not in j
        and "middle" not in j
        and "ring" not in j
        and "pinkie" not in j
        # Remove upper body.
        and "torso" not in j and "elbow" not in j and "shoulder" not in j
    )

    model = js.model.reduce(model=model_full, considered_joints=reduced_joints)

    return model


@pytest.fixture(scope="session")
def jaxsim_model_ur10() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the UR10 robot.

    Returns:
        The JaxSim model of the UR10 robot.

    """

    import robot_descriptions.ur10_description

    model_urdf_path = pathlib.Path(robot_descriptions.ur10_description.URDF_PATH)

    return build_jaxsim_model(model_description=model_urdf_path)


@pytest.fixture(scope="session")
def jaxsim_model_single_pendulum() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a single pendulum.

    Returns:
        The JaxSim model of a single pendulum.
    """

    # Use the SDF file for the single pendulum model
    model_path = pathlib.Path(__file__).parent / "assets" / "single_pendulum.sdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_garpez() -> js.model.JaxSimModel:
    """Fixture to create the original (unscaled) Garpez model."""

    # Use existing garpez.urdf file in workspace root
    model_path = pathlib.Path(__file__).parent.parent / "garpez.urdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_garpez_scaled(request) -> js.model.JaxSimModel:
    """Fixture to create the scaled version of the Garpez model."""

    # For now, just use the original garpez model
    # Scaling functionality would need to be reimplemented with SDFormat
    model_path = pathlib.Path(__file__).parent.parent / "garpez.urdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_double_pendulum() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a double pendulum.
    Returns:
        The JaxSim model of a double pendulum.
    """

    model_path = pathlib.Path(__file__).parent / "assets" / "double_pendulum.sdf"
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_cartpole() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a cartpole.
    Returns:
        The JaxSim model of a cartpole.
    """

    model_path = (
        pathlib.Path(__file__).parent.parent / "examples" / "assets" / "cartpole.urdf"
    )
    return build_jaxsim_model(model_description=model_path)


@pytest.fixture(scope="session")
def jaxsim_model_4_bar_linkage() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a 4-bar linkage (opened configuration).

    Returns:
        The JaxSim model of the 4-bar linkage.
    """

    model_path = pathlib.Path(__file__).parent / "assets" / "4_bar_opened.urdf"
    return build_jaxsim_model(model_description=model_path)


# ============================
# Collections of JaxSim models
# ============================


def get_jaxsim_model_fixture(
    model_name: str, request: pytest.FixtureRequest
) -> str | pathlib.Path:
    """
    Get the fixture providing the JaxSim model of a robot.

    Args:
        model_name: The name of the model.
        request: The request object.

    Returns:
        The JaxSim model of the robot.

    """

    match model_name:
        case "box":
            return request.getfixturevalue(jaxsim_model_box.__name__)
        case "sphere":
            return request.getfixturevalue(jaxsim_model_sphere.__name__)
        case "ergocub":
            return request.getfixturevalue(jaxsim_model_ergocub.__name__)
        case "ergocub_reduced":
            return request.getfixturevalue(jaxsim_model_ergocub_reduced.__name__)
        case "ur10":
            return request.getfixturevalue(jaxsim_model_ur10.__name__)
        case "single_pendulum":
            return request.getfixturevalue(jaxsim_model_single_pendulum.__name__)
        case "garpez":
            return request.getfixturevalue(jaxsim_model_garpez.__name__)
        case "garpez_scaled":
            return request.getfixturevalue(jaxsim_model_garpez_scaled.__name__)
        case _:
            raise ValueError(model_name)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "sphere",
        "ur10",
        "ergocub",
        "ergocub_reduced",
    ],
)
def jaxsim_models_all(request) -> pathlib.Path | str:
    """
    Fixture providing the JaxSim models of all supported robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "ur10",
        "ergocub_reduced",
    ],
)
def jaxsim_models_types(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of all types of supported robots.

    Note:
        At the moment, most of our tests use this fixture. It provides:
        - A robot with no joints.
        - A fixed-base robot.
        - A floating-base robot.

    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "sphere",
    ],
)
def jaxsim_models_no_joints(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of robots with no joints.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "ergocub",
        "ergocub_reduced",
    ],
)
def jaxsim_models_floating_base(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of floating-base robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "ur10",
    ],
)
def jaxsim_models_fixed_base(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of fixed-base robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(scope="function")
def set_jax_32bit(monkeypatch):
    """
    Fixture that temporarily sets JAX precision to 32-bit for the duration of the test.
    """

    del globals()["jaxsim"]
    del globals()["js"]

    # Temporarily disable x64
    monkeypatch.setenv("JAX_ENABLE_X64", "0")


@pytest.fixture(scope="function")
def jaxsim_model_box_32bit(set_jax_32bit, request) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a box with 32-bit precision.

    Returns:
        The JaxSim model of a box with 32-bit precision.

    """

    return get_jaxsim_model_fixture(model_name="box", request=request)
