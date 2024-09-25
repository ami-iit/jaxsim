import os
import pathlib

import jax
import pytest
import rod
import rod.urdf
import rod.urdf.exporter

import jaxsim
import jaxsim.api as js


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


# ================================
# Fixtures providing JaxSim models
# ================================

# All the fixtures in this section must have "session" scope.
# In this way, the models are generated only once and shared among all the tests.


# This is not a fixture.
def build_jaxsim_model(
    model_description: str | pathlib.Path | rod.Model,
) -> js.model.JaxSimModel:
    """
    Helper to build a JaxSim model from a model description.

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

    import rod.builder.primitives
    import rod.urdf.exporter

    # Create on-the-fly a ROD model of a box.
    rod_model = (
        rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=1.0, name="box")
        .build_model()
        .add_link(name="box_link")
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    rod_model.add_frame(
        rod.Frame(
            name="box_frame",
            attached_to="box_link",
            pose=rod.Pose(relative_to="box_link", pose=[1, 1, 1, 0.5, 0.4, 0.3]),
        )
    )

    # Export the URDF string.
    urdf_string = rod.urdf.exporter.UrdfExporter(
        pretty=True, gazebo_preserve_fixed_joints=True
    ).to_urdf_string(sdf=rod_model)

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def jaxsim_model_sphere() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a sphere.

    Returns:
        The JaxSim model of a sphere.
    """

    import rod.builder.primitives
    import rod.urdf.exporter

    # Create on-the-fly a ROD model of a sphere.
    rod_model = (
        rod.builder.primitives.SphereBuilder(radius=0.1, mass=1.0, name="sphere")
        .build_model()
        .add_link()
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    # Export the URDF string.
    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model
    )

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def ergocub_model_description_path() -> pathlib.Path:
    """
    Fixture providing the path to the URDF model description of the ErgoCub robot.

    Returns:
        The path to the URDF model description of the ErgoCub robot.
    """

    try:
        os.environ["ROBOT_DESCRIPTION_COMMIT"] = "v0.7.1"

        import robot_descriptions.ergocub_description

    finally:
        _ = os.environ.pop("ROBOT_DESCRIPTION_COMMIT", None)

    model_urdf_path = pathlib.Path(
        robot_descriptions.ergocub_description.URDF_PATH.replace(
            "ergoCubSN000", "ergoCubSN001"
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


# ============================
# Collections of JaxSim models
# ============================


def get_jaxsim_model_fixture(
    model_name: str, request: pytest.FixtureRequest
) -> str | pathlib.Path:
    """
    Factory to get the fixture providing the JaxSim model of a robot.

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
