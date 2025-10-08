import pytest
import sdformat

from jaxsim.mujoco import ModelToMjcf
from jaxsim.mujoco.loaders import MujocoCamera


@pytest.fixture
def mujoco_camera():

    return MujocoCamera.build_from_target_view(
        camera_name="test_camera",
        lookat=(0, 0, 0),
        distance=1,
        azimuth=0,
        elevation=0,
        fovy=45,
        degrees=True,
    )


def test_urdf_loading(jaxsim_model_single_pendulum, mujoco_camera):
    model = jaxsim_model_single_pendulum.built_from

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_sdf_loading(jaxsim_model_single_pendulum, mujoco_camera):

    root = sdformat.Root()
    root.load(str(jaxsim_model_single_pendulum.built_from))
    model = root.to_string()

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_sdformat_loading(jaxsim_model_single_pendulum, mujoco_camera):

    root = sdformat.Root()
    root.load(str(jaxsim_model_single_pendulum.built_from))
    model = root.model_by_index(0)

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_heightmap(jaxsim_model_single_pendulum, mujoco_camera):

    root = sdformat.Root()
    root.load(str(jaxsim_model_single_pendulum.built_from))
    model = root.model_by_index(0)

    _ = ModelToMjcf.convert(
        model=model,
        cameras=mujoco_camera,
        heightmap=True,
        heightmap_samples_xy=(51, 51),
    )


def test_inclined_plane(jaxsim_model_single_pendulum, mujoco_camera):

    root = sdformat.Root()
    root.load(str(jaxsim_model_single_pendulum.built_from))
    model = root.model_by_index(0)

    _ = ModelToMjcf.convert(
        model=model,
        cameras=mujoco_camera,
        plane_normal=(0.3, 0.3, 0.3),
    )
