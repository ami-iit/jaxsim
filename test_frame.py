import os

import jax
import jax.numpy as jnp
import jaxlie
import rod.builder.primitives

import jaxsim.api as js

rod_model = (
    rod.builder.primitives.BoxBuilder(x=0.8, y=0.3, z=0.1, mass=1.0, name="box")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build()
)

model = js.model.JaxSimModel.build_from_model_description(model_description=rod_model)

import pathlib

import jaxsim.mujoco

mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
    model.built_from,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="box_cam",
        lookat=[0, 0, 0.5],
        distance=2,
        azimuth=150,
        elevation=-10,
    ),
)

os.environ["JAXSIM_VISUALIZER_FRAME"] = "0.5 0.5 0.5 0 3 5"

# Create a helper for each parallel instance.
mj_helper = jaxsim.mujoco.MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string, assets=assets
)

# Create the video recorder.
recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_helper.model,
    data=mj_helper.data,
    fps=1,
    width=320 * 2,
    height=240 * 2,
)

subkey = jax.random.PRNGKey(0)

mj_helper.set_base_position(jnp.array([0.0, 0.0, 0.5]))

for _ in range(10):

    _, subkey = jax.random.split(subkey)

    random_orientation = jaxlie.SO3.sample_uniform(subkey).wxyz

    rpy_orientation = jaxlie.SO3(wxyz=random_orientation).as_rpy_radians()

    os.environ["JAXSIM_VISUALIZER_FRAME"] = "0.5 0.5 0.5 " + " ".join(
        map(str, rpy_orientation)
    )

    mj_helper.set_base_orientation(orientation=random_orientation)

    recorder.record_frame(camera_name="box_cam")

recorder.write_video(pathlib.Path("box_frame_test.mp4"))
