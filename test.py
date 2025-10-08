# %%
import jax.numpy as jnp
import rod.builder.primitives
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["JAXSIM_LOGGING_LEVEL"] = "DEBUG"

import jaxsim.api as js

rod_model = rod.Sdf.load("garpez.urdf").model

model = js.model.JaxSimModel.build_from_model_description(
    model_description=rod_model  # , contact_model=SoftContacts()
)

import jax

# Simulate the sphere falling under gravity
data = js.data.JaxSimModelData.build(
    model=model,
    joint_positions=jnp.array([0.8, 0.0, 0.0]),
)
num_steps = 3000  # Number of simulation steps

import pathlib

import jaxsim.mujoco

mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
    rod_model,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="sphere_cam",
        lookat=[0, 0, 0.3],
        distance=4,
        azimuth=150,
        elevation=-10,
    ),
)

# Create a helper for each parallel instance.
mj_helper = jaxsim.mujoco.MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string, assets=assets
)

# Create the video recorder.
recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_helper.model,
    data=mj_helper.data,
    fps=int(1 / model.time_step),
    width=320 * 2,
    height=240 * 2,
)

# %%

for _ in range(num_steps):
    with jax.disable_jit(False):

        data = js.model.step(
            model=model,
            data=data,
        )

    # mj_helper.set_base_position(position=data.base_position)
    # mj_helper.set_base_orientation(orientation=data.base_quaternion)
    mj_helper.set_joint_positions(
        joint_names=model.joint_names(), positions=data.joint_positions
    )

    recorder.record_frame(camera_name="sphere_cam")

recorder.write_video(pathlib.Path("garpez_relaxed.mp4"))

# %%
