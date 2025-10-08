import os
from pathlib import Path

import numpy as np
import rod
from rod.builder.primitives import BoxBuilder, PrimitiveBuilder

import jaxsim
import jaxsim.api as js
import jaxsim.mujoco

os.environ["MUJOCO_GL"] = "egl"

base_height = 2.15
upper_height = 1.0

# ===================
# Create the builders
# ===================

base_builder = BoxBuilder(
    name="base",
    mass=1.0,
    x=0.15,
    y=0.15,
    z=base_height,
)

upper_builder = BoxBuilder(
    name="upper",
    mass=0.5,
    x=0.15,
    y=0.15,
    z=upper_height,
)

# =================
# Create the joints
# =================

fixed = rod.Joint(
    name="fixed_joint",
    type="fixed",
    parent="world",
    child=base_builder.name,
)

pivot = rod.Joint(
    name="upper_joint",
    type="revolute",
    parent=base_builder.name,
    child=upper_builder.name,
    axis=rod.Axis(
        xyz=rod.Xyz([1, 0, 0]),
       ),
    )

# ================
# Create the links
# ================

base = (
    base_builder.build_link(
        name=base_builder.name,
        pose=PrimitiveBuilder.build_pose(pos=np.array([0, 0, base_height / 2])),
    )
    .add_inertial()
    .add_visual()
    .add_collision()
    .build()
)

upper_pose = PrimitiveBuilder.build_pose(pos=np.array([0, 0, upper_height / 2]))

upper = (
    upper_builder.build_link(
        name=upper_builder.name,
        pose=PrimitiveBuilder.build_pose(
            relative_to=base.name, pos=np.array([0, 0, upper_height])
        ),
    )
    .add_inertial(pose=upper_pose)
    .add_visual(pose=upper_pose)
    .add_collision(pose=upper_pose)
    .build()
)

rod_model = rod.Sdf(
    version="1.10",
    model=rod.Model(
        name="single_pendulum",
        link=[base, upper],
        joint=[fixed, pivot],
    ),
)

rod_model.model.resolve_frames()

model = js.model.JaxSimModel.build_from_model_description(
    model_description=rod_model,
    time_step=0.01,
    terrain=jaxsim.terrain.FlatTerrain.build(height=-1e3),
)

data = js.data.JaxSimModelData.build(model=model, joint_positions=0.5)

mjcf_string, assets = jaxsim.mujoco.loaders.RodModelToMjcf.convert(
    rod_model=rod_model.model,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="pendulum_camera",
        lookat=js.link.com_position(
            model=model,
            data=data,
            link_index=js.link.name_to_idx(model=model, link_name="base"),
            in_link_frame=False,
        ),
        distance=3,
        azimuth=150,
        elevation=-10,
    ),
)

mj_model_helper = jaxsim.mujoco.model.MujocoModelHelper.build_from_xml(
    mjcf_description=mjcf_string,
    assets=assets,
)

recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_model_helper.model,
    data=mj_model_helper.data,
    fps=int(1 / model.time_step),
    width=320 * 2,
    height=240 * 2,
)

joint_positions = []

for _ in range(1000):
    data = js.model.step(
        model=model,
        data=data,
    )

    joint_positions.append(data.joint_positions)
    recorder.record_frame(camera_name="pendulum_camera")

    mj_model_helper.set_joint_positions(
        joint_names=model.joint_names(), positions=data.joint_positions
    )

    print(f"Step: {_}/1000, Joint Position: {data.joint_positions}", end="\r")

recorder.write_video(path=Path.cwd() / Path("single_pendulum_viscous.mp4"), exist_ok=True)
