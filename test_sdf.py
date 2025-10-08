# %%
import functools
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import rod
import icub_models
import jaxsim
import jaxsim.api as js
import jaxsim.mujoco
from jaxsim import logging

logging.set_logging_level(logging.LoggingLevel.WARNING)
print(f"Running on {jax.devices()}")

# rod_sdf = rod.Sdf(
#     version="1.7",
#     model=SphereBuilder(radius=0.10, mass=1.0, name="sphere")
#     .build_model()
#     .add_link()
#     .add_inertial()
#     .add_visual()
#     .add_collision()
#     .build(),
# )

# rod_sdf = rod.Sdf(
#     version="1.7",
#     model=BoxBuilder(x=0.2, y=0.3, z=0.1, mass=1.0, name="box")
#     .build_model()
#     .add_link()
#     .add_inertial()
#     .add_visual()
#     .add_collision()
#     .build(),
# )
# rod_sdf = rod.Sdf(
#     version="1.7",
#     model=CylinderBuilder(radius=0.1, length=0.3, mass=1.0, name="cylinder")
#     .build_model()
#     .add_link()
#     .add_inertial()
#     .add_visual()
#     .add_collision()
#     .build(),
# )
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# model_path = resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf")
rod_sdf = rod.Sdf.load(model_path)

rod_sdf.model.switch_frame_convention(
    frame_convention=rod.FrameConvention.Urdf, explicit_frames=True
)

# Serialize the model to a SDF string.
model_sdf_string = rod_sdf.serialize(pretty=True)

# %%

JOINT_NAMES = [
    "r_shoulder_pitch",  # 0
    "r_shoulder_roll",  # 1
    "r_shoulder_yaw",  # 2
    "r_elbow",  # 3
    "l_shoulder_pitch",  # 4
    "l_shoulder_roll",  # 5
    "l_shoulder_yaw",  # 6
    "l_elbow",  # 7
    "r_hip_pitch",  # 8
    "r_hip_roll",  # 9
    "r_hip_yaw",  # 10
    "r_knee",  # 11
    "r_ankle_pitch",  # 12
    "r_ankle_roll",  # 13
    "l_hip_pitch",  # 14
    "l_hip_roll",  # 15
    "l_hip_yaw",  # 16
    "l_knee",  # 17
    "l_ankle_pitch",  # 18
    "l_ankle_roll",  # 19
]
input()
model = js.model.JaxSimModel.build_from_model_description(
    model_description=rod_sdf,
    time_step=0.001,
    contact_model=jaxsim.rbda.contacts.SoftContacts(),
)
input()
model = js.model.reduce(model=model, considered_joints=JOINT_NAMES)
exit()
# with model.editable(validate=False) as model:
#     model.contact_params = jaxsim.rbda.contacts.SoftContactsParams(K=1e5, D=1e3, mu=1.0)


data = js.data.random_model_data(
    model=model,
    key=jax.random.PRNGKey(0),
)

# %%
T = jnp.arange(
    start=0, stop=3.0, step=model.time_step
)  # Initialize the simulated time.

# %%

# Create a random JAX key.
key = jax.random.PRNGKey(seed=0)

# Split subkeys for sampling random initial data.
batch_size = 9
row_length = int(jnp.sqrt(batch_size))
row_dist = 0.3 * row_length
key, *subkeys = jax.random.split(key=key, num=batch_size + 1)

# Create the batched data by sampling the height from [0.5, 0.6] meters.
data_batch_t0 = jax.vmap(
    lambda key: js.data.random_model_data(
        model=model,
        key=key,
        base_pos_bounds=([0, 0, 0.8], [0, 0, 0.8]),
        base_vel_lin_bounds=(-0.0, 0.0),
        base_vel_ang_bounds=(-0.0, 0.0),
        base_rpy_bounds=([0, 0, 0], [0, 0, 0]),
        joint_pos_bounds=(0.0, 0.0),
        joint_vel_bounds=(-0.0, 0.0),
    )
)(jnp.vstack(subkeys))

x, y = jnp.meshgrid(
    jnp.linspace(-row_dist, row_dist, num=row_length),
    jnp.linspace(-row_dist, row_dist, num=row_length),
)
xy_coordinate = jnp.stack([x.flatten(), y.flatten()], axis=-1)

# Reset the x and y position to a grid.
data_batch_t0 = data_batch_t0.replace(
    model=model,
    base_position=data_batch_t0.base_position.at[:, :2].set(xy_coordinate),
)


@jax.jit
def step_single(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:

    # Close step over static arguments.
    return js.model.step(
        model=model,
        data=data,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))
def step_parallel(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:

    return step_single(
        model=model,
        data=data,
    )


# %% Compilation
with jax.disable_jit(disable=False):
    _ = step_single(model, data)
    _ = step_parallel(model, data_batch_t0)

# Benchmark the execution of a single step.
print("\nSingle simulation step:")

# %timeit step_single(model, data)

# On hardware accelerators, there's a range of batch_size values where
# increasing the number of parallel instances doesn't affect computation time.
# This range depends on the GPU/TPU specifications.
print(f"\nParallel simulation steps (batch_size={batch_size} on {jax.devices()[0]}):")

# %timeit step_parallel(model, data_batch_t0)

# %%

data_trajectory_list = []

for _ in T:

    print(f"{int(_ * 1000 + 1)}/{len(T)}", end="\r")

    data_batch_t0 = step_parallel(model, data_batch_t0)
    data_trajectory_list.append(data_batch_t0)

# %%
data_trajectory = jax.tree.map(lambda *leafs: jnp.stack(leafs), *data_trajectory_list)

print(f"W_p_B: shape={data_trajectory.base_position.shape}")

# %%
plt.plot(
    T,
    data_trajectory.base_position[:, :, 2],
    label=[f"Box_{k}" for k in range(batch_size)],
)
plt.legend(loc="upper left")
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.title("Height trajectory of the boxes")
plt.show()

# %%
if locals().get("recorder") is not None:
    del recorder

# %%
mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
    model.built_from.model,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="box_cam",
        lookat=[0, 0, 0.3],
        distance=4,
        azimuth=150,
        elevation=-10,
    ),
)

# Create a helper for each parallel instance.
mj_model_helpers = [
    jaxsim.mujoco.MujocoModelHelper.build_from_xml(
        mjcf_description=mjcf_string, assets=assets
    )
    for _ in range(batch_size)
]

# Create the video recorder.
recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_model_helpers[0].model,
    data=[helper.data for helper in mj_model_helpers],
    fps=60,
    width=320 * 2,
    height=240 * 2,
)


# viz = MujocoMultiVisualizer(
#     model=[helper.model for helper in mj_model_helpers],
#     data=[helper.data for helper in mj_model_helpers],
# )

# with viz.open() as viewer:
#     with viewer.lock():
#         for data_t in data_trajectory_list[:: int(1 / model.time_step / 60)]:
#             for helper, base_position, base_quaternion, joint_position in zip(
#                 mj_model_helpers,
#                 data_t.base_position,
#                 data_t.base_orientation,
#                 data_t.joint_positions,
#                 strict=True,
#             ):
#                 helper.set_base_position(position=base_position)
#                 helper.set_base_orientation(orientation=base_quaternion)
#                 if model.dofs() > 0:
#                     helper.set_joint_positions(
#                         positions=joint_position, joint_names=model.joint_names()
#                     )

#             viz.sync(viewer=viewer)

#     while viewer.is_running():
#         time.sleep(0.5)

for data_t in data_trajectory_list[:: int(1 / model.time_step / 60)]:
    try:
        for helper, base_position, base_quaternion, joint_position in zip(
            mj_model_helpers,
            data_t.base_position,
            data_t.base_orientation,
            data_t.joint_positions,
            strict=True,
        ):
            helper.set_base_position(position=base_position)
            helper.set_base_orientation(orientation=base_quaternion)

            if model.dofs() > 0:
                helper.set_joint_positions(
                    positions=joint_position, joint_names=model.joint_names()
                )

        # Record a new video frame.
        recorder.record_frame(camera_name="box_cam")
    finally:
        pass

# %%

media.show_video(recorder.frames, fps=recorder.fps)


# %%
