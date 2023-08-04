import pathlib
from typing import Dict, Tuple

import gym_ignition_models
import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim
import jaxsim.typing as jtp
from jaxsim import high_level
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData

# TODO:
#
# - contatti con metodi old sembrano ok
# - calcolare x con metodi old e testare quelli new?
# - (optional: migrare a nuovo jax se fixo a porto ABA fuori da exp fors?)
# v rifare modello ant con rod senza capsule e box collision per torso, sfere solo piedi
# - trovare parametri buoni per ant e creare environment
# - usare gymnasium e pytorch rl per training

# ================================
# Configure and run the simulation
# ================================

# model_sdf_path = pathlib.Path(
#     gym_ignition_models.get_model_resource(
#         # robot_name="iCubGazeboSimpleCollisionsV2_5",
#         robot_name="iCubGazeboV2_5",
#         resource_type=gym_ignition_models.ResourceType.URDF_PATH,
#     )
# )

# model_sdf_path = (
#     # pathlib.Path.home() / "git" / "jaxsim" / "examples" / "resources" / "cube.urdf"
#     pathlib.Path.home()
#     / "git"
#     / "jaxsim"
#     / "examples"
#     / "resources"
#     # / "double_box.urdf"
#     # / "sphere.urdf"
#     / "sphere.sdf"
# )

# model_sdf_path = (
#     pathlib.Path.home()
#     / "git"
#     / "isaac"
#     / "isaac-wp1-5"
#     / "URDF-Robot-Models"
#     / "ISC5RU_Chelsea"
#     / "ISC5RU_Chelsea.urdf"
# )

# model_sdf_path = (
#     pathlib.Path.home()
#     / "git"
#     / "isaac"
#     / "isaac-wp1-5"
#     / "URDF-Robot-Models"
#     / "ISC5RU_Hopper"
#     / "ISC5RU_Hopper.urdf"
# )

model_sdf_path = (
    pathlib.Path.home() / "git" / "jaxsim" / "examples" / "resources" / "ant.sdf"
)

assert model_sdf_path.exists()

# Simulation step parameters
step_size = 0.001
steps_per_run = 1

# Create the simulator
simulator = JaxSim.build(
    step_size=step_size,
    steps_per_run=steps_per_run,
    velocity_representation=high_level.model.VelRepr.Body,
    integrator_type=jaxsim.IntegratorType.EulerSemiImplicit,
    simulator_data=SimulatorData(
        # contact_parameters=SoftContactsParams(K=1e6, D=2e3, mu=0.5)
        contact_parameters=SoftContactsParams(K=10_000, D=20.0, mu=0.5)
        # contact_parameters=SoftContactsParams(K=5_000.0, D=10.0, mu=0.5)
    ),
).mutable(validate=False)

# Insert the model and get a mutable object
model = simulator.insert_model_from_description(model_description=model_sdf_path)

# Remove useless joints and lump inertial parameters of their connecting links
model.reduce(
    considered_joints=[
        name
        for name in model.joint_names()
        if "wrist" not in name and "neck" not in name
    ]
)

# Zero the model data
model.zero()

# Reset base quantities
model.reset_base_position(position=jnp.array([0.0, 0, 1.5]))
# model.reset_base_position(position=jnp.array([0.0, 0, 0.5]))
# model.reset_base_position(position=jnp.array([0.0, 0, 1.5]))
# model.reset_base_velocity(base_velocity=jnp.array([0.6, 0, 0, 0, 0, 0]))
model.reset_base_velocity(base_velocity=jnp.array([0.0, 0, 0, 0, 0, 0]))

# Reset joint quantities
s = model.joint_random_positions()
model.reset_joint_positions(positions=s)


# Create a logger class used as callback to extract data from an
# open-loop simulation over a given horizon
@jax_dataclasses.pytree_dataclass
class SimulatorLogger(simulator_callbacks.PostStepCallback):
    def post_step(
        self, sim: JaxSim, step_data: Dict[str, StepData]
    ) -> Tuple[JaxSim, jtp.PyTree]:
        # Return the StepData of each simulated model
        return sim, step_data


# Instantiate the logger specifying the target model
cb = SimulatorLogger()

# Store the initial state (useful for visualization purpose)
x0 = model.data.model_state

# Simulate 3.0 seconds
simulator, (cb, step_data) = simulator.step_over_horizon(
    horizon_steps=int(3.0 / simulator.dt()),
    callback_handler=cb,
    clear_inputs=True,
)

# simulator, (cb, step_data) = simulator.step_over_horizon(horizon_steps=int(3.0 / simulator.dt()), callback_handler=cb, clear_inputs=True)

# Extract the PhysicsModelState over the simulated horizon
step_data: Dict[str, StepData]
x = step_data[model.name()].tf_model_state

# Now you can inspect x and plot its data

# ========================
# Plot the simulation data
# ========================

import matplotlib.pyplot as plt

plt.plot(step_data[model.name()].tf, x.base_position, label=["x", "y", "z"])
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Trajectory of the model's base")
plt.show()

#
#
#

import time

import numpy as np
from loop_rate_limiters import RateLimiter
from meshcat_viz import MeshcatWorld
from meshcat_viz.contacts import ModelContacts

from jaxsim.physics.algos.soft_contacts import collidable_points_pos_vel
from jaxsim.sixd import se3

# Open the visualizer
world = MeshcatWorld()
world.open()

# Insert the model from a URDF/SDF resource
model_name = world.insert_model(
    model_description=model_sdf_path, is_urdf=model_sdf_path.suffix.endswith("urdf")
)
# world.remove_model(model_name=model_name)

# Compute the positions of all collidable points
# pos_cp = jax.jit(
#     lambda m, s, W_p_B, W_Q_B: collidable_points_pos_vel(
#         m, s, jnp.zeros_like(s), jnp.hstack([W_Q_B, W_p_B, jnp.zeros(6)])
#     )[0].T
# )

# Create the interaction forces object
contacts = ModelContacts(meshcat_model=world._meshcat_models[model_name])

# Add to the visualizer the contact forces
# for frame in model.link_names():
#     contacts.add_contact_frame(
#         contact_name=frame,
#         relative_to=world._meshcat_models[model_name].link_to_node[frame],
#         enable_force=True,
#         force_scale=0.025 / 1,  # 2.5cm / 1 N
#         force_radius=0.001,
#     )

import numpy.typing as npt


def local_contact_6D_forces(model: high_level.model.Model, W_f_cp):
    W_HH_L = model.forward_kinematics()
    L_HH_CL = jax.vmap(lambda L_p_cp: jnp.eye(4).at[0:3, 3].set(L_p_cp.squeeze()))(
        model.physics_model.gc.point.T
    )

    W_HH_CL = jax.vmap(lambda L_H_CL, idx: W_HH_L[idx] @ L_H_CL)(
        L_HH_CL, model.physics_model.gc.body
    )

    W_XXv_CL = jax.vmap(lambda W_H_CL: se3.SE3.from_matrix(W_H_CL).adjoint())(W_HH_CL)

    return jax.vmap(lambda W_Xv_CL, W_f: W_Xv_CL.T @ W_f)(W_XXv_CL, W_f_cp)


# def local_contact_6D_forces(model: high_level.model.Model, W_f_cp):
#
#     W_HH_L = model.forward_kinematics()
#     # L_HH_CL = jax.vmap(
#     #     lambda L_p_cp: jnp.eye(4).at[0:3, 3].set(L_p_cp.squeeze())
#     # )(model.physics_model.gc.point.T)
#     #
#     # W_HH_CL = jax.vmap(
#     #     lambda L_H_CL, idx: W_HH_L[idx] @ L_H_CL
#     # )(L_HH_CL, model.physics_model.gc.body)
#
#     W_HH_CW = jax.vmap(
#         lambda L_p_cp, idx: W_HH_L[idx].at[0:3, 0:3].set(jnp.eye(3)) @ jnp.eye(4).at[0:3, 3].set(L_p_cp.squeeze())
#     )(model.physics_model.gc.point.T, model.physics_model.gc.body)
#
#     W_XXv_CW = jax.vmap(lambda W_H_CW: se3.SE3.from_matrix(W_H_CW).adjoint())(W_HH_CW)
#
#     return jax.vmap(lambda W_Xv_CW, W_f: W_Xv_CW.T @ W_f)(W_XXv_CW, W_f_cp)


@jax.jit
def world_to_viz_contact_forces(model, W_p_B, W_Q_B, s, W_f_cp):
    m = model.copy()

    m.zero()
    m.reset_joint_positions(positions=s)
    m.reset_base_position(position=W_p_B)
    m.reset_base_orientation(orientation=W_Q_B)

    return local_contact_6D_forces(m, W_f_cp)


index_to_link_name = {link.index(): link.name() for link in model.links()}
parent_link_name = [index_to_link_name[idx] for idx in model.physics_model.gc.body]

# for L_p_cp, idx in zip(model.physics_model.gc.point.T, model.physics_model.gc.body):
for idx, (L_p_cp, name) in enumerate(
    zip(model.physics_model.gc.point.T, parent_link_name)
):
    L_H_CL = np.block([[np.eye(3), np.vstack(L_p_cp)], [0, 0, 0, 1]])

    contacts.add_contact_frame(
        contact_name=f"{name}_{idx}",
        transform=L_H_CL,
        relative_to=world._meshcat_models[model_name].link_to_node[name],
        enable_force=True,
        force_scale=0.025 / 1,  # 2.5cm / 1 N
        force_radius=0.001,
    )

# Initialize the base position
world.update_model(
    model_name=model_name,
    joint_names=model.joint_names(),
    joint_positions=np.array(x0.joint_positions),
    base_position=np.array(x0.base_position),
    base_quaternion=np.array(x0.base_quaternion),
)
# world.update_model(
#     model_name=model_name,
#     joint_names=model.joint_names(),
#     joint_positions=np.zeros_like(x0.joint_positions),
#     base_position=np.array([0, 0, 1.5]),
#     base_quaternion=np.array([1, 0, 0, 0]),
# )

# Function to compute FK
fk = jax.jit(lambda model: model.forward_kinematics())
time.sleep(2)

# rtf = 0.25
# downsampling = 10
rtf = 1.0
downsampling = 25
rate = RateLimiter(frequency=float(rtf / (simulator.dt() * downsampling)))

for s, W_p_B, W_Q_B, W_f_ext, W_fc_ext in list(
    zip(
        x.joint_positions,
        x.base_position,
        x.base_quaternion,
        step_data[model.name()].aux["t0"]["contact_forces_links"],
        step_data[model.name()].aux["t0"]["contact_forces_points"],
    )
)[::downsampling]:
    now = time.time()

    world.update_model(
        model_name=model_name,
        joint_names=model.joint_names(),
        joint_positions=s,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
    )

    # Positions of all collidable points
    # pos = pos_cp(model.physics_model, s, W_p_B, W_Q_B)
    # print(time.time() - now)

    CL_fc_ext = world_to_viz_contact_forces(model, W_p_B, W_Q_B, s, W_fc_ext)[:, 0:3]

    for idx, (CL_fc, name) in enumerate(zip(CL_fc_ext, parent_link_name)):
        # print(f"{name}_{idx}", np.array(CL_fc))
        contacts[f"{name}_{idx}"].force.set(force=np.array(CL_fc))
        # contacts[f"{name}_{idx}"].force.set(force=np.zeros_like(CL_fc))

    # for link_name, W_f_i in zip(model.link_names(), W_f_ext):
    #     # W_H_LW = (
    #     #     fk(model)[model.get_link(link_name).index()].at[0:3, 0:3].set(jnp.eye(3))
    #     # )
    #     # W_Xv_LW = se3.SE3.from_matrix(W_H_LW).adjoint()
    #     # LW_f_i = W_Xv_LW.transpose() @ W_f_i
    #     # contacts[link_name].force.set(force=LW_f_i[0:3])
    #     W_H_L = fk(model)[model.get_link(link_name).index()]
    #     W_Xv_L = se3.SE3.from_matrix(W_H_L).adjoint()
    #     L_f_i = W_Xv_L.transpose() @ W_f_i
    #     contacts[link_name].force.set(force=L_f_i[0:3])

    rate.sleep()
    print(time.time() - now)

# # visualize_in_gazebo = True
# visualize_in_gazebo = False
#
# # You need install:
# #
# # - Gazebo Sim
# # - The "viz" extra requires of the Python package (pip install jaxsim[viz])
#
# if visualize_in_gazebo:
#     # gui.close()
#
#     from jaxsim.viz import ignition
#
#     # Get the visualizer
#     gui = ignition.IgnitionVisualizer(
#         dt=float(simulator.dt()), rtf=0.2, ground_plane=True
#     )
#
#     # Open the visualizer
#     gui.open()
#
#     # Insert the model
#     model_name_viz = gui.insert_model(
#         sdf=model_sdf_path,
#         model_pose=(
#             x0.base_position.tolist(),
#             x0.base_quaternion.tolist(),
#         ),
#     )
#
#     for idx, (_, pos, quat, q) in enumerate(
#         zip(
#             list(step_data[model.name()].tf),
#             x.base_position,
#             x.base_quaternion,
#             x.joint_positions,
#         )
#     ):
#         # if idx == 245:
#         #     break
#
#         if jnp.isnan(pos).any() or jnp.isnan(quat).any() or jnp.isnan(q).any():
#             break
#
#         gui.update_model(
#             name=model_name_viz,
#             base_position=pos.tolist(),
#             base_quaternion=quat.tolist(),
#             q=q.tolist(),
#             joint_names=model.joint_names(),
#         )
#
#     # time.sleep(5)
#     # gui.close()
