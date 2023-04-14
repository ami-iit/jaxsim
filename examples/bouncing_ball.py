from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses
import rod
from rod.builder.primitives import SphereBuilder

import jaxsim.typing as jtp
from jaxsim import high_level
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData

# ================================
# Configure and run the simulation
# ================================

# Create the SDF model of a sphere
model_sdf_string = rod.Sdf(
    version="1.7",
    model=SphereBuilder(radius=0.10, mass=1.0, name="sphere")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build(),
).serialize(pretty=True)

# Simulation step parameters
step_size = 0.001
steps_per_run = 1

# Create the simulator
simulator = JaxSim.build(
    step_size=step_size,
    steps_per_run=steps_per_run,
    velocity_representation=high_level.model.VelRepr.Body,
    integrator_type=high_level.model.IntegratorType.EulerSemiImplicit,
    simulator_data=SimulatorData(
        contact_parameters=SoftContactsParams(K=1e6, D=2e3, mu=0.5),
    ),
).mutable(validate=False)

# Insert the model in the simulator and extract a mutable object
model = simulator.insert_model_from_sdf(sdf=model_sdf_string).mutable(validate=True)

# Zero the model data
model.zero()

# Reset base quantities
model.reset_base_position(position=jnp.array([0.0, 0, 0.50]))
model.reset_base_velocity(base_velocity=jnp.array([0.0, 0, 0, 0, 0, 0]))

# Reset joint positions
# s = model.joint_random_positions()
# model.reset_joint_positions(positions=s)


# Create a logger class used as callback to extract data from an
# open-loop simulation over a given horizon
@jax_dataclasses.pytree_dataclass
class SimulatorLogger(simulator_callbacks.PostStepCallback):
    def post_step(
        self, sim: JaxSim, step_data: Dict[str, StepData]
    ) -> Tuple[JaxSim, jtp.PyTree]:
        """Return the StepData object of each simulated model"""
        return sim, step_data


# Store the initial state (useful for visualization purpose)
x0 = model.data.model_state

# Simulate 3.0 seconds
simulator, (cb, step_data) = simulator.step_over_horizon(
    horizon_steps=int(3.0 / simulator.dt()),
    callback_handler=SimulatorLogger(),
    clear_inputs=True,
)

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

# ================
# 3D visualization
# ================


def local_contact_6D_forces(model: high_level.model.Model, W_f_cp):
    """Helper to convert contact forces from jaxsim to meshcat_viz."""

    # Compute with FK the pose of all model links
    W_HH_L = model.forward_kinematics()

    # Compute the transform between all collidable points and their parent link
    L_HH_CL = jax.vmap(lambda L_p_cp: jnp.eye(4).at[0:3, 3].set(L_p_cp.squeeze()))(
        model.physics_model.gc.point.T
    )

    # Combine the transforms to compute the world pose of all collidable points
    W_HH_CL = jax.vmap(lambda L_H_CL, idx: W_HH_L[idx] @ L_H_CL)(
        L_HH_CL, model.physics_model.gc.body
    )

    # Compute the X transformations for 6D velocities from H
    W_XXv_CL = jax.vmap(lambda W_H_CL: se3.SE3.from_matrix(W_H_CL).adjoint())(W_HH_CL)

    # Express the contact forces in the frame CL += (W_p_C, [L]), where W_p_C is the
    # position of the collidable point, and [L] is the orientation of its parent link
    return jax.vmap(lambda W_Xv_CL, W_f: W_Xv_CL.T @ W_f)(W_XXv_CL, W_f_cp)


@jax.jit
def world_to_viz_contact_forces(model, W_p_B, W_Q_B, s, W_f_cp):
    """Compiled version for jaxsim to meshcat_viz conversion of contact forces."""

    m = model.copy()

    m.zero()
    m.reset_joint_positions(positions=s)
    m.reset_base_position(position=W_p_B)
    m.reset_base_orientation(orientation=W_Q_B)

    return local_contact_6D_forces(m, W_f_cp)


import numpy as np
from loop_rate_limiters import RateLimiter
from meshcat_viz import MeshcatWorld
from meshcat_viz.contacts import ModelContacts

from jaxsim.sixd import se3

# Open the visualizer
world = MeshcatWorld()
world.open()

# Insert the model from a URDF/SDF resource
model_name = world.insert_model(model_description=model_sdf_string, is_urdf=False)
# world.remove_model(model_name=model_name)

# Create the interaction forces object
contacts = ModelContacts(meshcat_model=world._meshcat_models[model_name])

# Extract information of collidable points
index_to_link_name = {link.index(): link.name() for link in model.links()}
parent_link_name = [index_to_link_name[idx] for idx in model.physics_model.gc.body]

# Add all jaxsim contact forces to the visualizer
for idx, (L_p_cp, name) in enumerate(
    zip(model.physics_model.gc.point.T, parent_link_name)
):
    # Create the transform between the link frame L and the collidable point frame C[L]
    L_H_CL = np.block([[np.eye(3), np.vstack(L_p_cp)], [0, 0, 0, 1]])

    # Enable visualization of forces for all contact frames corresponding to
    # collidable points
    contacts.add_contact_frame(
        contact_name=f"{name}_{idx}",
        transform=L_H_CL,
        relative_to=world._meshcat_models[model_name].link_to_node[name],
        enable_force=True,
        force_scale=0.025 / 1,  # 2.5cm / 1 N
        force_radius=0.001,
    )

# Initialize the model state
world.update_model(
    model_name=model_name,
    joint_names=model.joint_names(),
    joint_positions=np.array(x0.joint_positions),
    base_position=np.array(x0.base_position),
    base_quaternion=np.array(x0.base_quaternion),
)

# Function to compute FK
fk = jax.jit(lambda model: model.forward_kinematics())

rtf = 1.0
down_sampling = 25
rate = RateLimiter(frequency=float(rtf / (simulator.dt() * down_sampling)))

# Visualization loop
for s, W_p_B, W_Q_B, W_f_ext, W_fc_ext in list(
    zip(
        x.joint_positions,
        x.base_position,
        x.base_quaternion,
        step_data[model.name()].aux["t0"]["contact_forces_links"],
        step_data[model.name()].aux["t0"]["contact_forces_points"],
    )
)[::down_sampling]:
    # Update the visualized model state
    world.update_model(
        model_name=model_name,
        joint_names=model.joint_names(),
        joint_positions=s,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
    )

    # Compute the forces expressed in the right frame for visualization
    CL_fc_ext = world_to_viz_contact_forces(model, W_p_B, W_Q_B, s, W_fc_ext)[:, 0:3]

    # Update the visualized contact forces
    for idx, (CL_fc, name) in enumerate(zip(CL_fc_ext, parent_link_name)):
        contacts[f"{name}_{idx}"].force.set(force=np.array(CL_fc))
