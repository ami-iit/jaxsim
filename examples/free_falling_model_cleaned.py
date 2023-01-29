import pathlib
from typing import Dict, Tuple

import gym_ignition_models
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim import high_level
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData

# ================================
# Configure and run the simulation
# ================================


model_sdf_path = pathlib.Path(
    gym_ignition_models.get_model_resource(
        robot_name="iCubGazeboSimpleCollisionsV2_5",
        resource_type=gym_ignition_models.ResourceType.SDF_PATH,
    )
)

assert model_sdf_path.exists()

# Simulation step parameters
step_size = 0.001
steps_per_run = 2

# Create the simulator
simulator = JaxSim.build(
    step_size=step_size,
    steps_per_run=steps_per_run,
    velocity_representation=high_level.model.VelRepr.Body,
    integrator_type=high_level.model.IntegratorType.EulerSemiImplicit,
    simulator_data=SimulatorData(
        contact_parameters=SoftContactsParams(K=1e6, D=2e3, mu=0.5)
    ),
).mutable(validate=False)

# Insert the model and get a mutable object
model = simulator.insert_model_from_sdf(sdf=model_sdf_path).mutable(validate=True)

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
model.reset_base_position(position=jnp.array([0.0, 0, 3.0]))
model.reset_base_velocity(base_velocity=jnp.array([0.6, 0, 0, 0, 0, 0]))

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

# Simulate 2.5 seconds
simulator, (cb, step_data) = simulator.step_over_horizon(
    horizon_steps=int(2.5 / simulator.dt()),
    callback_handler=cb,
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
