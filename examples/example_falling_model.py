import pathlib
import sys
import time

import jax.numpy as jnp

from jaxsim import high_level
from jaxsim.math.quaternion import Quaternion
from jaxsim.math.rotation import Rotation
from jaxsim.simulation import ode_data, ode_integration

# ================================
# Configure and run the simulation
# ================================

# Find model description
base_path = __file__ if not hasattr(sys, "ps1") is not None else pathlib.Path().cwd()
model_sdf_path = pathlib.Path(base_path) / "resources" / "cube.sdf"
assert model_sdf_path.exists()

# Create the model
model = high_level.model.Model.build_from_sdf(sdf=model_sdf_path)

# Define the initial state of the model
x0 = ode_data.ODEState.zero(physics_model=model.physics_model)
physics_model_state = x0.physics_model.replace(
    base_position=jnp.array([0.0, 0, 5.0]),
    base_quaternion=Quaternion.from_dcm(
        Rotation.x(jnp.pi / 4) @ Rotation.y(3.14 / 3)
    ).squeeze(),
)
x0 = x0.replace(physics_model=physics_model_state)

# Define the simulated time
dt = 0.001
t = jnp.arange(start=0, stop=5.0, step=dt)

# Integrate the system in open loop with Forward Euler
x, _ = ode_integration.ode_integration_euler(
    x0=x0,
    t=t,
    physics_model=model.physics_model,
    num_sub_steps=1,
    return_aux=True,
)

# ========================
# Plot the simulation data
# ========================

import matplotlib.pyplot as plt

plt.plot(t, x.physics_model.base_position, label=["x", "y", "z"])
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Trajectory of the model's base")
plt.show()
