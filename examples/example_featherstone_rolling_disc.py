import pathlib
import sys
import time

import jax.numpy as jnp
import jax_dataclasses

from jaxsim import high_level
from jaxsim.math.quaternion import Quaternion
from jaxsim.math.rotation import Rotation
from jaxsim.physics.model.ground_contact import GroundContact
from jaxsim.simulation import ode_data, ode_integration


# We do not yet support the cylinder primitive for collision detection
def add_disc_contact_points(
    model: high_level.model.Model, num_points_circle: int = 16
) -> "high_level.model.Model":

    import numpy as np

    ang = np.arange(num_points_circle) * (2 * np.pi / num_points_circle)

    R = 0.05
    T = 0.01

    Y = np.ones(num_points_circle) * T / 2
    X = np.sin(ang) * R
    Z = np.cos(ang) * R

    gc = GroundContact(
        body=np.zeros(2 * num_points_circle, dtype=int),
        point=jnp.vstack(
            [
                np.hstack([X, X]),
                np.hstack([-Y, Y]),
                np.hstack([Z, Z]),
            ]
        ),
    )

    physics_model = jax_dataclasses.replace(model.physics_model, gc=gc)
    return jax_dataclasses.replace(model, physics_model=physics_model)


# ================================
# Configure and run the simulation
# ================================

# Find model description
base_path = __file__ if not hasattr(sys, "ps1") is not None else pathlib.Path().cwd()
model_sdf_path = pathlib.Path(base_path) / "resources" / "disc.sdf"
assert model_sdf_path.exists()

# Create the model
model = high_level.model.Model.build_from_sdf(sdf=model_sdf_path)
model = add_disc_contact_points(model=model, num_points_circle=32)

# Define the initial state of the model
x0 = ode_data.ODEState.zero(physics_model=model.physics_model)
physics_model_state = x0.physics_model.replace(
    base_position=jnp.array([0, 0, 0.1]),
    base_quaternion=Quaternion.from_dcm(Rotation.z(0.45)).squeeze(),
    base_linear_velocity=jnp.array([1.0, 0, 0]),
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
    num_sub_steps=5,
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
