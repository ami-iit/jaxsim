import pathlib

import jax.numpy as jnp
import numpy as np

from jaxsim import high_level
from jaxsim.high_level.model import Model, ModelData
from jaxsim.physics.algos.aba import aba
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation.simulator import JaxSim, SimulatorData

sdf_path = (
    pathlib.Path.home()
    / "git"
    / "element_rl-for-codesign"
    / "assets"
    / "model"
    / "Stickbot.urdf"
)

simulator = JaxSim.build(
    step_size=1e-3,
    steps_per_run=1,
    velocity_representation=high_level.model.VelRepr.Body,
    integrator_type=high_level.model.IntegratorType.EulerSemiImplicit,
    simulator_data=SimulatorData(
        contact_parameters=SoftContactsParams(K=1e6, D=1.5e4, mu=0.5),
    ),
).mutable(validate=False)

# Insert model into the simulator
model = simulator.insert_model_from_description(model_description=sdf_path).mutable(
    validate=True
)

# ! Test motor dynamics implementation

# Set motor inertias
model.set_motor_inertias(inertias=np.array([0.5] * 34))
print(model.physics_model._joint_motor_inertia)

# Set motor viscous friction
model.set_motor_viscous_friction(viscous_frictions=np.array([0.5] * 34))
print(model.physics_model._joint_motor_viscous_friction)
# Set motor transmission ratios
model.set_motor_gear_ratios(gear_ratios=np.array([0.5] * 34))
print(model.physics_model._joint_motor_gear_ratio)

# ! Test ABA with motor dynamics
# q = jnp.array([0.0] * 34)
# qd = jnp.array([0.0] * 34)
# tau = jnp.array([0.0] * 34)

# x_fb = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

# W_a_WB, qdd = aba(model.physics_model, x_fb, q, qd, tau)

# C_v̇_WB, sdd = model.forward_dynamics_aba(tau=None)

# print(C_v̇_WB)
# print(sdd)
