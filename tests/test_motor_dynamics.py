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

tau = jnp.array([0.1] * model.dofs())

# Perform FD with CRB without motor
C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)

print("C_v̇_WB nomotor: ", C_v̇_WB)
print("a_WB nomotor: ", crb)

# Set motor parameters
model.set_motor_gear_ratios(jnp.array([1 / 100.0] * model.dofs()))
model.set_motor_inertias(jnp.array([0.000_05] * model.dofs()))
model.set_motor_viscous_frictions(jnp.array([0.0003] * model.dofs()))

# Perform FD with CRB with motor
C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)

print("C_v̇_WB: ", C_v̇_WB)
print("CRB: ", crb)

# Perform FD with ABA with motor
a_WB, aba = model.forward_dynamics_aba(tau=tau)
print("a_WB: ", a_WB)
print("ABA: ", aba)

# Set motor inertia to zero and perform FD with CRB and ABA
model.set_motor_inertias(jnp.array([0.0] * model.dofs()))
print("Motor inertia set to zero:")
C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)

print("C_v̇_WB: ", C_v̇_WB)
print("CRB: ", crb)

a_WB, aba = model.forward_dynamics_aba(tau=tau)
print("a_WB: ", a_WB)
print("ABA: ", aba)

try:
    np.testing.assert_allclose(crb, aba)
    np.testing.assert_allclose(C_v̇_WB, a_WB)
except Exception as e:
    print(e)
