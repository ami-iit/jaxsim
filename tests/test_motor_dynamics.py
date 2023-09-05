import pathlib

import jax.numpy as jnp
import numpy as np
import jax
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
model.reduce(
    considered_joints=[
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "torso_roll",
        "torso_yaw",
    ]
)

print("Model links: ", model.link_names())
tau = jnp.array([0.1] * model.dofs())

with jax.disable_jit():
    print("=====================================================")
    print("============ Test without motor dynamics ============")
    print("=====================================================")

    C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)

    print("C_v̇_WB: ", C_v̇_WB)
    print("CRB: ", crb)

    a_WB, aba = model.forward_dynamics_aba(tau=tau)
    print("a_WB: ", a_WB)
    print("ABA: ", aba)

    try:
        np.testing.assert_allclose(crb, aba, rtol=0.5)
        np.testing.assert_allclose(C_v̇_WB, a_WB, rtol=0.5)
    except Exception as e:
        print(e)

    print("=====================================================")
    print("============ Test with motor dynamics ===============")
    print("=====================================================")

    # Set motor parameters
    model.set_motor_gear_ratios(jnp.array([1 / 100.0] * model.dofs()))
    model.set_motor_inertias(jnp.array([5e-4] * model.dofs()))
    model.set_motor_viscous_frictions(jnp.array([0.0003] * model.dofs()))

    C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)
    print("C_v̇_WB: ", C_v̇_WB)
    print("CRB: ", crb)

    a_WB, aba = model.forward_dynamics_aba(tau=tau)
    print("a_WB: ", a_WB)
    print("ABA: ", aba)

    try:
        np.testing.assert_allclose(crb, aba, rtol=0.5)
        np.testing.assert_allclose(C_v̇_WB, a_WB, rtol=0.5)
    except Exception as e:
        print(e)

    print("=====================================================")
    print("================ Inertia set to zero ================")
    print("=====================================================")

    # Set motor inertia to zero and perform FD with CRB and ABA
    model.set_motor_inertias(jnp.array([0.0] * model.dofs()))

    C_v̇_WB, crb = model.forward_dynamics_crb(tau=tau)
    print("C_v̇_WB: ", C_v̇_WB)
    print("CRB: ", crb)

    a_WB, aba = model.forward_dynamics_aba(tau=tau)
    print("a_WB: ", a_WB)
    print("ABA: ", aba)

    try:
        np.testing.assert_allclose(crb, aba, rtol=0.5)
        np.testing.assert_allclose(C_v̇_WB, a_WB, rtol=0.5)
    except Exception as e:
        print(e)
