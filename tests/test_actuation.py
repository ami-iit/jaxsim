import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.rbda
from jaxsim import VelRepr


def test_tn_curve(jaxsim_model_single_pendulum: js.model.JaxSimModel):

    model = jaxsim_model_single_pendulum
    new_act_params = jaxsim.rbda.actuation.ActuationParams()
    with new_act_params.editable(validate=False) as new_act_params:
        new_act_params.torque_max = 10
        new_act_params.omega_th = 1
        new_act_params.omega_max = 2
    with model.editable(validate=False) as model:
        model.actuation_params = new_act_params
    data = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
    )
    new_joint_velocities = 1.5 * jnp.ones(model.dofs())
    joint_torques_0 = 30 * jnp.ones(model.dofs())
    data_0 = data.replace(model=model, joint_velocities=new_joint_velocities)

    τ_total = js.actuation_model.compute_resultant_torques(
        model, data_0, joint_force_references=joint_torques_0
    )
    assert (τ_total <= joint_torques_0).all()
    new_joint_velocities = 2.5 * jnp.ones(model.dofs())
    joint_torques_0 = 30 * jnp.ones(model.dofs())
    data_0 = data.replace(model=model, joint_velocities=new_joint_velocities)

    τ_total = js.actuation_model.compute_resultant_torques(
        model, data_0, joint_force_references=joint_torques_0
    )
    assert (τ_total == jnp.zeros(model.dofs())).all()
