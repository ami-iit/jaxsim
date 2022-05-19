import pathlib

import gym_ignition.rbd.idyntree.kindyncomputations as kindyncomputations
import gym_ignition_models
import jax.numpy as jnp
import numpy as np
import pytest
import utils_testing

import jaxsim.physics.algos.aba as aba
from jaxsim import high_level
from jaxsim.high_level.model import Model, VelRepr


@pytest.mark.parametrize(
    "model_name, vel_repr",
    [
        ("pendulum", VelRepr.Inertial),
        # ("pendulum", VelRepr.Body),
        # ("pendulum", VelRepr.Mixed),
        ("panda", VelRepr.Inertial),
        # ("panda", VelRepr.Body),
        # ("panda", VelRepr.Mixed),
        ("iCubGazeboV2_5", VelRepr.Inertial),
        # ("iCubGazeboV2_5", VelRepr.Body),
        # ("iCubGazeboV2_5", VelRepr.Mixed),
    ],
)
def test_aba(model_name: str, vel_repr: VelRepr):

    gravity = np.array([0, 0, -10.0])

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )

    urdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.URDF_PATH
    )

    # Build the JAXsim model
    model = Model.build_from_sdf(sdf=sdf_file_path, vel_repr=vel_repr, gravity=gravity)

    # Random model data
    model = model.update_data(
        model_state=utils_testing.random_physics_model_state(model.physics_model),
        model_input=utils_testing.random_physics_model_input(model.physics_model),
    )
    tau = model.data.model_input.tau

    # ==========================
    # Ground truth with iDynTree
    # ==========================

    kin_dyn = utils_testing.get_kindyncomputations(
        model=model, urdf_path=pathlib.Path(urdf_file_path)
    )

    assert kin_dyn.get_joint_positions() == pytest.approx(model.joint_positions())
    assert kin_dyn.get_joint_velocities() == pytest.approx(model.joint_velocities())
    assert kin_dyn.get_world_base_transform() == pytest.approx(
        model.base_transform(), abs=0.001
    )
    assert kin_dyn.get_model_velocity() == pytest.approx(model.generalized_velocity())

    M = kin_dyn.get_mass_matrix()
    h = kin_dyn.get_bias_forces()
    f_ext = model.external_forces().flatten()
    S = np.block([np.zeros(shape=(kin_dyn.dofs, 6)), np.eye(kin_dyn.dofs)]).T

    J = np.vstack(
        [
            kin_dyn.get_frame_jacobian(frame_name=link_name)
            for link_name in model.physics_model.description.link_names()
        ]
    )

    if model.floating_base():

        nu_dot_idyntree = np.linalg.inv(M) @ (S @ tau - h + J.T @ f_ext)
        sdd_idyntree = nu_dot_idyntree[6:]
        a_WB_idyntree = nu_dot_idyntree[0:6]

    else:

        sdd_idyntree = np.linalg.inv(M[6:, 6:]) @ (tau - h[6:] + (J.T @ f_ext)[6:])
        a_WB_idyntree = jnp.zeros(6)

    # ========================
    # Ground truth with JAXsim
    # ========================

    a_WB_jaxsim, sdd_jaxsim = model.forward_dynamics_crb(tau=tau)

    assert sdd_jaxsim.squeeze() == pytest.approx(sdd_idyntree.squeeze(), abs=0.5)
    assert a_WB_jaxsim.squeeze() == pytest.approx(a_WB_idyntree.squeeze(), abs=0.2)

    # ========
    # Test ABA
    # ========

    a_WB, sdd = model.forward_dynamics_aba(tau=tau)

    assert sdd.squeeze() == pytest.approx(sdd_idyntree.squeeze(), abs=0.5)
    assert sdd.squeeze() == pytest.approx(sdd_jaxsim.squeeze(), abs=0.5)

    if model.floating_base():
        assert a_WB.squeeze() == pytest.approx(a_WB_idyntree.squeeze(), abs=0.200)
        assert a_WB.squeeze() == pytest.approx(a_WB_jaxsim.squeeze(), abs=0.200)
