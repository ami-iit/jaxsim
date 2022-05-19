import pathlib

import gym_ignition.rbd.idyntree.kindyncomputations as kindyncomputations
import gym_ignition_models
import jax.numpy as jnp
import numpy as np
import pytest
import utils_testing

import jaxsim.physics.algos.rnea as rnea
from jaxsim.high_level.model import Model, VelRepr
from jaxsim.parsers.sdf import build_model_from_sdf
from jaxsim.parsers.sdf.utils import flip_velocity_serialization
from jaxsim.physics.model.physics_model import PhysicsModel


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
def test_rnea(model_name: str, vel_repr: VelRepr):

    gravity = np.array([0, 0, -10.0])

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )

    urdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.URDF_PATH
    )

    # Build the JAXsim model
    model = Model.build_from_sdf(sdf=sdf_file_path, vel_repr=vel_repr, gravity=gravity)

    # Model acceleration
    base_net_acceleration = utils_testing.get_rng().uniform(size=6, low=-1)
    base_net_acceleration = jnp.zeros_like(base_net_acceleration)
    joint_accelerations = utils_testing.get_rng().uniform(size=model.dofs(), low=-1)

    if model.floating_base():
        qdd = np.hstack([base_net_acceleration, joint_accelerations])
    else:
        qdd = np.hstack([jnp.zeros(6), joint_accelerations])

    # Random model data
    model = model.update_data(
        model_state=utils_testing.random_physics_model_state(model.physics_model),
        model_input=utils_testing.random_physics_model_input(model.physics_model),
    )

    # ==========================
    # Ground truth with iDynTree
    # ==========================

    kin_dyn = utils_testing.get_kindyncomputations(
        model=model, urdf_path=pathlib.Path(urdf_file_path)
    )

    # Ground truth with iDynTree
    M = kin_dyn.get_mass_matrix()
    h = kin_dyn.get_bias_forces()
    g = kin_dyn.get_generalized_gravity_forces()
    J = np.vstack(
        [
            kin_dyn.get_frame_jacobian(frame_name=link_name)
            for link_name in model.physics_model.description.link_names()
        ]
    )
    f_ext = model.external_forces().flatten()
    Sτ = M @ qdd + h - J.T @ f_ext

    # ============================
    # Test gravity and bias forces
    # ============================

    g_jaxsim = model.free_floating_gravity_forces()
    h_jaxsim = model.free_floating_generalized_forces()

    if model.floating_base():
        assert g_jaxsim == pytest.approx(g, abs=0.100)
        assert h_jaxsim == pytest.approx(h, abs=0.100)
    else:
        assert g_jaxsim[6:] == pytest.approx(g[6:], abs=0.100)
        assert h_jaxsim[6:] == pytest.approx(h[6:], abs=0.100)

    # ==============================
    # Test complete inverse dynamics
    # ==============================

    f0_jaxsim, tau_jaxsim = model.inverse_dynamics(
        joint_accelerations=joint_accelerations, a0=base_net_acceleration
    )

    if model.floating_base():
        assert jnp.hstack([f0_jaxsim, tau_jaxsim]) == pytest.approx(Sτ, abs=0.100)
    else:
        assert tau_jaxsim == pytest.approx(Sτ[6:], abs=0.100)
