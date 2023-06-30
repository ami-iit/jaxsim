import pathlib

import jax
import numpy as np
import pytest

from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model

from . import utils_idyntree, utils_models, utils_rng


@pytest.mark.parametrize(
    "robot, vel_repr",
    [
        (utils_models.Robot.DoublePendulum, VelRepr.Inertial),
        (utils_models.Robot.DoublePendulum, VelRepr.Body),
        (utils_models.Robot.DoublePendulum, VelRepr.Mixed),
        (utils_models.Robot.Ur10, VelRepr.Inertial),
        (utils_models.Robot.Ur10, VelRepr.Body),
        (utils_models.Robot.Ur10, VelRepr.Mixed),
        (utils_models.Robot.AnymalC, VelRepr.Inertial),
        (utils_models.Robot.AnymalC, VelRepr.Body),
        (utils_models.Robot.AnymalC, VelRepr.Mixed),
        (utils_models.Robot.Cassie, VelRepr.Inertial),
        (utils_models.Robot.Cassie, VelRepr.Body),
        (utils_models.Robot.Cassie, VelRepr.Mixed),
        # (utils_models.Robot.iCub, VelRepr.Inertial),
        # (utils_models.Robot.iCub, VelRepr.Body),
        # (utils_models.Robot.iCub, VelRepr.Mixed),
    ],
)
def test_eom(robot: utils_models.Robot, vel_repr: VelRepr) -> None:
    """Unit test of all the terms of the floating-base Equations of Motion."""

    # Initialize the gravity
    gravity = np.array([0, 0, -10.0])

    # Get the URDF of the robot
    urdf_file_path = utils_models.ModelFactory.get_model_description(robot=robot)

    # Build the high-level model
    model_jaxsim = Model.build_from_model_description(
        model_description=urdf_file_path,
        vel_repr=vel_repr,
        gravity=gravity,
        is_urdf=True,
    ).mutable(mutable=True, validate=True)

    # Initialize the model with a random state
    model_jaxsim.data.model_state = utils_rng.random_physics_model_state(
        physics_model=model_jaxsim.physics_model
    )

    # Initialize the model with a random input
    model_jaxsim.data.model_input = utils_rng.random_physics_model_input(
        physics_model=model_jaxsim.physics_model
    )

    # Get the joint torques
    tau = model_jaxsim.joint_generalized_forces_targets()

    # ==========================
    # Ground truth with iDynTree
    # ==========================

    kin_dyn = utils_idyntree.KinDynComputations.build(
        urdf=pathlib.Path(urdf_file_path),
        considered_joints=model_jaxsim.joint_names(),
        vel_repr=vel_repr,
        gravity=gravity,
    )

    kin_dyn.set_robot_state(
        joint_positions=np.array(model_jaxsim.joint_positions()),
        joint_velocities=np.array(model_jaxsim.joint_velocities()),
        base_transform=np.array(model_jaxsim.base_transform()),
        base_velocity=np.array(model_jaxsim.base_velocity()),
    )

    assert kin_dyn.joint_names() == model_jaxsim.joint_names()
    assert kin_dyn.gravity == pytest.approx(model_jaxsim.physics_model.gravity[0:3])
    assert kin_dyn.joint_positions() == pytest.approx(model_jaxsim.joint_positions())
    assert kin_dyn.joint_velocities() == pytest.approx(model_jaxsim.joint_velocities())
    assert kin_dyn.base_velocity() == pytest.approx(model_jaxsim.base_velocity())
    assert kin_dyn.frame_transform(model_jaxsim.base_frame()) == pytest.approx(
        model_jaxsim.base_transform()
    )

    M_idt = kin_dyn.mass_matrix()
    h_idt = kin_dyn.bias_forces()
    g_idt = kin_dyn.gravity_forces()

    J_idt = np.vstack(
        [
            kin_dyn.jacobian_frame(frame_name=link_name)
            for link_name in model_jaxsim.link_names()
        ]
    )

    # ================================
    # Test individual terms of the EoM
    # ================================

    jit_enabled = True
    fn = jax.jit if jit_enabled else lambda x: x

    M_jaxsim = fn(model_jaxsim.free_floating_mass_matrix)()
    g_jaxsim = fn(model_jaxsim.free_floating_gravity_forces)()
    h_jaxsim = fn(model_jaxsim.free_floating_bias_forces)()
    J_jaxsim = np.vstack([link.jacobian() for link in model_jaxsim.links()])

    sl = np.s_[0:] if model_jaxsim.floating_base() else np.s_[6:]
    assert M_jaxsim[sl, sl] == pytest.approx(M_idt[sl, sl], abs=1e-3)
    assert g_jaxsim[sl] == pytest.approx(g_idt[sl], abs=1e-3)
    assert h_jaxsim[sl] == pytest.approx(h_idt[sl], abs=1e-3)
    assert J_jaxsim == pytest.approx(J_idt, abs=1e-3)

    # ===========================================
    # Test the forward dynamics computed with CRB
    # ===========================================

    J_ff = fn(model_jaxsim.generalized_free_floating_jacobian)()
    f_ext = fn(model_jaxsim.external_forces)().flatten()
    nud = np.hstack(fn(model_jaxsim.forward_dynamics_crb)(tau=tau))
    S = np.block(
        [np.zeros(shape=(model_jaxsim.dofs(), 6)), np.eye(model_jaxsim.dofs())]
    ).T

    assert h_jaxsim[sl] == pytest.approx(
        (S @ tau + J_ff.T @ f_ext - M_jaxsim @ nud)[sl], abs=1e-3
    )
