import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from pytest import param as p

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree, utils_models, utils_rng
from .utils_models import Robot


@pytest.mark.parametrize(
    "robot, vel_repr",
    [
        p(*[Robot.DoublePendulum, VelRepr.Inertial], id="DoublePendulum-Inertial"),
        p(*[Robot.DoublePendulum, VelRepr.Body], id="DoublePendulum-Body"),
        p(*[Robot.DoublePendulum, VelRepr.Mixed], id="DoublePendulum-Mixed"),
        p(*[Robot.Ur10, VelRepr.Inertial], id="Ur10-Inertial"),
        p(*[Robot.Ur10, VelRepr.Body], id="Ur10-Body"),
        p(*[Robot.Ur10, VelRepr.Mixed], id="Ur10-Mixed"),
        p(*[Robot.AnymalC, VelRepr.Inertial], id="AnymalC-Inertial"),
        p(*[Robot.AnymalC, VelRepr.Body], id="AnymalC-Body"),
        p(*[Robot.AnymalC, VelRepr.Mixed], id="AnymalC-Mixed"),
        p(*[Robot.Cassie, VelRepr.Inertial], id="Cassie-Inertial"),
        p(*[Robot.Cassie, VelRepr.Body], id="Cassie-Body"),
        p(*[Robot.Cassie, VelRepr.Mixed], id="Cassie-Mixed"),
    ],
)
def test_eom(robot: utils_models.Robot, vel_repr: VelRepr) -> None:
    """Unit test of all the terms of the floating-base Equations of Motion."""

    # Initialize the gravity
    gravity = np.array([0, 0, -10.0])

    # Get the URDF of the robot
    urdf_file_path = utils_models.ModelFactory.get_model_description(robot=robot)

    # Build the model
    model_jaxsim = js.model.JaxSimModel.build_from_model_description(
        model_description=urdf_file_path,
        is_urdf=True,
        gravity=gravity,
    )

    random_state = utils_rng.random_model_state(model=model_jaxsim)

    # Initialize the model with a random state
    data = js.data.JaxSimModelData.build(
        model=model_jaxsim, velocity_representation=vel_repr, **random_state
    )

    # Initialize the model with a random input
    tau, f_ext = utils_rng.random_model_input(model=model_jaxsim)

    link_indices = [
        js.link.name_to_idx(model=model_jaxsim, link_name=link)
        for link in model_jaxsim.link_names()
    ]

    # ==========================
    # Ground truth with iDynTree
    # ==========================

    kin_dyn = utils_idyntree.KinDynComputations.build(
        urdf=pathlib.Path(urdf_file_path),
        considered_joints=list(model_jaxsim.joint_names()),
        vel_repr=vel_repr,
        gravity=gravity,
    )

    kin_dyn.set_robot_state(
        joint_positions=np.array(data.joint_positions()),
        joint_velocities=np.array(data.joint_velocities()),
        base_transform=np.array(data.base_transform()),
        base_velocity=np.array(data.base_velocity()),
    )

    assert kin_dyn.joint_names() == list(model_jaxsim.joint_names())
    assert kin_dyn.gravity == pytest.approx(data.gravity[0:3])
    assert kin_dyn.joint_positions() == pytest.approx(data.joint_positions())
    assert kin_dyn.joint_velocities() == pytest.approx(data.joint_velocities())
    assert kin_dyn.base_velocity() == pytest.approx(data.base_velocity())
    assert kin_dyn.frame_transform(model_jaxsim.base_link()) == pytest.approx(
        data.base_transform()
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

    M_jaxsim = js.model.free_floating_mass_matrix(model=model_jaxsim, data=data)
    g_jaxsim = js.model.free_floating_gravity_forces(model=model_jaxsim, data=data)
    J_jaxsim = jnp.vstack(
        [
            js.link.jacobian(model=model_jaxsim, data=data, link_index=idx)
            for idx in link_indices
        ]
    )
    h_jaxsim = js.model.free_floating_bias_forces(model=model_jaxsim, data=data)

    # Support both fixed-base and floating-base models by slicing the first six rows
    sl = np.s_[0:] if model_jaxsim.floating_base() else np.s_[6:]

    assert M_jaxsim[sl, sl] == pytest.approx(M_idt[sl, sl], abs=1e-3)
    assert g_jaxsim[sl] == pytest.approx(g_idt[sl], abs=1e-3)
    assert h_jaxsim[sl] == pytest.approx(h_idt[sl], abs=1e-3)
    assert J_jaxsim == pytest.approx(J_idt, abs=1e-3)

    # ===========================================
    # Test the forward dynamics computed with CRB
    # ===========================================

    J_ff = js.model.generalized_free_floating_jacobian(model=model_jaxsim, data=data)
    ν̇ = np.hstack(
        js.model.forward_dynamics_crb(model=model_jaxsim, data=data, joint_forces=tau)
    )
    S = np.block(
        [np.zeros(shape=(model_jaxsim.dofs(), 6)), np.eye(model_jaxsim.dofs())]
    ).T

    assert h_jaxsim[sl] == pytest.approx(
        (S @ tau + J_ff.T @ f_ext - M_jaxsim @ ν̇)[sl], abs=1e-3
    )
