import numpy as np
import pytest
from pytest import param as p

from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model

from . import utils_models, utils_rng
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
def test_aba(robot: utils_models.Robot, vel_repr: VelRepr) -> None:
    """
    Unit test of the ABA algorithm against forward dynamics computed from the EoM.
    """

    # Initialize the gravity
    gravity = np.array([0, 0, -10.0])

    # Get the URDF of the robot
    urdf_file_path = utils_models.ModelFactory.get_model_description(robot=robot)

    # Build the high-level model
    model = Model.build_from_model_description(
        model_description=urdf_file_path,
        vel_repr=vel_repr,
        gravity=gravity,
        is_urdf=True,
    ).mutable(mutable=True, validate=True)

    # Initialize the model with a random state
    model.data.model_state = utils_rng.random_physics_model_state(
        physics_model=model.physics_model
    )

    # Initialize the model with a random input
    model.data.model_input = utils_rng.random_physics_model_input(
        physics_model=model.physics_model
    )

    # Get the joint torques
    tau = model.joint_generalized_forces_targets()

    # Compute model acceleration with ABA
    v̇_WB_aba, s̈_aba = model.forward_dynamics_aba(tau=tau)

    # ==============================================
    # Compute forward dynamics with dedicated method
    # ==============================================

    v̇_WB, s̈ = model.forward_dynamics_crb(tau=tau)

    assert s̈.squeeze() == pytest.approx(s̈_aba.squeeze(), abs=0.5)
    assert v̇_WB.squeeze() == pytest.approx(v̇_WB_aba.squeeze(), abs=0.2)
