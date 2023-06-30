import jax
import numpy as np
import pytest

from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model

from . import utils_models, utils_rng


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
    jit_enabled = True
    fn = jax.jit if jit_enabled else lambda x: x
    a_WB_aba, sdd_aba = fn(model.forward_dynamics_aba)(tau=tau)

    # ==============================================
    # Compute forward dynamics with dedicated method
    # ==============================================

    a_WB, sdd = model.forward_dynamics_crb(tau=tau)

    assert sdd.squeeze() == pytest.approx(sdd_aba.squeeze(), abs=0.5)
    assert a_WB.squeeze() == pytest.approx(a_WB_aba.squeeze(), abs=0.2)
