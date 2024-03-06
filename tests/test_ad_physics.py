import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from pytest import param as p

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_models, utils_rng
from .utils_models import Robot


@pytest.mark.parametrize(
    "robot, vel_repr",
    [
        p(*[Robot.Ur10, VelRepr.Inertial], id="Ur10-Inertial"),
        p(*[Robot.AnymalC, VelRepr.Inertial], id="AnymalC-Inertial"),
        p(*[Robot.Cassie, VelRepr.Inertial], id="Cassie-Inertial"),
    ],
)
def test_ad_physics(robot: utils_models.Robot, vel_repr: VelRepr) -> None:
    """Unit test of the application of Automatic Differentiation on RBD algorithms."""

    robot = Robot.Ur10
    vel_repr = VelRepr.Inertial

    # Initialize the gravity
    gravity = np.array([0, 0, -10.0])

    # Get the URDF of the robot
    urdf_file_path = utils_models.ModelFactory.get_model_description(robot=robot)

    # Build the model
    model = js.model.JaxSimModel.build_from_model_description(
        model_description=urdf_file_path,
        is_urdf=True,
        gravity=gravity,
    )

    random_state = utils_rng.random_model_state(model=model)

    # Initialize the model with a random state
    data = js.data.JaxSimModelData.build(
        model=model, velocity_representation=vel_repr, **random_state
    )

    # Initialize the model with a random input
    tau, f_ext = utils_rng.random_model_input(model=model)

    # ========================
    # Extract state and inputs
    # ========================

    # State
    s = data.joint_positions(model=model)
    ṡ = data.joint_velocities(model=model)
    xfb = data.state.physics_model.xfb()

    # Perturbation used for computing finite differences
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    physics_model = model.physics_model

    # =====================================================
    # Check first-order and second-order derivatives of ABA
    # =====================================================

    import jaxsim.physics.algos.aba

    aba = lambda xfb, s, ṡ, tau, f_ext: jaxsim.physics.algos.aba.aba(
        model=physics_model, xfb=xfb, q=s, qd=ṡ, tau=tau, f_ext=f_ext
    )

    check_grads(
        f=aba,
        args=(xfb, s, ṡ, tau, f_ext),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # ======================================================
    # Check first-order and second-order derivatives of RNEA
    # ======================================================

    import jaxsim.physics.algos.rnea

    W_v̇_WB = utils_rng.get_rng().uniform(size=6, low=-1)
    s̈ = utils_rng.get_rng().uniform(size=physics_model.dofs(), low=-1)

    rnea = lambda xfb, s, ṡ, s̈, W_v̇_WB, f_ext: jaxsim.physics.algos.rnea.rnea(
        model=physics_model, xfb=xfb, q=s, qd=ṡ, qdd=s̈, a0fb=W_v̇_WB, f_ext=f_ext
    )

    check_grads(
        f=rnea,
        args=(xfb, s, ṡ, s̈, W_v̇_WB, f_ext),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # ======================================================
    # Check first-order and second-order derivatives of CRBA
    # ======================================================

    import jaxsim.physics.algos.crba

    crba = lambda s: jaxsim.physics.algos.crba.crba(model=physics_model, q=s)

    check_grads(
        f=crba,
        args=(s,),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # ====================================================
    # Check first-order and second-order derivatives of FK
    # ====================================================

    import jaxsim.physics.algos.forward_kinematics

    fk = (
        lambda xfb, s: jaxsim.physics.algos.forward_kinematics.forward_kinematics_model(
            model=physics_model, xfb=xfb, q=s
        )
    )

    check_grads(
        f=fk,
        args=(xfb, s),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # ==========================================================
    # Check first-order and second-order derivatives of Jacobian
    # ==========================================================

    import jaxsim.physics.algos.jacobian

    link_indices = [
        js.link.name_to_idx(model=model, link_name=link) for link in model.link_names()
    ]

    jacobian = lambda s: jaxsim.physics.algos.jacobian.jacobian(
        model=physics_model, q=s, body_index=link_indices[-1]
    )

    check_grads(
        f=jacobian,
        args=(s,),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # =====================================================================
    # Check first-order and second-order derivatives of soft contacts model
    # =====================================================================

    import jaxsim.physics.algos.soft_contacts

    p = utils_rng.get_rng().uniform(size=3, low=-1)
    v = utils_rng.get_rng().uniform(size=3, low=-1)
    m = utils_rng.get_rng().uniform(size=3, low=-1)

    parameters = jaxsim.physics.algos.soft_contacts.SoftContactsParams.build(
        K=10_000, D=20.0, mu=0.5
    )

    soft_contacts = lambda p, v, m: jaxsim.physics.algos.soft_contacts.SoftContacts(
        parameters=parameters
    ).contact_model(position=p, velocity=v, tangential_deformation=m)

    check_grads(
        f=soft_contacts,
        args=(p, v, m),
        order=2,
        modes=["rev", "fwd"],
        eps=ε,
    )
