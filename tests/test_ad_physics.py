import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from pytest import param as p

from jaxsim.high_level.common import VelRepr
from jaxsim.high_level.model import Model

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

    # ========================
    # Extract state and inputs
    # ========================

    # Extract the physics model used in the low-level physics algorithms
    physics_model = model.physics_model

    # State
    s = model.joint_positions()
    ṡ = model.joint_velocities()
    xfb = model.data.model_state.xfb()

    # Inputs
    f_ext = model.external_forces()
    tau = model.joint_generalized_forces_targets()

    # Perturbation used for computing finite differences
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

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

    link_indices = [l.index() for l in model.links()]

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
