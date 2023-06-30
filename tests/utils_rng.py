import numpy as np

from jaxsim import logging, sixd
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.physics.model.physics_model_state import (
    PhysicsModelInput,
    PhysicsModelState,
)
from jaxsim.utils import Mutability

# Initialize a global RNG used by all tests
test_rng = None


def get_rng(seed: int = None) -> np.random.Generator:
    """
    Get a random number generator that can be used to produce reproducibile sequences.

    Args:
        seed: The optional seed of the RNG (ignored if the RNG

    Returns:
        A random number generator.
    """

    global test_rng

    if test_rng is not None and seed is not None:
        msg = "Seed was already configured globally, ignoring the new one"
        logging.warning(msg=msg)

    seed = seed if seed is not None else 42
    test_rng = test_rng if test_rng is not None else np.random.default_rng(seed=seed)

    return test_rng


def random_physics_model_state(physics_model: PhysicsModel) -> PhysicsModelState:
    """
    Generate a random `PhysicsModelState` object.

    Args:
        physics_model: the physics model to which the random state refers to.

    Returns:
        The random `PhysicsModelState` object.
    """

    rng = get_rng()

    with PhysicsModelState.zero(physics_model=physics_model).mutable_context(
        mutability=Mutability.MUTABLE
    ) as state:
        # Generate random joint quantities
        state.joint_positions = rng.uniform(size=physics_model.dofs(), low=-1)
        state.joint_velocities = rng.uniform(size=physics_model.dofs(), low=-1)

        # Generate random base quantities
        state.base_position = rng.uniform(size=3, low=-1)
        state.base_quaternion = sixd.so3.SO3.from_rpy_radians(
            *rng.uniform(low=0, high=2 * np.pi, size=3)
        ).as_quaternion_xyzw()[np.array([3, 0, 1, 2])]

        # If floating-base, generate random base velocities
        if physics_model.is_floating_base:
            state.base_linear_velocity = rng.uniform(size=3, low=-1)
            state.base_angular_velocity = rng.uniform(size=3, low=-1)

    return state


def random_physics_model_input(physics_model: PhysicsModel) -> PhysicsModelInput:
    """
    Generate a random `PhysicsModelInput` object.

    Args:
        physics_model: the physics model to which the random state refers to.

    Returns:
        The random `PhysicsModelInput` object.
    """

    rng = get_rng()

    with PhysicsModelInput.zero(physics_model=physics_model).mutable_context(
        mutability=Mutability.MUTABLE
    ) as model_input:
        # Generate random joint torques and external forces
        model_input.tau = 10 * rng.uniform(size=physics_model.dofs(), low=-1)
        model_input.f_ext = 10 * rng.uniform(size=[physics_model.NB, 6], low=-1)

        # Zero the base force if the robot is fixed base
        if not physics_model.is_floating_base and physics_model.NB > 0:
            model_input.f_ext[0] = np.zeros(6)

    return model_input
