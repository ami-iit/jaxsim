import jaxlie
import numpy as np

from jaxsim import logging
from jaxsim.api.model import JaxSimModel

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


def random_model_state(
    model: JaxSimModel,
) -> dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dictionary containing random model state.

    Args:
        model: the model to which the random state refers to.

    Returns:
        The dictionary containing the random model state.
    """

    rng = get_rng()

    state = {
        "joint_positions": rng.uniform(size=model.dofs(), low=-1),
        "joint_velocities": rng.uniform(size=model.dofs(), low=-1),
        "base_position": rng.uniform(size=3, low=-1),
        "base_quaternion": jaxlie.SO3.from_rpy_radians(
            *rng.uniform(low=0, high=2 * np.pi, size=3)
        ).as_quaternion_xyzw()[np.array([3, 0, 1, 2])],
    }

    # If floating-base, generate random base velocities
    if model.floating_base():
        state | {
            "base_linear_velocity": rng.uniform(size=3, low=-1),
            "base_angular_velocity": rng.uniform(size=3, low=-1),
        }

    return state


def random_model_input(model: JaxSimModel) -> dict[np.ndarray, np.ndarray]:
    """
    Generate a dictionary containing random joint torques and external forces.

    Args:
        model: the model to which the random state refers to.

    Returns:
        A dictionary containing the random joint torques and external forces.
    """

    rng = get_rng()

    tau = 10 * rng.uniform(size=model.dofs(), low=-1)
    f_ext = 10 * rng.uniform(size=[model.number_of_links(), 6], low=-1)

    # Zero the base force if the robot is fixed base
    if not model.floating_base() and model.number_of_links() > 0:
        f_ext[0] = np.zeros(6)

    return {"tau": tau, "f_ext": f_ext}
