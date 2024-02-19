import jax

import jaxsim.typing as jtp

from . import data as Data
from . import model as Model


@jax.jit
def collidable_point_kinematics(
    model: Model.JaxSimModel, data: Data.JaxSimModelData
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """
    Compute the position and 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position and velocity of the collidable points in the world frame.

    Note:
        The collidable point velocity is the plain coordinate derivative of the position.
        If we attach a frame C = (p_C, [C]) to the collidable point, it corresponds to
        the linear component of the mixed 6D frame velocity.
    """

    from jaxsim.physics.algos.soft_contacts import collidable_points_pos_vel

    W_p_Ci, W_ṗ_Ci = collidable_points_pos_vel(
        model=model.physics_model,
        q=data.state.physics_model.joint_positions,
        qd=data.state.physics_model.joint_velocities,
        xfb=data.state.physics_model.xfb(),
    )

    return W_p_Ci.T, W_ṗ_Ci.T


@jax.jit
def collidable_point_positions(
    model: Model.JaxSimModel, data: Data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the position of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the collidable points in the world frame.
    """

    return collidable_point_kinematics(model=model, data=data)[0]


@jax.jit
def collidable_point_velocities(
    model: Model.JaxSimModel, data: Data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The 3D velocity of the collidable points.
    """

    return collidable_point_kinematics(model=model, data=data)[1]
