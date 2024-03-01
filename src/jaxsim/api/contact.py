import functools

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.physics.algos import soft_contacts

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


@functools.partial(jax.jit, static_argnames=["link_names"])
def in_contact(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
    *,
    link_names: tuple[str, ...] | None = None,
) -> jtp.Vector:
    """
    Return whether the links are in contact with the terrain.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_names:
            The names of the links to consider. If None, all links are considered.

    Returns:
        A boolean vector indicating whether the links are in contact with the terrain.
    """

    link_names = link_names if link_names is not None else model.link_names()

    if set(link_names) - set(model.link_names()) != set():
        raise ValueError("One or more link names are not part of the model")

    from jaxsim.physics.algos.soft_contacts import collidable_points_pos_vel

    W_p_Ci, _ = collidable_points_pos_vel(
        model=model.physics_model,
        q=data.state.physics_model.joint_positions,
        qd=data.state.physics_model.joint_velocities,
        xfb=data.state.physics_model.xfb(),
    )

    terrain_height = jax.vmap(lambda x, y: model.terrain.height(x=x, y=y))(
        W_p_Ci[0, :], W_p_Ci[1, :]
    )

    below_terrain = W_p_Ci[2, :] <= terrain_height

    links_in_contact = jax.vmap(
        lambda link_index: jnp.where(
            model.physics_model.gc.body == link_index,
            below_terrain,
            jnp.zeros_like(below_terrain, dtype=bool),
        ).any()
    )(jnp.arange(model.number_of_links()))

    return links_in_contact


@jax.jit
def estimate_good_soft_contacts_parameters(
    model: Model.JaxSimModel,
    static_friction_coefficient: jtp.FloatLike = 0.5,
    number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
    damping_ratio: jtp.FloatLike = 1.0,
    max_penetration: jtp.FloatLike | None = None,
) -> soft_contacts.SoftContactsParams:
    """
    Estimate good soft contacts parameters for the given model.

    Args:
        model: The model to consider.
        static_friction_coefficient: The static friction coefficient.
        number_of_active_collidable_points_steady_state:
            The number of active collidable points in steady state supporting
            the weight of the robot.
        damping_ratio: The damping ratio.
        max_penetration:
            The maximum penetration allowed in steady state when the robot is
            supported by the configured number of active collidable points.

    Returns:
        The estimated good soft contacts parameters.

    Note:
        This method provides a good starting point for the soft contacts parameters.
        The user is encouraged to fine-tune the parameters based on the
        specific application.
    """

    def estimate_model_height(model: Model.JaxSimModel) -> jtp.Float:
        """"""

        zero_data = Data.JaxSimModelData.build(
            model=model, soft_contacts_params=soft_contacts.SoftContactsParams()
        )

        W_pz_CoM = Model.com_position(model=model, data=zero_data)[2]

        if model.physics_model.is_floating_base:
            W_pz_C = collidable_point_positions(model=model, data=zero_data)[:, -1]
            return 2 * (W_pz_CoM - W_pz_C.min())

        return 2 * W_pz_CoM

    max_δ = (
        max_penetration
        if max_penetration is not None
        else 0.005 * estimate_model_height(model=model)
    )

    nc = number_of_active_collidable_points_steady_state

    sc_parameters = soft_contacts.SoftContactsParams.build_default_from_physics_model(
        physics_model=model.physics_model,
        static_friction_coefficient=static_friction_coefficient,
        max_penetration=max_δ,
        number_of_active_collidable_points_steady_state=nc,
        damping_ratio=damping_ratio,
    )

    return sc_parameters
