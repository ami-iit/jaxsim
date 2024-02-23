from typing import Any, Protocol

import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.physics.algos.soft_contacts
import jaxsim.typing as jtp
from jaxsim import VelRepr, integrators
from jaxsim.integrators.common import Time
from jaxsim.math.quaternion import Quaternion
from jaxsim.physics.algos.soft_contacts import SoftContactsState
from jaxsim.physics.model.physics_model_state import PhysicsModelState
from jaxsim.simulation.ode_data import ODEState

from . import contact as Contact
from . import data as Data
from . import model as Model


class SystemDynamicsFromModelAndData(Protocol):
    def __call__(
        self,
        model: Model.JaxSimModel,
        data: Data.JaxSimModelData,
        **kwargs: dict[str, Any],
    ) -> tuple[ODEState, dict[str, Any]]: ...


def wrap_system_dynamics_for_integration(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
    *,
    system_dynamics: SystemDynamicsFromModelAndData,
    **kwargs,
) -> jaxsim.integrators.common.SystemDynamics[ODEState, ODEState]:
    """
    Wrap generic system dynamics operating on `JaxSimModel` and `JaxSimModelData`
    for integration with `jaxsim.integrators`.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        system_dynamics: The system dynamics to wrap.
        **kwargs: Additional kwargs to close over the system dynamics.

    Returns:
        The system dynamics closed over the model, the data, and the additional kwargs.
    """

    # We allow to close `system_dynamics` over additional kwargs.
    kwargs_closed = kwargs

    def f(x: ODEState, t: Time, **kwargs) -> tuple[ODEState, dict[str, Any]]:

        # Close f over the `data` parameter.
        with data.editable(validate=True) as data_rw:
            data_rw.state = x
            data_rw.time_ns = jnp.array(t * 1e9).astype(jnp.uint64)

        # Close f over the `model` parameter.
        return system_dynamics(model=model, data=data_rw, **kwargs_closed | kwargs)

    f: jaxsim.integrators.common.SystemDynamics[ODEState, ODEState]
    return f


# ==================================
# Functions defining system dynamics
# ==================================


@jax.jit
def system_velocity_dynamics(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
    *,
    joint_forces: jtp.Vector | None = None,
    external_forces: jtp.Vector | None = None,
) -> tuple[jtp.Vector, jtp.Vector, jtp.Matrix, dict[str, Any]]:
    """
    Compute the dynamics of the system velocity.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces: The joint forces to apply.
        external_forces: The external forces to apply to the links.

    Returns:
        A tuple containing the derivative of the base 6D velocity in inertial-fixed
        representation, the derivative of the joint velocities, the derivative of
        the material deformation, and the dictionary of auxiliary data returned by
        the system dynamics evalutation.
    """

    # Build joint torques if not provided
    τ = (
        jnp.atleast_1d(joint_forces.squeeze())
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    ).astype(float)

    # Build external forces if not provided
    f_ext = (
        jnp.atleast_2d(external_forces.squeeze())
        if external_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # ======================
    # Compute contact forces
    # ======================

    # Initialize the 6D forces W_f ∈ ℝ^{n_L × 6} applied to links due to contact
    # with the terrain.
    W_f_Li_terrain = jnp.zeros_like(f_ext).astype(float)

    # Initialize the 6D contact forces W_f ∈ ℝ^{n_c × 3} applied to collidable points,
    # expressed in the world frame.
    W_f_Ci = None

    # Initialize the derivative of the tangential deformation ṁ ∈ ℝ^{n_c × 3}.
    ṁ = jnp.zeros_like(data.state.soft_contacts.tangential_deformation).astype(float)

    if model.physics_model.gc.body.size > 0:
        # Compute the position and linear velocities (mixed representation) of
        # all collidable points belonging to the robot.
        W_p_Ci, W_ṗ_Ci = Contact.collidable_point_kinematics(model=model, data=data)

        # Compute the 3D forces applied to each collidable point.
        W_f_Ci, ṁ = jax.vmap(
            lambda p, ṗ, m: jaxsim.physics.algos.soft_contacts.SoftContacts(
                parameters=data.soft_contacts_params, terrain=model.terrain
            ).contact_model(position=p, velocity=ṗ, tangential_deformation=m)
        )(W_p_Ci, W_ṗ_Ci, data.state.soft_contacts.tangential_deformation.T)

        # Sum the forces of all collidable points rigidly attached to a body.
        # Since the contact forces W_f_Ci are expressed in the world frame,
        # we don't need any coordinate transformation.
        W_f_Li_terrain = jax.vmap(
            lambda nc: (
                jnp.vstack(jnp.equal(model.physics_model.gc.body, nc).astype(int))
                * W_f_Ci
            ).sum(axis=0)
        )(jnp.arange(model.number_of_links()))

    # ====================
    # Enforce joint limits
    # ====================

    # TODO: enforce joint limits
    τ_position_limit = jnp.zeros_like(τ).astype(float)

    # ====================
    # Joint friction model
    # ====================

    τ_friction = jnp.zeros_like(τ).astype(float)

    if model.dofs() > 0:
        # Static and viscous joint friction parameters
        kc = jnp.array(list(model.physics_model._joint_friction_static.values()))
        kv = jnp.array(list(model.physics_model._joint_friction_viscous.values()))

        # Compute the joint friction torque
        τ_friction = -(
            jnp.diag(kc) @ jnp.sign(data.state.physics_model.joint_positions)
            + jnp.diag(kv) @ data.state.physics_model.joint_velocities
        )

    # ========================
    # Compute forward dynamics
    # ========================

    # Compute the total joint forces
    τ_total = τ + τ_friction + τ_position_limit

    # Compute the total external 6D forces applied to the links
    W_f_L_total = f_ext + W_f_Li_terrain

    # - Joint accelerations: s̈ ∈ ℝⁿ
    # - Base inertial-fixed acceleration: W_v̇_WB = (W_p̈_B, W_ω̇_B) ∈ ℝ⁶
    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        W_v̇_WB, s̈ = Model.forward_dynamics_aba(
            model=model,
            data=data,
            joint_forces=τ_total,
            external_forces=W_f_L_total,
        )

    return W_v̇_WB, s̈, ṁ.T, dict()


@jax.jit
def system_position_dynamics(
    model: Model.JaxSimModel, data: Data.JaxSimModelData
) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    """
    Compute the dynamics of the system position.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        A tuple containing the derivative of the base position, the derivative of the
        base quaternion, and the derivative of the joint positions.
    """

    ṡ = data.state.physics_model.joint_velocities
    W_Q_B = data.state.physics_model.base_quaternion

    with data.switch_velocity_representation(velocity_representation=VelRepr.Mixed):
        W_ṗ_B = data.base_velocity()[0:3]

    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        W_ω_WB = data.base_velocity()[3:6]

    W_Q̇_B = Quaternion.derivative(
        quaternion=W_Q_B,
        omega=W_ω_WB,
        omega_in_body_fixed=False,
    ).squeeze()

    return W_ṗ_B, W_Q̇_B, ṡ


@jax.jit
def system_dynamics(
    model: Model.JaxSimModel,
    data: Data.JaxSimModelData,
    *,
    joint_forces: jtp.Vector | None = None,
    external_forces: jtp.Vector | None = None,
) -> tuple[ODEState, dict[str, Any]]:
    """
    Compute the dynamics of the system.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces: The joint forces to apply.
        external_forces: The external forces to apply to the links.

    Returns:
        A tuple with an `ODEState` object storing in each of its attributes the
        corresponding derivative, and the dictionary of auxiliary data returned
        by the system dynamics evaluation.
    """

    # Compute the accelerations and the material deformation rate.
    W_v̇_WB, s̈, ṁ, aux_dict = system_velocity_dynamics(
        model=model,
        data=data,
        joint_forces=joint_forces,
        external_forces=external_forces,
    )

    # Extract the velocities.
    W_ṗ_B, W_Q̇_B, ṡ = system_position_dynamics(model=model, data=data)

    # Create an ODEState object populated with the derivative of each leaf.
    # Our integrators, operating on generic pytrees, will be able to handle it
    # automatically as state derivative.
    ode_state_derivative = ODEState.build(
        physics_model_state=PhysicsModelState.build(
            base_position=W_ṗ_B,
            base_quaternion=W_Q̇_B,
            joint_positions=ṡ,
            base_linear_velocity=W_v̇_WB[0:3],
            base_angular_velocity=W_v̇_WB[3:6],
            joint_velocities=s̈,
        ),
        soft_contacts_state=SoftContactsState.build(
            tangential_deformation=ṁ,
        ),
    )

    return ode_state_derivative, aux_dict
