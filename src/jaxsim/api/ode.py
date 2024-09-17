from typing import Any, Protocol

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim.integrators import Time
from jaxsim.math import Quaternion

from .common import VelRepr
from .ode_data import ODEState


class SystemDynamicsFromModelAndData(Protocol):
    def __call__(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        **kwargs: dict[str, Any],
    ) -> tuple[ODEState, dict[str, Any]]: ...


def wrap_system_dynamics_for_integration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
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
    kwargs_closed = kwargs.copy()

    # Create a local copy of model and data.
    # The wrapped dynamics will hold a reference of this object.
    model_closed = model.copy()
    data_closed = data.copy().replace(
        state=js.ode_data.ODEState.zero(model=model_closed, data=data)
    )

    def f(x: ODEState, t: Time, **kwargs_f) -> tuple[ODEState, dict[str, Any]]:

        # Allow caller to override the closed data and model objects.
        data_f = kwargs_f.pop("data", data_closed)
        model_f = kwargs_f.pop("model", model_closed)

        # Update the state and time stored inside data.
        with data_f.editable(validate=True) as data_rw:
            data_rw.state = x
            data_rw.time_ns = jnp.array(t * 1e9).astype(data_rw.time_ns.dtype)

        # Evaluate the system dynamics, allowing to override the kwargs originally
        # passed when the closure was created.
        return system_dynamics(
            model=model_f,
            data=data_rw,
            **(kwargs_closed | kwargs_f),
        )

    f: jaxsim.integrators.common.SystemDynamics[ODEState, ODEState]
    return f


# ==================================
# Functions defining system dynamics
# ==================================


@jax.jit
def system_velocity_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.Vector | None = None,
    link_forces: jtp.Vector | None = None,
) -> tuple[jtp.Vector, jtp.Vector, dict[str, Any]]:
    """
    Compute the dynamics of the system velocity.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces: The joint forces to apply.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.

    Returns:
        A tuple containing the derivative of the base 6D velocity in inertial-fixed
        representation, the derivative of the joint velocities, and auxiliary data
        returned by the system dynamics evaluation.
    """

    # Build link forces if not provided.
    # These forces are expressed in the frame corresponding to the velocity
    # representation of data.
    O_f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # We expect that the 6D forces included in the `link_forces` argument are expressed
    # in the frame corresponding to the velocity representation of `data`.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        link_forces=O_f_L,
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # ======================
    # Compute contact forces
    # ======================

    # Initialize the 6D forces W_f ∈ ℝ^{n_L × 6} applied to links due to contact
    # with the terrain.
    W_f_Li_terrain = jnp.zeros_like(O_f_L).astype(float)

    # Initialize a dictionary of auxiliary data.
    # This dictionary is used to store additional data computed by the contact model.
    aux_data = {}

    if len(model.kin_dyn_parameters.contact_parameters.body) > 0:

        # Note: the following code should be kept in sync with the function
        # `jaxsim.api.model.link_contact_forces`. We cannot merge them since
        # here we need to get also aux_data.

        # Compute the 6D forces W_f ∈ ℝ^{n_c × 6} applied to each collidable point
        # along with contact-specific auxiliary states.
        with data.switch_velocity_representation(VelRepr.Inertial):
            W_f_Ci, aux_data = js.contact.collidable_point_dynamics(
                model=model,
                data=data,
                link_forces=references.link_forces(model=model, data=data),
            )

        # Construct the vector defining the parent link index of each collidable point.
        # We use this vector to sum the 6D forces of all collidable points rigidly
        # attached to the same link.
        parent_link_index_of_collidable_points = jnp.array(
            model.kin_dyn_parameters.contact_parameters.body, dtype=int
        )

        # Sum the forces of all collidable points rigidly attached to a body.
        # Since the contact forces W_f_Ci are expressed in the world frame,
        # we don't need any coordinate transformation.
        mask = parent_link_index_of_collidable_points[:, jnp.newaxis] == jnp.arange(
            model.number_of_links()
        )

        W_f_Li_terrain = mask.T @ W_f_Ci

    # ===========================
    # Compute system acceleration
    # ===========================

    # Compute the total link forces
    with (
        data.switch_velocity_representation(VelRepr.Inertial),
        references.switch_velocity_representation(VelRepr.Inertial),
    ):
        references = references.apply_link_forces(
            model=model,
            data=data,
            forces=W_f_Li_terrain,
            additive=True,
        )

        # Get the link forces in inertial representation
        f_L_total = references.link_forces(model=model, data=data)

        v̇_WB, s̈ = system_acceleration(
            model=model, data=data, joint_forces=joint_forces, link_forces=f_L_total
        )

    return v̇_WB, s̈, aux_data


def system_acceleration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the system acceleration in the active representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces: The joint forces to apply.
        link_forces:
            The 6D forces to apply to the links expressed in the same representation of data.

    Returns:
        A tuple containing the base 6D acceleration in in the active representation
        and the joint accelerations.
    """

    # ====================
    # Validate input data
    # ====================

    # Build link forces if not provided.
    f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # Build joint torques if not provided.
    τ = (
        jnp.atleast_1d(joint_forces.squeeze())
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    ).astype(float)

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
        kc = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_static
        ).astype(float)
        kv = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_viscous
        ).astype(float)

        # Compute the joint friction torque.
        τ_friction = -(
            jnp.diag(kc) @ jnp.sign(data.state.physics_model.joint_velocities)
            + jnp.diag(kv) @ data.state.physics_model.joint_velocities
        )

    # ========================
    # Compute forward dynamics
    # ========================

    # Compute the total joint forces.
    τ_total = τ + τ_friction + τ_position_limit

    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        velocity_representation=data.velocity_representation,
        joint_force_references=τ_total,
        link_forces=f_L,
    )

    # - Joint accelerations: s̈ ∈ ℝⁿ
    # - Base acceleration: v̇_WB ∈ ℝ⁶
    v̇_WB, s̈ = js.model.forward_dynamics_aba(
        model=model,
        data=data,
        joint_forces=references.joint_force_references(model=model),
        link_forces=references.link_forces(model=model, data=data),
    )

    return v̇_WB, s̈


@jax.jit
def system_position_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    """
    Compute the dynamics of the system position.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient for adjusting the quaternion norm.

    Returns:
        A tuple containing the derivative of the base position, the derivative of the
        base quaternion, and the derivative of the joint positions.
    """

    ṡ = data.joint_velocities(model=model)
    W_Q_B = data.base_orientation(dcm=False)

    with data.switch_velocity_representation(velocity_representation=VelRepr.Mixed):
        W_ṗ_B = data.base_velocity()[0:3]

    with data.switch_velocity_representation(velocity_representation=VelRepr.Inertial):
        W_ω_WB = data.base_velocity()[3:6]

    W_Q̇_B = Quaternion.derivative(
        quaternion=W_Q_B,
        omega=W_ω_WB,
        omega_in_body_fixed=False,
        K=baumgarte_quaternion_regularization,
    ).squeeze()

    return W_ṗ_B, W_Q̇_B, ṡ


@jax.jit
def system_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.Vector | None = None,
    link_forces: jtp.Vector | None = None,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> tuple[ODEState, dict[str, Any]]:
    """
    Compute the dynamics of the system.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces: The joint forces to apply.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient used to adjust the norm of the
            quaternion (only used in integrators not operating on the SO(3) manifold).

    Returns:
        A tuple with an `ODEState` object storing in each of its attributes the
        corresponding derivative, and the dictionary of auxiliary data returned
        by the system dynamics evaluation.
    """
    from jaxsim.rbda.contacts.relaxed_rigid import RelaxedRigidContacts
    from jaxsim.rbda.contacts.rigid import RigidContacts
    from jaxsim.rbda.contacts.soft import SoftContacts

    # Compute the accelerations and the material deformation rate.
    W_v̇_WB, s̈, aux_dict = system_velocity_dynamics(
        model=model,
        data=data,
        joint_forces=joint_forces,
        link_forces=link_forces,
    )

    ode_state_kwargs = {}

    match model.contact_model:
        case SoftContacts():
            ode_state_kwargs["tangential_deformation"] = aux_dict["m_dot"]

        case RigidContacts() | RelaxedRigidContacts():
            pass

        case _:
            raise ValueError("Unable to determine contact state class prefix.")

    # Extract the velocities.
    W_ṗ_B, W_Q̇_B, ṡ = system_position_dynamics(
        model=model,
        data=data,
        baumgarte_quaternion_regularization=baumgarte_quaternion_regularization,
    )

    # Create an ODEState object populated with the derivative of each leaf.
    # Our integrators, operating on generic pytrees, will be able to handle it
    # automatically as state derivative.
    ode_state_derivative = ODEState.build_from_jaxsim_model(
        model=model,
        base_position=W_ṗ_B,
        base_quaternion=W_Q̇_B,
        joint_positions=ṡ,
        base_linear_velocity=W_v̇_WB[0:3],
        base_angular_velocity=W_v̇_WB[3:6],
        joint_velocities=s̈,
        **ode_state_kwargs,
    )

    return ode_state_derivative, aux_dict
