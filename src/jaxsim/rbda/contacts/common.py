from __future__ import annotations

import abc
import dataclasses
import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.math import STANDARD_GRAVITY
from jaxsim.utils import JaxsimDataclass

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@functools.partial(jax.jit, static_argnames=("terrain",))
def compute_penetration_data(
    p: jtp.VectorLike,
    v: jtp.VectorLike,
    terrain: jaxsim.terrain.Terrain,
) -> tuple[jtp.Float, jtp.Float, jtp.Vector]:
    """
    Compute the penetration data (depth, rate, and terrain normal) of a collidable point.

    Args:
        p: The position of the collidable point.
        v:
            The linear velocity of the point (linear component of the mixed 6D velocity
            of the implicit frame `C = (W_p_C, [W])` associated to the point).
        terrain: The considered terrain.

    Returns:
        A tuple containing the penetration depth, the penetration velocity,
        and the considered terrain normal.
    """

    # Pre-process the position and the linear velocity of the collidable point.
    W_ṗ_C = jnp.array(v).squeeze()
    px, py, pz = jnp.array(p).squeeze()

    # Compute the terrain normal and the contact depth.
    n̂ = terrain.normal(x=px, y=py).squeeze()
    h = jnp.array([0, 0, terrain.height(x=px, y=py) - pz])

    # Compute the penetration depth normal to the terrain.
    δ = jnp.maximum(0.0, jnp.dot(h, n̂))

    # Compute the penetration normal velocity.
    δ_dot = -jnp.dot(W_ṗ_C, n̂)

    # Enforce the penetration rate to be zero when the penetration depth is zero.
    δ_dot = jnp.where(δ > 0, δ_dot, 0.0)

    return δ, δ_dot, n̂


class ContactsParams(JaxsimDataclass):
    """
    Abstract class representing the parameters of a contact model.

    Note:
        This class is supposed to store only the tunable parameters of the contact
        model, i.e. all those parameters that can be changed during runtime.
        If the contact model has also static parameters, they should be stored
        in the corresponding `ContactModel` class.
    """

    @classmethod
    @abc.abstractmethod
    def build(cls: type[Self], **kwargs) -> Self:
        """
        Create a `ContactsParams` instance with specified parameters.

        Returns:
            The `ContactsParams` instance.
        """
        pass

    def build_default_from_jaxsim_model(
        self: type[Self],
        model: js.model.JaxSimModel,
        *,
        stiffness: jtp.FloatLike | None = None,
        damping: jtp.FloatLike | None = None,
        standard_gravity: jtp.FloatLike = STANDARD_GRAVITY,
        static_friction_coefficient: jtp.FloatLike = 0.5,
        max_penetration: jtp.FloatLike = 0.001,
        number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
        damping_ratio: jtp.FloatLike = 1.0,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
        **kwargs,
    ) -> Self:
        """
        Create a `ContactsParams` instance with default parameters.

        Args:
            model: The robot model considered by the contact model.
            stiffness: The stiffness of the contact model.
            damping: The damping of the contact model.
            standard_gravity: The standard gravity acceleration.
            static_friction_coefficient: The static friction coefficient.
            max_penetration: The maximum penetration depth.
            number_of_active_collidable_points_steady_state:
                The number of active collidable points in steady state.
            damping_ratio: The damping ratio.
            p: The first parameter of the contact model.
            q: The second parameter of the contact model.
            **kwargs: Optional additional arguments.

        Returns:
            The `ContactsParams` instance.

        Note:
            The `stiffness` is intended as the terrain stiffness in the Soft Contacts model,
            while it is the Baumgarte stabilization stiffness in the Rigid Contacts model.

            The `damping` is intended as the terrain damping in the Soft Contacts model,
            while it is the Baumgarte stabilization damping in the Rigid Contacts model.

            The `damping_ratio` parameter allows to operate on the following conditions:
            - ξ > 1.0: over-damped
            - ξ = 1.0: critically damped
            - ξ < 1.0: under-damped
        """

        # Use symbols for input parameters.
        ξ = damping_ratio
        δ_max = max_penetration
        μc = static_friction_coefficient

        # Compute the total mass of the model.
        m = jnp.array(model.kin_dyn_parameters.link_parameters.mass).sum()

        # Rename the standard gravity.
        g = standard_gravity

        # Compute the average support force on each collidable point.
        f_average = m * g / number_of_active_collidable_points_steady_state

        # Compute the stiffness to get the desired steady-state penetration.
        # Note that this is dependent on the non-linear exponent used in
        # the damping term of the Hunt/Crossley model.
        K = f_average / jnp.power(δ_max, 1 + p) if stiffness is None else stiffness

        # Compute the damping using the damping ratio.
        critical_damping = 2 * jnp.sqrt(K * m)
        D = ξ * critical_damping if damping is None else damping

        return self.build(
            K=K,
            D=D,
            mu=μc,
            p=p,
            q=q,
            **kwargs,
        )

    @abc.abstractmethod
    def valid(self, **kwargs) -> jtp.BoolLike:
        """
        Check if the parameters are valid.

        Returns:
            True if the parameters are valid, False otherwise.
        """
        pass


class ContactModel(JaxsimDataclass):
    """
    Abstract class representing a contact model.
    """

    @classmethod
    @abc.abstractmethod
    def build(
        cls: type[Self],
        **kwargs,
    ) -> Self:
        """
        Create a `ContactModel` instance with specified parameters.

        Returns:
            The `ContactModel` instance.
        """

        pass

    @abc.abstractmethod
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        **kwargs,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the contact forces.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            **kwargs: Optional additional arguments, specific to the contact model.

        Returns:
            A tuple containing as first element the computed 6D contact force applied to
            the contact points and expressed in the world frame, and as second element
            a dictionary of optional additional information.
        """

        pass

    @classmethod
    def zero_state_variables(cls, model: js.model.JaxSimModel) -> dict[str, jtp.Array]:
        """
        Build zero state variables of the contact model.

        Args:
            model: The robot model considered by the contact model.

        Note:
            There are contact models that require to extend the state vector of the
            integrated ODE system with additional variables. Our integrators are
            capable of operating on a generic state, as long as it is a PyTree.
            This method builds the zero state variables of the contact model as a
            dictionary of JAX arrays.

        Returns:
            A dictionary storing the zero state variables of the contact model.
        """

        return {}

    @property
    def _parameters_class(cls) -> type[ContactsParams]:
        """
        Return the class of the contact parameters.

        Returns:
            The class of the contact parameters.
        """
        import importlib

        return getattr(
            importlib.import_module("jaxsim.rbda.contacts"),
            (
                cls.__name__ + "Params"
                if isinstance(cls, type)
                else cls.__class__.__name__ + "Params"
            ),
        )

    def update_contact_state(
        self: type[Self], old_contact_state: dict[str, jtp.Array]
    ) -> dict[str, jtp.Array]:
        """
        Update the contact state.

        Args:
            old_contact_state: The old contact state.

        Returns:
            The updated contact state.
        """

        # Import the contact models to avoid circular imports.
        from .relaxed_rigid import RelaxedRigidContacts
        from .rigid import RigidContacts
        from .soft import SoftContacts

        match self:
            case SoftContacts():
                return {"tangential_deformation": old_contact_state["m_dot"]}
            case RigidContacts() | RelaxedRigidContacts():
                return {}

    @jax.jit
    @js.common.named_scope
    def update_velocity_after_impact(
        self: type[Self], model: js.model.JaxSimModel, data: js.data.JaxSimModelData
    ) -> js.data.JaxSimModelData:
        """
        Update the velocity after an impact.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.

        Returns:
            The updated data of the considered model.
        """

        # Import the rigid contact model to avoid circular imports.
        from jaxsim.api.common import VelRepr

        from .rigid import RigidContacts

        if isinstance(self, RigidContacts):
            # Extract the indices corresponding to the enabled collidable points.
            indices_of_enabled_collidable_points = (
                model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
            )

            W_p_C = js.contact.collidable_point_positions(model, data)[
                indices_of_enabled_collidable_points
            ]

            # Compute the penetration depth of the collidable points.
            δ, *_ = jax.vmap(
                jaxsim.rbda.contacts.common.compute_penetration_data,
                in_axes=(0, 0, None),
            )(W_p_C, jnp.zeros_like(W_p_C), model.terrain)

            original_representation = data.velocity_representation

            with data.switch_velocity_representation(VelRepr.Mixed):
                J_WC = js.contact.jacobian(model, data)[
                    indices_of_enabled_collidable_points
                ]
                M = js.model.free_floating_mass_matrix(model, data)
                BW_ν_pre_impact = data.generalized_velocity

                # Compute the impact velocity.
                # It may be discontinuous in case new contacts are made.
                BW_ν_post_impact = (
                    jaxsim.rbda.contacts.RigidContacts.compute_impact_velocity(
                        generalized_velocity=BW_ν_pre_impact,
                        inactive_collidable_points=(δ <= 0),
                        M=M,
                        J_WC=J_WC,
                    )
                )

                BW_ν_post_impact_inertial = data.other_representation_to_inertial(
                    array=BW_ν_post_impact[0:6],
                    other_representation=VelRepr.Mixed,
                    transform=data._base_transform.at[0:3, 0:3].set(jnp.eye(3)),
                    is_force=False,
                )

                # Reset the generalized velocity.
                data = dataclasses.replace(
                    data,
                    velocity_representation=original_representation,
                    _base_linear_velocity=BW_ν_post_impact_inertial[0:3],
                    _base_angular_velocity=BW_ν_post_impact_inertial[3:6],
                    _joint_velocities=BW_ν_post_impact[6:],
                )

        return data
