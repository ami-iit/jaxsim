from __future__ import annotations

import abc

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import STANDARD_GRAVITY, Skew
from jaxsim.utils import CollidableShapeType, JaxsimDataclass

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from .detection import box_plane, cylinder_plane, sphere_plane

MAX_STIFFNESS = 1e6
MAX_DAMPING = 1e4

# Define a mapping from collidable shape types to distance functions.
_COLLISION_MAP = {
    CollidableShapeType.Sphere: sphere_plane,
    CollidableShapeType.Box: box_plane,
    CollidableShapeType.Cylinder: cylinder_plane,
}


@jax.jit
def compute_penetration_data(
    model: js.model.JaxSimModel,
    *,
    shape_type: CollidableShapeType,
    shape_size: jtp.Vector,
    link_transforms: jtp.Matrix,
    link_velocities: jtp.Matrix,
) -> tuple[jtp.Float, jtp.Float, jtp.Vector]:
    """
    Compute the penetration data (depth, rate, and terrain normal) of a collidable point.

    Args:
        model: The model to consider.
        shape_type: The type of the collidable shape.
        shape_size: The size parameters of the collidable shape.
        link_transforms: The transforms from the world frame to each link.
        link_velocities: The linear and angular velocities of each link.

    Returns:
        A tuple containing the penetration depth, the penetration velocity,
        the terrain normal, the contact point position, and the contact point velocity
        expressed in mixed representation.
    """

    W_H_L, W_ṗ_L = link_transforms, link_velocities

    # Pre-process the position and the linear velocity of the collidable point.
    # Note that we consider 3 candidate contact points also for spherical shapes,
    # in which the output is padded with zeros.
    # This is to allow parallel evaluation of the collision types.
    δ, W_H_C = jax.lax.switch(
        shape_type,
        (sphere_plane, box_plane, cylinder_plane),
        model.terrain,
        shape_size,
        W_H_L,
    )

    W_p_C = W_H_C[:, :3, 3]
    n̂ = W_H_C[:, :3, 2]

    def process_shape_kinematics(W_p_Ci: jtp.Vector) -> jtp.Vector:

        # Compute the velocity of the contact points.
        CW_ṗ_Ci = jnp.block([jnp.eye(3), -Skew.wedge(vector=W_p_Ci).squeeze()]) @ W_ṗ_L

        return CW_ṗ_Ci

    CW_ṗ_C = jax.vmap(process_shape_kinematics)(W_p_C)

    δ = jnp.maximum(0.0, -δ)

    δ̇ = -jax.vmap(jnp.dot)(CW_ṗ_C, n̂)
    δ̇ = jnp.where(δ > 0, δ̇, 0.0)

    return δ, δ̇, n̂, W_p_C, CW_ṗ_C


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

        # Compute the stiffness to get the desired steady-state penetration.
        # Note that this is dependent on the non-linear exponent used in
        # the damping term of the Hunt/Crossley model.
        if stiffness is None:
            # Compute the average support force on each collidable point.
            f_average = m * standard_gravity

            stiffness = f_average / jnp.power(δ_max, 1 + p)
            stiffness = jnp.clip(stiffness, 0, MAX_STIFFNESS)

        # Compute the damping using the damping ratio.
        critical_damping = 2 * jnp.sqrt(stiffness * m)
        if damping is None:
            damping = ξ * critical_damping
            damping = jnp.clip(damping, 0, MAX_DAMPING)

        return self.build(
            K=stiffness,
            D=damping,
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
    def _parameters_class(self) -> type[ContactsParams]:
        """
        Return the class of the contact parameters.

        Returns:
            The class of the contact parameters.
        """
        import importlib

        return getattr(
            importlib.import_module("jaxsim.rbda.contacts"),
            (
                self.__name__ + "Params"
                if isinstance(self, type)
                else self.__class__.__name__ + "Params"
            ),
        )

    @abc.abstractmethod
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

    @abc.abstractmethod
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
