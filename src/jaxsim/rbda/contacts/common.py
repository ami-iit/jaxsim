from __future__ import annotations

import abc
import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation
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

        Returns:
            A tuple containing as first element the computed 6D contact force applied to
            the contact points and expressed in the world frame, and as second element
            a dictionary of optional additional information.
        """

        pass

    def compute_link_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        **kwargs,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the link contact forces.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.

        Returns:
            A tuple containing as first element the 6D contact force applied to the
            links and expressed in the frame of the velocity representation of data,
            and as second element a dictionary of optional additional information.
        """

        # Compute the contact forces expressed in the inertial frame.
        # This function, contrarily to `compute_contact_forces`, already handles how
        # the optional kwargs should be passed to the specific contact models.
        W_f_C, aux_dict = js.contact.collidable_point_dynamics(
            model=model, data=data, **kwargs
        )

        # Compute the 6D forces applied to the links equivalent to the forces applied
        # to the frames associated to the collidable points.
        with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

            W_f_L = self.link_forces_from_contact_forces(
                model=model, data=data, contact_forces=W_f_C
            )

        # Store the link forces in the references object for easy conversion.
        references = js.references.JaxSimModelReferences.build(
            model=model,
            data=data,
            link_forces=W_f_L,
            velocity_representation=jaxsim.VelRepr.Inertial,
        )

        # Convert the link forces to the frame corresponding to the velocity
        # representation of data.
        with references.switch_velocity_representation(data.velocity_representation):
            f_L = references.link_forces(model=model, data=data)

        return f_L, aux_dict

    @staticmethod
    def link_forces_from_contact_forces(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        contact_forces: jtp.MatrixLike,
    ) -> jtp.Matrix:
        """
        Compute the link forces from the contact forces.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            contact_forces: The contact forces computed by the contact model.

        Returns:
            The 6D contact forces applied to the links and expressed in the frame of
            the velocity representation of data.
        """

        # Get the object storing the contact parameters of the model.
        contact_parameters = model.kin_dyn_parameters.contact_parameters

        # Extract the indices corresponding to the enabled collidable points.
        indices_of_enabled_collidable_points = (
            contact_parameters.indices_of_enabled_collidable_points
        )

        # Convert the contact forces to a JAX array.
        f_C = jnp.atleast_2d(jnp.array(contact_forces, dtype=float).squeeze())

        # Get the pose of the enabled collidable points.
        W_H_C = js.contact.transforms(model=model, data=data)[
            indices_of_enabled_collidable_points
        ]

        # Convert the contact forces to inertial-fixed representation.
        W_f_C = jax.vmap(
            lambda f_C, W_H_C: (
                ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                    array=f_C,
                    other_representation=data.velocity_representation,
                    transform=W_H_C,
                    is_force=True,
                )
            )
        )(f_C, W_H_C)

        # Construct the vector defining the parent link index of each collidable point.
        # We use this vector to sum the 6D forces of all collidable points rigidly
        # attached to the same link.
        parent_link_index_of_collidable_points = jnp.array(
            contact_parameters.body, dtype=int
        )[indices_of_enabled_collidable_points]

        # Create the mask that associate each collidable point to their parent link.
        # We use this mask to sum the collidable points to the right link.
        mask = parent_link_index_of_collidable_points[:, jnp.newaxis] == jnp.arange(
            model.number_of_links()
        )

        # Sum the forces of all collidable points rigidly attached to a body.
        # Since the contact forces W_f_C are expressed in the world frame,
        # we don't need any coordinate transformation.
        W_f_L = mask.T @ W_f_C

        # Compute the link transforms.
        W_H_L = (
            js.model.forward_kinematics(model=model, data=data)
            if data.velocity_representation is not jaxsim.VelRepr.Inertial
            else jnp.zeros(shape=(model.number_of_links(), 4, 4))
        )

        # Convert the inertial-fixed link forces to the velocity representation of data.
        f_L = jax.vmap(
            lambda W_f_L, W_H_L: (
                ModelDataWithVelocityRepresentation.inertial_to_other_representation(
                    array=W_f_L,
                    other_representation=data.velocity_representation,
                    transform=W_H_L,
                    is_force=True,
                )
            )
        )(W_f_L, W_H_L)

        return f_L

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
