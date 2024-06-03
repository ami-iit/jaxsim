from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.utils.tracing import not_tracing

from .common import VelRepr
from .ode_data import ODEInput

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxSimModelReferences(js.common.ModelDataWithVelocityRepresentation):
    """
    Class containing the references for a `JaxSimModel` object.
    """

    input: ODEInput

    @staticmethod
    def zero(
        model: js.model.JaxSimModel,
        velocity_representation: VelRepr = VelRepr.Inertial,
    ) -> JaxSimModelReferences:
        """
        Create a `JaxSimModelReferences` object with zero references.

        Args:
            model: The model for which to create the zero references.
            velocity_representation: The velocity representation to use.

        Returns:
            A `JaxSimModelReferences` object with zero state.
        """

        return JaxSimModelReferences.build(
            model=model, velocity_representation=velocity_representation
        )

    @staticmethod
    def build(
        model: js.model.JaxSimModel,
        joint_force_references: jtp.Vector | None = None,
        link_forces: jtp.Matrix | None = None,
        data: js.data.JaxSimModelData | None = None,
        velocity_representation: VelRepr | None = None,
    ) -> JaxSimModelReferences:
        """
        Create a `JaxSimModelReferences` object with the given references.

        Args:
            model: The model for which to create the state.
            joint_force_references: The joint force references.
            link_forces: The link 6D forces in the desired representation.
            data:
                The data of the model, only needed if the velocity representation is
                not inertial-fixed.
            velocity_representation: The velocity representation to use.

        Returns:
            A `JaxSimModelReferences` object with the given references.
        """

        # Create or adjust joint force references.
        joint_force_references = jnp.atleast_1d(
            joint_force_references.squeeze()
            if joint_force_references is not None
            else jnp.zeros(model.dofs())
        ).astype(float)

        # Create or adjust link forces.
        f_L = jnp.atleast_2d(
            link_forces.squeeze()
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        ).astype(float)

        # Select the velocity representation.
        velocity_representation = (
            velocity_representation
            if velocity_representation is not None
            else (
                data.velocity_representation if data is not None else VelRepr.Inertial
            )
        )

        # Create a zero references object.
        references = JaxSimModelReferences(
            input=ODEInput.zero(model=model),
            velocity_representation=velocity_representation,
        )

        # Store the joint force references.
        references = references.set_joint_force_references(
            forces=joint_force_references,
            model=model,
            joint_names=model.joint_names(),
        )

        # Apply the link forces.
        references = references.apply_link_forces(
            forces=f_L,
            model=model,
            data=data,
            link_names=model.link_names(),
            additive=False,
        )

        return references

    def valid(self, model: js.model.JaxSimModel | None = None) -> bool:
        """
        Check if the current references are valid for the given model.

        Args:
            model: The model to check against.

        Returns:
            `True` if the current references are valid for the given model,
            `False` otherwise.
        """

        valid = True

        if model is not None:
            valid = valid and self.input.valid(model=model)

        return valid

    # ==================
    # Extract quantities
    # ==================

    @functools.partial(jax.jit, static_argnames=["link_names"])
    def link_forces(
        self,
        model: js.model.JaxSimModel | None = None,
        data: js.data.JaxSimModelData | None = None,
        link_names: tuple[str, ...] | None = None,
    ) -> jtp.Matrix:
        """
        Return the link forces expressed in the frame of the active representation.

        Args:
            model: The model to consider.
            data: The data of the considered model.
            link_names: The names of the links corresponding to the forces.

        Returns:
            If no model and no link names are provided, the link forces as a
            `(n_links,6)` matrix corresponding to the default link serialization
            of the original model used to build the actuation object.
            If a model is provided and no link names are provided, the link forces
            as a `(n_links,6)` matrix corresponding to the serialization of the
            provided model.
            If both a model and link names are provided, the link forces as a
            `(len(link_names),6)` matrix corresponding to the serialization of
            the passed link names vector.

        Note:
            The returned link forces are those passed as user inputs when integrating
            the dynamics of the model. They are summed with other forces related
            e.g. to the contact model and other kinematic constraints.
        """

        W_f_L = self.input.physics_model.f_ext

        # Return all link forces in inertial-fixed representation using the implicit
        # serialization.
        if model is None:
            if self.velocity_representation is not VelRepr.Inertial:
                msg = "Missing model to use a representation different from {}"
                raise ValueError(msg.format(VelRepr.Inertial.name))

            if link_names is not None:
                raise ValueError("Link names cannot be provided without a model")

            return self.input.physics_model.f_ext

        # If we have the model, we can extract the link names, if not provided.
        link_names = link_names if link_names is not None else model.link_names()
        link_idxs = js.link.names_to_idxs(link_names=link_names, model=model)

        # In inertial-fixed representation, we already have the link forces.
        if self.velocity_representation is VelRepr.Inertial:
            return W_f_L[link_idxs, :]

        if data is None:
            msg = "Missing model data to use a representation different from {}"
            raise ValueError(msg.format(VelRepr.Inertial.name))

        if not_tracing(self.input.physics_model.f_ext) and not data.valid(model=model):
            raise ValueError("The provided data is not valid for the model")

        # Helper function to convert a single 6D force to the active representation.
        def convert(f_L: jtp.Vector) -> jtp.Vector:
            return JaxSimModelReferences.inertial_to_other_representation(
                array=f_L,
                other_representation=self.velocity_representation,
                transform=data.base_transform(),
                is_force=True,
            )

        # Convert to the desired representation.
        f_L = jax.vmap(convert)(W_f_L[link_idxs, :])

        return f_L

    def joint_force_references(
        self,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> jtp.Vector:
        """
        Return the joint force references.

        Args:
            model: The model to consider.
            joint_names: The names of the joints corresponding to the forces.

        Returns:
            If no model and no joint names are provided, the joint forces as a
            `(DoFs,)` vector corresponding to the default joint serialization
            of the original model used to build the actuation object.
            If a model is provided and no joint names are provided, the joint forces
            as a `(DoFs,)` vector corresponding to the serialization of the
            provided model.
            If both a model and joint names are provided, the joint forces as a
            `(len(joint_names),)` vector corresponding to the serialization of
            the passed joint names vector.

        Note:
            The returned joint forces are those passed as user inputs when integrating
            the dynamics of the model. They are summed with other joint forces related
            e.g. to the enforcement of other kinematic constraints. Keep also in mind
            that the presence of joint friction and other similar effects can make the
            actual joint forces different from the references.
        """

        if model is None:
            if joint_names is not None:
                raise ValueError("Joint names cannot be provided without a model")

            return self.input.physics_model.tau

        if not_tracing(self.input.physics_model.tau) and not self.valid(model=model):
            msg = "The actuation object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()
        joint_idxs = js.joint.names_to_idxs(joint_names=joint_names, model=model)

        return jnp.atleast_1d(
            self.input.physics_model.tau[joint_idxs].squeeze()
        ).astype(float)

    # ================
    # Store quantities
    # ================

    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def set_joint_force_references(
        self,
        forces: jtp.VectorLike,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> Self:
        """
        Set the joint force references.

        Args:
            forces: The joint force references.
            model:
                The model to consider, only needed if a joint serialization different
                from the implicit one is used.
            joint_names: The names of the joints corresponding to the forces.

        Returns:
            A new `JaxSimModelReferences` object with the given joint force references.
        """

        forces = jnp.array(forces)

        def replace(forces: jtp.VectorLike) -> JaxSimModelReferences:
            return self.replace(
                validate=True,
                input=self.input.replace(
                    physics_model=self.input.physics_model.replace(
                        tau=jnp.atleast_1d(forces.squeeze()).astype(float)
                    )
                ),
            )

        if model is None:
            return replace(forces=forces)

        if not_tracing(forces) and not self.valid(model=model):
            msg = "The references object is not compatible with the provided model"
            raise ValueError(msg)

        joint_names = joint_names if joint_names is not None else model.joint_names()
        joint_idxs = js.joint.names_to_idxs(joint_names=joint_names, model=model)

        return replace(forces=self.input.physics_model.tau.at[joint_idxs].set(forces))

    @functools.partial(jax.jit, static_argnames=["link_names", "additive"])
    def apply_link_forces(
        self,
        forces: jtp.MatrixLike,
        model: js.model.JaxSimModel | None = None,
        data: js.data.JaxSimModelData | None = None,
        link_names: tuple[str, ...] | str | None = None,
        additive: bool = False,
    ) -> Self:
        """
        Apply the link forces.

        Args:
            forces: The link 6D forces in the active representation.
            model:
                The model to consider, only needed if a link serialization different
                from the implicit one is used.
            data:
                The data of the considered model, only needed if the velocity
                representation is not inertial-fixed.
            link_names: The names of the links corresponding to the forces.
            additive:
                Whether to add the forces to the existing ones instead of replacing them.

        Returns:
            A new `JaxSimModelReferences` object with the given link forces.

        Note:
            The link forces must be expressed in the active representation.
            Then, we always convert and store forces in inertial-fixed representation.
        """

        f_L = jnp.atleast_2d(forces).astype(float)

        # Helper function to replace the link forces.
        def replace(forces: jtp.MatrixLike) -> JaxSimModelReferences:
            return self.replace(
                validate=True,
                input=self.input.replace(
                    physics_model=self.input.physics_model.replace(
                        f_ext=jnp.atleast_2d(forces.squeeze()).astype(float)
                    )
                ),
            )

        # In this case, we allow only to set the inertial 6D forces to all links
        # using the implicit link serialization.
        if model is None:
            if self.velocity_representation is not VelRepr.Inertial:
                msg = "Missing model to use a representation different from {}"
                raise ValueError(msg.format(VelRepr.Inertial.name))

            if link_names is not None:
                raise ValueError("Link names cannot be provided without a model")

            W_f_L = f_L

            W_f0_L = (
                jnp.zeros_like(W_f_L)
                if not additive
                else self.input.physics_model.f_ext
            )

            return replace(forces=W_f0_L + W_f_L)

        # If we have the model, we can extract the link names if not provided.
        link_names = link_names if link_names is not None else model.link_names()

        # Make sure that the link names are a tuple if they are provided by the user.
        link_names = (link_names,) if isinstance(link_names, str) else link_names

        if len(link_names) != f_L.shape[0]:
            msg = "The number of link names ({}) must match the number of forces ({})"
            raise ValueError(msg.format(len(link_names), f_L.shape[0]))

        # Extract the link indices.
        link_idxs = js.link.names_to_idxs(link_names=link_names, model=model)

        # Compute the bias depending on whether we either set or add the link forces.
        W_f0_L = (
            jnp.zeros_like(f_L)
            if not additive
            else self.input.physics_model.f_ext[link_idxs, :]
        )

        # If inertial-fixed representation, we can directly store the link forces.
        if self.velocity_representation is VelRepr.Inertial:
            W_f_L = f_L
            return replace(
                forces=self.input.physics_model.f_ext.at[link_idxs, :].set(
                    W_f0_L + W_f_L
                )
            )

        if data is None:
            msg = "Missing model data to use a representation different from {}"
            raise ValueError(msg.format(VelRepr.Inertial.name))

        if not_tracing(forces) and not data.valid(model=model):
            raise ValueError("The provided data is not valid for the model")

        # Helper function to convert a single 6D force to the inertial representation.
        def convert(f_L: jtp.Vector) -> jtp.Vector:
            return JaxSimModelReferences.other_representation_to_inertial(
                array=f_L,
                other_representation=self.velocity_representation,
                transform=data.base_transform(),
                is_force=True,
            )

        W_f_L = jax.vmap(convert)(f_L)

        return replace(
            forces=self.input.physics_model.f_ext.at[link_idxs, :].set(W_f0_L + W_f_L)
        )
