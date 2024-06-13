from __future__ import annotations

import importlib

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.utils import JaxsimDataclass

# =============================================================================
# Define the input and state of the ODE system defining the integrated dynamics
# =============================================================================

# Note: the ODE system is the combination of the floating-base dynamics and the
#       soft-contacts dynamics.


@jax_dataclasses.pytree_dataclass
class ODEInput(JaxsimDataclass):
    """
    The input to the ODE system.

    Attributes:
        physics_model: The input to the physics model.
    """

    physics_model: PhysicsModelInput

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
        joint_forces: jtp.VectorJax | None = None,
        link_forces: jtp.MatrixJax | None = None,
    ) -> ODEInput:
        """
        Build an `ODEInput` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the ODE input.
            joint_forces: The vector of joint forces.
            link_forces: The matrix of external forces applied to the links.

        Returns:
            The `ODEInput` built from the `JaxSimModel`.

        Note:
            If any of the input components are not provided, they are built from the
            `JaxSimModel` and initialized to zero.
        """

        return ODEInput.build(
            physics_model_input=PhysicsModelInput.build_from_jaxsim_model(
                model=model,
                joint_forces=joint_forces,
                link_forces=link_forces,
            ),
            model=model,
        )

    @staticmethod
    def build(
        physics_model_input: PhysicsModelInput | None = None,
        model: js.model.JaxSimModel | None = None,
    ) -> ODEInput:
        """
        Build an `ODEInput` from a `PhysicsModelInput`.

        Args:
            physics_model_input: The `PhysicsModelInput` associated with the ODE input.
            model: The `JaxSimModel` associated with the ODE input.

        Returns:
            A `ODEInput` instance.
        """

        physics_model_input = (
            physics_model_input
            if physics_model_input is not None
            else PhysicsModelInput.zero(model=model)
        )

        return ODEInput(physics_model=physics_model_input)

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> ODEInput:
        """
        Build a zero `ODEInput` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the ODE input.

        Returns:
            A zero `ODEInput` instance.
        """

        return ODEInput.build(model=model)

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `ODEInput` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `ODEInput` against.

        Returns:
            `True` if the ODE input is valid for the given model, `False` otherwise.
        """

        return self.physics_model.valid(model=model)


@jax_dataclasses.pytree_dataclass
class ODEState(JaxsimDataclass):
    """
    The state of the ODE system.

    Attributes:
        physics_model: The state of the physics model.
        contact_state: The state of the contact model.
    """

    physics_model: PhysicsModelState
    contact_state: js.contact.ContactsState

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
        joint_positions: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        base_position: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        tangential_deformation: jtp.Matrix | None = None,
    ) -> ODEState:
        """
        Build an `ODEState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the ODE state.
            joint_positions: The vector of joint positions.
            joint_velocities: The vector of joint velocities.
            base_position: The 3D position of the base link.
            base_quaternion: The quaternion defining the orientation of the base link.
            base_linear_velocity:
                The linear velocity of the base link in inertial-fixed representation.
            base_angular_velocity:
                The angular velocity of the base link in inertial-fixed representation.
            tangential_deformation:
                The matrix of 3D tangential material deformations corresponding to
                each collidable point.

        Returns:
            The `ODEState` built from the `JaxSimModel`.

        Note:
            If any of the state components are not provided, they are built from the
            `JaxSimModel` and initialized to zero.
            The contact model will be inherited from the `model`.
        """

        # Get the contact model from the `JaxSimModel`
        prefix = type(model.contact_model).__name__.split("Contact")[0]

        if prefix:
            module_name = f"{prefix.lower()}_contacts"
            class_name = f"{prefix.capitalize()}ContactsState"
        else:
            raise ValueError("Unable to determine contact state class prefix.")

        return ODEState.build(
            model=model,
            physics_model_state=PhysicsModelState.build_from_jaxsim_model(
                model=model,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                base_position=base_position,
                base_quaternion=base_quaternion,
                base_linear_velocity=base_linear_velocity,
                base_angular_velocity=base_angular_velocity,
            ),
            contact_state=getattr(
                importlib.import_module(f"jaxsim.api.{module_name}"), class_name
            ).build_from_jaxsim_model(
                model=model,
                **(
                    dict(tangential_deformation=tangential_deformation)
                    if tangential_deformation is not None
                    else dict()
                ),
            ),
        )

    @staticmethod
    def build(
        physics_model_state: PhysicsModelState | None = None,
        contact_state: js.contact.ContactsState | None = None,
        model: js.model.JaxSimModel | None = None,
    ) -> ODEState:
        """
        Build an `ODEState` from a `PhysicsModelState` and a `ContactsState`.

        Args:
            physics_model_state: The state of the physics model.
            contact_state: The state of the contact model.
            model: The `JaxSimModel` associated with the ODE state.

        Returns:
            A `ODEState` instance.
        """

        physics_model_state = (
            physics_model_state
            if physics_model_state is not None
            else PhysicsModelState.zero(model=model)
        )

        # Get the contact model from the `JaxSimModel`
        try:
            prefix = type(model.contact_model).__name__.split("Contact")[0]
        except AttributeError:
            logging.warning(
                "Unable to determine contact state class prefix. Using default soft contacts."
            )
            prefix = "Soft"

        module_name = f"{prefix.lower()}_contacts"
        class_name = f"{prefix.capitalize()}ContactsState"

        contact_state = (
            contact_state
            if contact_state is not None
            else getattr(
                importlib.import_module(f"jaxsim.api.{module_name}"), class_name
            ).zero(model=model)
        )

        return ODEState(physics_model=physics_model_state, contact_state=contact_state)

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> ODEState:
        """
        Build a zero `ODEState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the ODE state.

        Returns:
            A zero `ODEState` instance.
        """

        model_state = ODEState.build(model=model)

        return model_state

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `ODEState` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `ODEState` against.

        Returns:
            `True` if the ODE state is valid for the given model, `False` otherwise.
        """

        return self.physics_model.valid(model=model) and self.contact_state.valid(
            model=model
        )


# ==================================================
# Define the input and state of floating-base robots
# ==================================================


@jax_dataclasses.pytree_dataclass
class PhysicsModelState(JaxsimDataclass):
    """
    Class storing the state of the physics model dynamics.

    Attributes:
        joint_positions: The vector of joint positions.
        joint_velocities: The vector of joint velocities.
        base_position: The 3D position of the base link.
        base_quaternion: The quaternion defining the orientation of the base link.
        base_linear_velocity:
            The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in inertial-fixed representation.

    """

    # Joint state
    joint_positions: jtp.Vector
    joint_velocities: jtp.Vector

    # Base state
    base_position: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )
    base_quaternion: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.array([1.0, 0, 0, 0])
    )
    base_linear_velocity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )
    base_angular_velocity: jtp.Vector = jax_dataclasses.field(
        default_factory=lambda: jnp.zeros(3)
    )

    def __hash__(self) -> int:

        return hash(
            (
                hash(tuple(jnp.atleast_1d(self.joint_positions.flatten().tolist()))),
                hash(tuple(jnp.atleast_1d(self.joint_velocities.flatten().tolist()))),
                hash(tuple(self.base_position.flatten().tolist())),
                hash(tuple(self.base_quaternion.flatten().tolist())),
            )
        )

    def __eq__(self, other: PhysicsModelState) -> bool:

        if not isinstance(other, PhysicsModelState):
            return False

        return hash(self) == hash(other)

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
        joint_positions: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        base_position: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
    ) -> PhysicsModelState:
        """
        Build a `PhysicsModelState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the state.
            joint_positions: The vector of joint positions.
            joint_velocities: The vector of joint velocities.
            base_position: The 3D position of the base link.
            base_quaternion: The quaternion defining the orientation of the base link.
            base_linear_velocity:
                The linear velocity of the base link in inertial-fixed representation.
            base_angular_velocity:
                The angular velocity of the base link in inertial-fixed representation.

        Note:
            If any of the state components are not provided, they are built from the
            `JaxSimModel` and initialized to zero.

        Returns:
            A `PhysicsModelState` instance.
        """

        return PhysicsModelState.build(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_position=base_position,
            base_quaternion=base_quaternion,
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=base_angular_velocity,
            number_of_dofs=model.dofs(),
        )

    @staticmethod
    def build(
        joint_positions: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        base_position: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        number_of_dofs: jtp.Int | None = None,
    ) -> PhysicsModelState:
        """
        Build a `PhysicsModelState`.

        Args:
            joint_positions: The vector of joint positions.
            joint_velocities: The vector of joint velocities.
            base_position: The 3D position of the base link.
            base_quaternion: The quaternion defining the orientation of the base link.
            base_linear_velocity:
                The linear velocity of the base link in inertial-fixed representation.
            base_angular_velocity:
                The angular velocity of the base link in inertial-fixed representation.
            number_of_dofs:
                The number of degrees of freedom of the physics model.

        Returns:
            A `PhysicsModelState` instance.
        """

        joint_positions = (
            joint_positions
            if joint_positions is not None
            else jnp.zeros(number_of_dofs)
        )

        joint_velocities = (
            joint_velocities
            if joint_velocities is not None
            else jnp.zeros(number_of_dofs)
        )

        base_position = base_position if base_position is not None else jnp.zeros(3)

        base_quaternion = (
            base_quaternion
            if base_quaternion is not None
            else jnp.array([1.0, 0, 0, 0])
        )

        base_linear_velocity = (
            base_linear_velocity if base_linear_velocity is not None else jnp.zeros(3)
        )

        base_angular_velocity = (
            base_angular_velocity if base_angular_velocity is not None else jnp.zeros(3)
        )

        physics_model_state = PhysicsModelState(
            joint_positions=jnp.array(joint_positions, dtype=float),
            joint_velocities=jnp.array(joint_velocities, dtype=float),
            base_position=jnp.array(base_position, dtype=float),
            base_quaternion=jnp.array(base_quaternion, dtype=float),
            base_linear_velocity=jnp.array(base_linear_velocity, dtype=float),
            base_angular_velocity=jnp.array(base_angular_velocity, dtype=float),
        )

        # assert state.valid(physics_model)
        return physics_model_state

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> PhysicsModelState:
        """
        Build a `PhysicsModelState` with all components initialized to zero.

        Args:
            model: The `JaxSimModel` associated with the state.

        Returns:
            A `PhysicsModelState` instance.
        """

        return PhysicsModelState.build_from_jaxsim_model(model=model)

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `PhysicsModelState` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `PhysicsModelState` against.

        Returns:
            `True` if the `PhysicsModelState` is valid for the given model,
            `False` otherwise.
        """

        shape = self.joint_positions.shape
        expected_shape = (model.dofs(),)

        if shape != expected_shape:
            return False

        shape = self.joint_velocities.shape
        expected_shape = (model.dofs(),)

        if shape != expected_shape:
            return False

        shape = self.base_position.shape
        expected_shape = (3,)

        if shape != expected_shape:
            return False

        shape = self.base_quaternion.shape
        expected_shape = (4,)

        if shape != expected_shape:
            return False

        shape = self.base_linear_velocity.shape
        expected_shape = (3,)

        if shape != expected_shape:
            return False

        shape = self.base_angular_velocity.shape
        expected_shape = (3,)

        if shape != expected_shape:
            return False

        return True


@jax_dataclasses.pytree_dataclass
class PhysicsModelInput(JaxsimDataclass):
    """
    Class storing the inputs of the physics model dynamics.

    Attributes:
        tau: The vector of joint forces.
        f_ext: The matrix of external forces applied to the links.
    """

    tau: jtp.VectorJax
    f_ext: jtp.MatrixJax

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
        joint_forces: jtp.VectorJax | None = None,
        link_forces: jtp.MatrixJax | None = None,
    ) -> PhysicsModelInput:
        """
        Build a `PhysicsModelInput` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the input.
            joint_forces: The vector of joint forces.
            link_forces: The matrix of external forces applied to the links.

        Returns:
            A `PhysicsModelInput` instance.

        Note:
            If any of the input components are not provided, they are built from the
            `JaxSimModel` and initialized to zero.
        """

        return PhysicsModelInput.build(
            joint_forces=joint_forces,
            link_forces=link_forces,
            number_of_dofs=model.dofs(),
            number_of_links=model.number_of_links(),
        )

    @staticmethod
    def build(
        joint_forces: jtp.VectorJax | None = None,
        link_forces: jtp.MatrixJax | None = None,
        number_of_dofs: jtp.Int | None = None,
        number_of_links: jtp.Int | None = None,
    ) -> PhysicsModelInput:
        """
        Build a `PhysicsModelInput`.

        Args:
            joint_forces: The vector of joint forces.
            link_forces: The matrix of external forces applied to the links.
            number_of_dofs: The number of degrees of freedom of the model.
            number_of_links: The number of links of the model.

        Returns:
            A `PhysicsModelInput` instance.
        """

        joint_forces = (
            joint_forces if joint_forces is not None else jnp.zeros(number_of_dofs)
        )

        link_forces = (
            link_forces
            if link_forces is not None
            else jnp.zeros(shape=(number_of_links, 6))
        )

        return PhysicsModelInput(
            tau=jnp.array(joint_forces, dtype=float),
            f_ext=jnp.array(link_forces, dtype=float),
        )

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> PhysicsModelInput:
        """
        Build a `PhysicsModelInput` with all components initialized to zero.

        Args:
            model: The `JaxSimModel` associated with the input.

        Returns:
            A `PhysicsModelInput` instance.
        """

        return PhysicsModelInput.build_from_jaxsim_model(model=model)

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `PhysicsModelInput` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `PhysicsModelInput` against.

        Returns:
            `True` if the `PhysicsModelInput` is valid for the given model,
            `False` otherwise.
        """

        shape = self.tau.shape
        expected_shape = (model.dofs(),)

        if shape != expected_shape:
            return False

        shape = self.f_ext.shape
        expected_shape = (model.number_of_links(), 6)

        if shape != expected_shape:
            return False

        return True
