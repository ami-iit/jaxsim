from __future__ import annotations

import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.ode_data import PhysicsModelInput, PhysicsModelState, SoftContactsState
from jaxsim.utils import JaxsimDataclass


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

    def valid(self, model: js.model.MujocoModelHelper) -> bool:
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
        soft_contacts: The state of the soft-contacts model.
    """

    physics_model: PhysicsModelState
    soft_contacts: SoftContactsState

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
        """

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
            soft_contacts_state=SoftContactsState.build_from_jaxsim_model(
                model=model,
                tangential_deformation=tangential_deformation,
            ),
        )

    @staticmethod
    def build(
        physics_model_state: PhysicsModelState | None = None,
        soft_contacts_state: SoftContactsState | None = None,
        model: js.model.JaxSimModel | None = None,
    ) -> ODEState:
        """
        Build an `ODEState` from a `PhysicsModelState` and a `SoftContactsState`.

        Args:
            physics_model_state: The state of the physics model.
            soft_contacts_state: The state of the soft-contacts model.
            model: The `JaxSimModel` associated with the ODE state.

        Returns:
            A `ODEState` instance.
        """

        physics_model_state = (
            physics_model_state
            if physics_model_state is not None
            else PhysicsModelState.zero(model=model)
        )

        soft_contacts_state = (
            soft_contacts_state
            if soft_contacts_state is not None
            else SoftContactsState.zero(model=model)
        )

        return ODEState(
            physics_model=physics_model_state, soft_contacts=soft_contacts_state
        )

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

        # assert model_state.valid(physics_model)
        return model_state

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `ODEState` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `ODEState` against.

        Returns:
            `True` if the ODE state is valid for the given model, `False` otherwise.
        """

        return self.physics_model.valid(model=model) and self.soft_contacts.valid(
            model=model
        )
