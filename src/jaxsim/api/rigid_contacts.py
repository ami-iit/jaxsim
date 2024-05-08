from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.math import Skew
from jaxsim.terrain import FlatTerrain, Terrain

from . import link
from .common import VelRepr
from .contact import ContactModel, ContactParams, ContactsState
from .data import JaxSimModelData
from .model import (
    JaxSimModel,
    free_floating_bias_forces,
    free_floating_mass_matrix,
    link_bias_accelerations,
)


@jax_dataclasses.pytree_dataclass
class RigidContactsParams(ContactParams):
    """
    Create a RigidContactsParams instance with specified parameters.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, *args, **kwargs) -> RigidContactsParams:
        return RigidContactsParams()

    @staticmethod
    def build_default_from_jaxsim_model(
        model: JaxSimModel,
        *args,
        **kwargs,
    ) -> RigidContactsParams:
        """
        Create a RigidContactsParams instance with good default parameters.

        Args:
            model: The target model.

        Returns:
            A `RigidContactsParams` instance with the specified parameters.
        """
        return RigidContactsParams.build()


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    parameters: RigidContactsParams = dataclasses.field(
        default_factory=RigidContactsParams
    )

    terrain: Terrain = dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        model: JaxSimModel,
        data: JaxSimModelData,
        tau: jtp.Vector | None = None,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces and material deformation rate.

        Args:
            position: The position of the collidable point.
            velocity: The linear velocity of the collidable point.
            model: The model to consider.
            data: The data of the considered model.
            tau: The joint torques.

        Returns:
            A tuple containing the contact force and material deformation rate.
        """

        W_p_Ci, CW_vl_WC = position, velocity

        S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T
        h = free_floating_bias_forces(model=model, data=data)

        M = free_floating_mass_matrix(model=model, data=data)
        M_inv = jnp.linalg.inv(M)

        J̇ν = link_bias_accelerations(model=model, data=data)

        τ = tau if tau is not None else jnp.zeros(model.dofs())

        def process_point_dynamics(
            body_pos_vel: jtp.Vector,
        ) -> tuple[jtp.Vector, jtp.Vector]:
            """
            Process the dynamics of a single point.

            Args:
                body_pos_vel: The body, position, and velocity of the considered point.
                model: The model to consider.
                data: The data of the considered model.

            Returns:
                The contact force acting on the point.
            """
            body, W_p_Ci, CW_vl_WC_i = body_pos_vel

            B_Jh = link.jacobian(
                model=model,
                data=data,
                link_index=body,
                output_vel_repr=VelRepr.Body,
            )

            C_Xf_B = jnp.vstack(
                [
                    jnp.block([jnp.eye(3), Skew.wedge(W_p_Ci)]),
                    jnp.block([jnp.zeros(shape=(3, 3)), jnp.eye(3)]),
                ]
            )

            JC_i = C_Xf_B @ B_Jh

            px, py, pz = W_p_Ci
            active_contact = pz < self.terrain.height(x=px, y=py)

            f_i = -jnp.linalg.inv(JC_i @ M_inv @ JC_i.T) @ (
                J̇ν[body]
                + JC_i @ M_inv @ (S @ τ - h)
                + 30 * jnp.array([0, 0, W_p_Ci[2], 0, 0, 0])
                + 40 * jnp.array([0, 0, CW_vl_WC_i[2], 0, 0, 0])
            )

            return jax.lax.cond(
                active_contact,
                true_fun=lambda _: f_i,
                false_fun=lambda _: jnp.zeros(shape=(6,)),
                operand=None,
            )

        body_pos_vel = (
            jnp.array(model.kin_dyn_parameters.contact_parameters.body),
            W_p_Ci,
            CW_vl_WC,
        )

        # with jax.disable_jit(True):
        f = jax.vmap(process_point_dynamics)(
            body_pos_vel,
        )

        return f, jnp.zeros(shape=(model.dofs(), 6))


@jax_dataclasses.pytree_dataclass
class RigidContactsState(ContactsState):
    """
    Class storing the state of the rigid contacts model.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, **kwargs) -> RigidContactsState:
        return RigidContactsState()

    @staticmethod
    def build_from_jaxsim_model(
        model: JaxSimModel,
        **kwargs,
    ) -> RigidContactsState:
        """
        Create a RigidContactsState instance with good default parameters.

        Args:
            model: The target model.

        Returns:
            A `RigidContactsState` instance with the specified parameters.
        """
        return RigidContactsState.build()

    @staticmethod
    def zero(model: JaxSimModel) -> RigidContactsState:
        return RigidContactsState.build_from_jaxsim_model(model=model)
