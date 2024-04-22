from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.terrain import FlatTerrain, Terrain


@jax_dataclasses.pytree_dataclass
class RigidContactsParams(js.contact.ContactParams):
    """
    Create a SoftContactsParams instance with specified parameters.
    """

    def build(self) -> RigidContactsParams:
        return RigidContactsParams()


@jax_dataclasses.pytree_dataclass
class RigidContacts(js.contact.ContactModel):
    """Soft contacts model."""

    parameters: RigidContactsParams = dataclasses.field(
        default_factory=RigidContactsParams
    )

    terrain: Terrain = dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        jdot_nu: jtp.Vector,
        j: jtp.Vector,
        nu: jtp.Vector,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces and material deformation rate.

        Args:
            position: The position of the collidable point.
            velocity: The linear velocity of the collidable point.

        Returns:
            A tuple containing the contact force and material deformation rate.
        """

        # Compute the contact force.
        contact_force = jnp.zeros_like(j)
