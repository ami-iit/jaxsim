from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax_dataclasses
import jaxlie
import numpy as np
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class LinkDescription(JaxsimDataclass):
    """
    In-memory description of a robot link.

    Attributes:
        name: The name of the link.
        mass: The mass of the link.
        inertia: The inertia tensor of the link.
        index: An optional index for the link (it gets automatically assigned).
        parent: The parent link of this link.
        pose: The pose transformation matrix of the link.
        children: The children links.
    """

    name: Static[str]
    mass: float = dataclasses.field(repr=False)
    inertia: jtp.Matrix = dataclasses.field(repr=False)
    index: int | None = None
    parent: LinkDescription = dataclasses.field(default=None, repr=False)
    pose: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.eye(4), repr=False)

    children: Static[tuple[LinkDescription]] = dataclasses.field(
        default_factory=list, repr=False
    )

    def __hash__(self) -> int:

        return hash(
            (
                hash(self.name),
                hash(float(self.mass)),
                hash(tuple(np.atleast_1d(self.inertia).flatten().tolist())),
                hash(int(self.index)) if self.index is not None else 0,
                hash(tuple(np.atleast_1d(self.pose).flatten().tolist())),
                hash(tuple(self.children)),
                # Here only using the name to prevent circular recursion:
                hash(self.parent.name) if self.parent is not None else 0,
            )
        )

    def __eq__(self, other: LinkDescription) -> bool:

        if not isinstance(other, LinkDescription):
            return False

        return hash(self) == hash(other)

    @property
    def name_and_index(self) -> str:
        """
        Get a formatted string with the link's name and index.

        Returns:
            str: The formatted string.

        """
        return f"#{self.index}_<{self.name}>"

    def lump_with(
        self, link: LinkDescription, lumped_H_removed: jtp.Matrix
    ) -> LinkDescription:
        """
        Combine the current link with another link, preserving mass and inertia.

        Args:
            link: The link to combine with.
            lumped_H_removed: The transformation matrix between the two links.

        Returns:
            The combined link.
        """

        # Get the 6D inertia of the link to remove
        I_removed = link.inertia

        # Create the SE3 object. Note the inverse.
        r_H_l = jaxlie.SE3.from_matrix(lumped_H_removed).inverse()
        r_X_l = r_H_l.adjoint()

        # Move the inertia
        I_removed_in_lumped_frame = r_X_l.transpose() @ I_removed @ r_X_l

        # Create the new combined link
        lumped_link = self.replace(
            mass=self.mass + link.mass,
            inertia=self.inertia + I_removed_in_lumped_frame,
        )

        return lumped_link
