import dataclasses
from typing import List

import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.sixd import se3
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class LinkDescription(JaxsimDataclass):
    """
    In-memory description of a robot link.

    Attributes:
        name (str): The name of the link.
        mass (float): The mass of the link.
        inertia (jtp.Matrix): The inertia matrix of the link.
        index (Optional[int]): An optional index for the link.
        parent (Optional[LinkDescription]): The parent link of this link.
        pose (jtp.Matrix): The pose transformation matrix of the link.
        children (List[LinkDescription]): List of child links.
    """

    name: Static[str]
    mass: float
    inertia: jtp.Matrix
    index: int | None = None
    parent: Static["LinkDescription"] = dataclasses.field(default=None, repr=False)
    pose: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.eye(4), repr=False)
    children: Static[List["LinkDescription"]] = dataclasses.field(
        default_factory=list, repr=False
    )

    def __hash__(self) -> int:
        return hash(self.__repr__())

    @property
    def name_and_index(self) -> str:
        """
        Get a formatted string with the link's name and index.

        Returns:
            str: The formatted string.

        """
        return f"#{self.index}_<{self.name}>"

    def lump_with(
        self, link: "LinkDescription", lumped_H_removed: jtp.Matrix
    ) -> "LinkDescription":
        """
        Combine the current link with another link, preserving mass and inertia.

        Args:
            link (LinkDescription): The link to combine with.
            lumped_H_removed (jtp.Matrix): The transformation matrix between the two links.

        Returns:
            LinkDescription: The combined link.

        """
        # Get the 6D inertia of the link to remove
        I_removed = link.inertia

        # Create the SE3 object. Note the inverse.
        r_H_l = se3.SE3.from_matrix(lumped_H_removed).inverse()
        r_X_l = r_H_l.adjoint()

        # Move the inertia
        I_removed_in_lumped_frame = r_X_l.transpose() @ I_removed @ r_X_l

        # Create the new combined link
        lumped_link = self.replace(
            mass=self.mass + link.mass,
            inertia=self.inertia + I_removed_in_lumped_frame,
        )

        return lumped_link
