from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

import jaxsim.typing as jtp
from jaxsim.math import Adjoint


@dataclasses.dataclass(eq=False, unsafe_hash=False)
class LinkDescription:
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

    name: str
    mass: float = dataclasses.field(repr=False)
    _inertia: tuple[float] = dataclasses.field(repr=False)
    index: int | None = None
    parent_name: str | None = dataclasses.field(default=None, repr=False)
    _pose: tuple[float] = dataclasses.field(
        default=tuple(np.eye(4).tolist()), repr=False
    )

    children: tuple[LinkDescription] = dataclasses.field(
        default_factory=list, repr=False
    )

    @property
    def inertia(self) -> npt.NDArray:
        """
        Get the inertia tensor of the link.

        Returns:
            npt.NDArray: The inertia tensor of the link.
        """
        return np.array(self._inertia)

    @inertia.setter
    def inertia(self, inertia: npt.NDArray) -> None:
        """
        Set the inertia tensor of the link.

        Args:
            inertia: The inertia tensor of the link.
        """
        self._inertia = tuple(inertia.tolist())

    @property
    def pose(self) -> npt.NDArray:
        """
        Get the pose transformation matrix of the link.

        Returns:
            npt.NDArray: The pose transformation matrix of the link.
        """
        return np.array(self._pose)

    @pose.setter
    def pose(self, pose: npt.NDArray) -> None:
        """
        Set the pose transformation matrix of the link.

        Args:
            pose: The pose transformation matrix of the link.
        """
        self._pose = tuple(pose.tolist())

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

        # Get the 6D inertia of the link to remove.
        I_removed = link.inertia

        # Create the SE3 object. Note the inverse.
        r_X_l = np.array(
            Adjoint.from_transform(transform=lumped_H_removed, inverse=True)
        )

        # Move the inertia
        I_removed_in_lumped_frame = r_X_l.transpose() @ I_removed @ r_X_l

        # Create the new combined link
        lumped_link = dataclasses.replace(
            self,
            mass=float(self.mass + link.mass),
            _inertia=tuple((self.inertia + I_removed_in_lumped_frame).tolist()),
        )

        return lumped_link
