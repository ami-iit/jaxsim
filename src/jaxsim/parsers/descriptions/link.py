import copy
import dataclasses
from typing import List

import jax.numpy as jnp
import numpy.typing as npt

from jaxsim.sixd import se3


@dataclasses.dataclass
class LinkDescription:

    name: str
    mass: float
    inertia: npt.NDArray
    index: int = None
    parent: "LinkDescription" = None
    pose: npt.NDArray = dataclasses.field(default_factory=lambda: jnp.eye(4))
    children: List["LinkDescription"] = dataclasses.field(default_factory=list)

    def __hash__(self) -> int:

        return hash(self.__repr__())

    @property
    def name_and_index(self) -> str:

        return f"#{self.index}_<{self.name}>"

    def lump_with(
        self, link: "LinkDescription", lumped_H_removed: npt.NDArray
    ) -> "LinkDescription":

        from jaxsim.parsers.sdf.utils import flip_velocity_serialization

        # Convert ang-lin serialization used in physics algorithms to lin-ang
        I_removed = flip_velocity_serialization(link.inertia)

        # Create the SE3 object. Note the inverse.
        H = se3.SE3.from_matrix(lumped_H_removed).inverse()

        # Move the inertia
        I_removed_in_lumped_frame = H.adjoint().transpose() @ I_removed @ H.adjoint()

        # Switch back to ang-lin serialization
        I_removed_in_lumped_frame_anglin = flip_velocity_serialization(
            I_removed_in_lumped_frame
        )

        lumped_link = copy.deepcopy(self)
        lumped_link.mass = self.mass + link.mass
        lumped_link.inertia = self.inertia + I_removed_in_lumped_frame_anglin

        return lumped_link
