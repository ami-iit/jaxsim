import dataclasses
from typing import Any, Tuple

import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.parsers
import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class Joint(JaxsimDataclass):
    """
    High-level class to operate on a single joint of a simulated model.
    """

    joint_description: Static[jaxsim.parsers.descriptions.JointDescription]

    _parent_model: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def parent_model(self) -> "jaxsim.high_level.model.Model":
        return self._parent_model

    def valid(self) -> bool:
        return self.parent_model is not None

    def index(self) -> int:
        return self.joint_description.index

    def dofs(self) -> int:
        return 1

    def name(self) -> str:
        return self.joint_description.name

    def position(self, dof: int = 0) -> float:
        return self.parent_model.joint_positions(joint_names=[self.name()])[dof]

    def velocity(self, dof: int = 0) -> float:
        return self.parent_model.joint_velocities(joint_names=[self.name()])[dof]

    def acceleration(self, dof: int = 0) -> float:
        return self.parent_model.joint_accelerations(joint_names=[self.name()])[dof]

    def force(self, dof: int = 0) -> float:
        return self.parent_model.joint_generalized_forces(joint_names=[self.name()])[
            dof
        ]

    def position_limit(self, dof: int = 0) -> Tuple[float, float]:
        if dof != 0:
            msg = "Only joints with 1 DoF are currently supported"
            raise ValueError(msg)

        return self.joint_description.position_limit

    # =================
    # Multi-DoF methods
    # =================

    def joint_position(self) -> jtp.Vector:
        return self.parent_model.joint_positions(joint_names=[self.name()])

    def joint_velocity(self) -> jtp.Vector:
        return self.parent_model.joint_velocities(joint_names=[self.name()])

    def joint_acceleration(self) -> jtp.Vector:
        return self.parent_model.joint_accelerations(joint_names=[self.name()])

    def joint_force(self) -> jtp.Vector:
        return self.parent_model.joint_generalized_forces(joint_names=[self.name()])
