import dataclasses
from typing import Dict, List, Union

import jax.numpy as jnp
import jax_dataclasses
import numpy as np
from jax_dataclasses import pytree_dataclass, static_field

import jaxsim.parsers
import jaxsim.physics
import jaxsim.typing as jtp
from jaxsim.math.plucker import Plucker
from jaxsim.parsers.descriptions import JointDescriptor, JointType
from jaxsim.physics import default_gravity
from jaxsim.utils import tracing

from .ground_contact import GroundContact
from .physics_model_state import PhysicsModelState


@pytree_dataclass
class PhysicsModel:

    NB: int = static_field()
    initial_state: PhysicsModelState = jax_dataclasses.field(default=None)
    gravity: jtp.Vector = dataclasses.field(
        default_factory=lambda: jnp.hstack(
            [np.zeros(3), jaxsim.physics.default_gravity()]
        )
    )
    is_floating_base: bool = static_field(default_factory=lambda: False)
    gc: GroundContact = dataclasses.field(default_factory=lambda: GroundContact())
    description: jaxsim.parsers.descriptions.model.ModelDescription = static_field(
        default=None
    )

    _parent_array_dict: Dict[int, int] = static_field(default_factory=dict)
    _jtype_dict: Dict[int, Union[JointType, JointDescriptor]] = static_field(
        default_factory=dict
    )
    _tree_transforms_dict: Dict[int, jtp.Matrix] = dataclasses.field(
        default_factory=dict
    )
    _link_inertias_dict: Dict[int, jtp.Matrix] = dataclasses.field(default_factory=dict)

    def __post_init__(self):

        if self.initial_state is None:
            initial_state = PhysicsModelState.zero(physics_model=self)
            object.__setattr__(self, "initial_state", initial_state)

    @staticmethod
    def build_from(
        model_description: jaxsim.parsers.descriptions.model.ModelDescription,
        gravity: jtp.Vector = default_gravity(),
    ):

        if gravity.size != 3:
            raise ValueError(gravity.size)

        # Currently, we assume that the link frame matches the frame of its parent joint
        for l in model_description:
            if not jnp.allclose(l.pose, jnp.eye(4)):
                raise ValueError(f"Link '{l.name}' has unsupported pose:\n{l.pose}")

        # ===================================
        # Initialize physics model parameters
        # ===================================

        # Get the number of bodies, including the base link
        num_of_bodies = len(model_description)

        # Build the parent array λ of the floating-base model.
        # Note: the parent of the base link is not set since it's not defined.
        parent_array_dict = {
            link.index: link.parent.index
            for link in model_description
            if link.parent is not None
        }

        # Get the 6D inertias of all links
        link_spatial_inertias_dict = {
            link.index: link.inertia for link in iter(model_description)
        }

        # Dict from the joint index to its type.
        # Note: the joint index is equal to its child link index.
        joint_types_dict = {
            joint.index: joint.jtype for joint in model_description.joints
        }

        # Transform between model's root and model's base link
        # (this is just the pose of the base link in the SDF description)
        base_link = model_description.links_dict[model_description.link_names()[0]]
        R_H_B = model_description.transform(name=base_link.name)
        B_H_R = np.linalg.inv(R_H_B)
        tree_transform_0 = Plucker.from_transform(transform=B_H_R)

        # Compute the tree transforms: pre(i)_X_λ(i).
        # Given a joint 'i', it is the coordinate transform between its predecessor
        # frame [pre(i)] and the frame of its parent link [λ(i)].
        tree_transforms_dict = {
            0: tree_transform_0,
            **{
                j.index: Plucker.from_transform(
                    transform=model_description.relative_transform(
                        relative_to=j.name, name=j.parent.name
                    )
                )
                for j in model_description.joints
            },
        }

        # =======================
        # Build the initial state
        # =======================

        # Initial joint positions
        q0 = jnp.array(
            [
                model_description.joints_dict[j.name].initial_position
                for j in model_description.joints
            ]
        )

        # Build the initial state
        initial_state = PhysicsModelState(
            joint_positions=q0,
            joint_velocities=jnp.zeros_like(q0),
            base_position=model_description.root_pose.root_position,
            base_quaternion=model_description.root_pose.root_quaternion,
        )

        # =======================
        # Build the physics model
        # =======================

        # Initialize the model
        physics_model = PhysicsModel(
            NB=num_of_bodies,
            initial_state=initial_state,
            _parent_array_dict=parent_array_dict,
            _jtype_dict=joint_types_dict,
            _tree_transforms_dict=tree_transforms_dict,
            _link_inertias_dict=link_spatial_inertias_dict,
            gravity=jnp.hstack([np.zeros(3), gravity.squeeze()]),
            is_floating_base=True,
            gc=GroundContact.build_from(model_description=model_description),
            description=(model_description),
        )

        # Floating-base models
        if not model_description.fixed_base:
            return physics_model

        # Fixed-base models
        with jax_dataclasses.copy_and_mutate(physics_model) as physics_model_fixed:
            physics_model_fixed.is_floating_base = False

        return physics_model_fixed

    def dofs(self) -> int:

        return len(list(self._jtype_dict.keys()))

    @property
    def parent(self) -> jtp.Vector:
        return self.parent_array()

    def parent_array(self) -> jtp.Vector:
        """Returns λ(i)"""

        return jnp.array([-1] + list(self._parent_array_dict.values()))

    def support_body_array(self, body_index: int) -> jtp.Vector:
        """Returns κ(i)"""

        kappa: List[int] = [body_index]

        if body_index == 0:
            return np.array(kappa)

        while True:

            i = self._parent_array_dict[kappa[-1]]

            if i == 0:
                break

            kappa.append(i)

        kappa.append(0)
        return np.array(list(reversed(kappa)), dtype=int)

    @property
    def tree_transforms(self) -> jtp.Array:

        X_tree = jnp.array(
            [
                self._tree_transforms_dict.get(idx, jnp.eye(6))
                for idx in np.arange(start=0, stop=self.NB)
            ]
        )

        return X_tree

    @property
    def spatial_inertias(self) -> jtp.Array:

        M_links = jnp.array(
            [
                self._link_inertias_dict.get(idx, jnp.zeros(6))
                for idx in np.arange(start=0, stop=self.NB)
            ]
        )

        return M_links

    def jtype(self, joint_index: int) -> JointType:

        if joint_index == 0 or joint_index >= self.NB:
            raise ValueError(joint_index)

        return self._jtype_dict[joint_index]

    def joint_transforms(self, q: jtp.Vector) -> jtp.Array:

        from jaxsim.math.joint import jcalc

        if not tracing(q) and q.shape[0] != self.dofs():
            raise ValueError(q.shape)

        Xj = jnp.stack(
            [jnp.zeros(shape=(6, 6))]
            + [
                jcalc(jtyp=self.jtype(index + 1), q=joint_position)[0]
                for index, joint_position in enumerate(q)
            ]
        )

        return Xj

    def motion_subspaces(self, q: jtp.Vector) -> jtp.Array:

        from jaxsim.math.joint import jcalc

        if not tracing(q) and q.shape[0] != self.dofs():
            raise ValueError(q.shape)

        SS = jnp.stack(
            [jnp.vstack(jnp.zeros(6))]
            + [
                jcalc(jtyp=self.jtype(index + 1), q=joint_position)[1]
                for index, joint_position in enumerate(q)
            ]
        )

        return SS

    def __eq__(self, other: "PhysicsModel") -> bool:

        same = True
        same = same and self.NB == other.NB
        same = same and np.allclose(self.gravity, other.gravity)

        return same

    def __hash__(self):

        return hash(self.__repr__())
