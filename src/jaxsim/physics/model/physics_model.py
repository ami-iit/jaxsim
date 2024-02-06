import dataclasses
from typing import Dict, Union

import jax.lax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
from jax_dataclasses import Static

import jaxsim.parsers
import jaxsim.physics
import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import JointDescriptor, JointType
from jaxsim.physics import default_gravity
from jaxsim.sixd import se3
from jaxsim.utils import JaxsimDataclass, not_tracing

from .ground_contact import GroundContact
from .physics_model_state import PhysicsModelState


@jax_dataclasses.pytree_dataclass
class PhysicsModel(JaxsimDataclass):
    """
    A read-only class to store all the information necessary to run RBDAs on a model.

    This class contains information about the physics model, including the number of bodies, initial state, gravity,
    floating base configuration, ground contact points, and more.

    Attributes:
        NB (Static[int]): The number of bodies in the physics model.
        initial_state (PhysicsModelState): The initial state of the physics model (default: None).
        gravity (jtp.Vector): The gravity vector (default: [0, 0, 0, 0, 0, 0]).
        is_floating_base (Static[bool]): A flag indicating whether the model has a floating base (default: False).
        gc (GroundContact): The ground contact points of the model (default: empty GroundContact instance).
        description (Static[jaxsim.parsers.descriptions.model.ModelDescription]): A description of the model (default: None).
    """

    NB: Static[int]
    initial_state: PhysicsModelState = dataclasses.field(default=None)
    gravity: jtp.Vector = dataclasses.field(
        default_factory=lambda: jnp.hstack(
            [np.zeros(3), jaxsim.physics.default_gravity()]
        )
    )
    is_floating_base: Static[bool] = dataclasses.field(default=False)
    gc: GroundContact = dataclasses.field(default_factory=lambda: GroundContact())
    description: Static[jaxsim.parsers.descriptions.model.ModelDescription] = (
        dataclasses.field(default=None)
    )

    _parent_array_dict: Static[Dict[int, int]] = dataclasses.field(default_factory=dict)
    _jtype_dict: Static[Dict[int, Union[JointType, JointDescriptor]]] = (
        dataclasses.field(default_factory=dict)
    )
    _tree_transforms_dict: Dict[int, jtp.Matrix] = dataclasses.field(
        default_factory=dict
    )
    _link_inertias_dict: Dict[int, jtp.Matrix] = dataclasses.field(default_factory=dict)

    _joint_friction_static: Dict[int, float] = dataclasses.field(default_factory=dict)
    _joint_friction_viscous: Dict[int, float] = dataclasses.field(default_factory=dict)

    _joint_limit_spring: Dict[int, float] = dataclasses.field(default_factory=dict)
    _joint_limit_damper: Dict[int, float] = dataclasses.field(default_factory=dict)

    _joint_motor_inertia: Dict[int, float] = dataclasses.field(default_factory=dict)
    _joint_motor_gear_ratio: Dict[int, float] = dataclasses.field(default_factory=dict)
    _joint_motor_viscous_friction: Dict[int, float] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.initial_state is None:
            initial_state = PhysicsModelState.zero(physics_model=self)
            object.__setattr__(self, "initial_state", initial_state)

    @staticmethod
    def build_from(
        model_description: jaxsim.parsers.descriptions.model.ModelDescription,
        gravity: jtp.Vector = default_gravity(),
    ) -> "PhysicsModel":
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

        # Dicts from the joint index to the static and viscous friction.
        # Note: the joint index is equal to its child link index.
        joint_friction_static = {
            joint.index: jnp.array(joint.friction_static, dtype=float)
            for joint in model_description.joints
        }
        joint_friction_viscous = {
            joint.index: jnp.array(joint.friction_viscous, dtype=float)
            for joint in model_description.joints
        }

        # Dicts from the joint index to the spring and damper joint limits parameters.
        # Note: the joint index is equal to its child link index.
        joint_limit_spring = {
            joint.index: jnp.array(joint.position_limit_spring, dtype=float)
            for joint in model_description.joints
        }
        joint_limit_damper = {
            joint.index: jnp.array(joint.position_limit_damper, dtype=float)
            for joint in model_description.joints
        }

        # Dicts from the joint index to the motor inertia, gear ratio and viscous friction.
        # Note: the joint index is equal to its child link index.
        joint_motor_inertia = {
            joint.index: jnp.array(joint.motor_inertia, dtype=float)
            for joint in model_description.joints
        }
        joint_motor_gear_ratio = {
            joint.index: jnp.array(joint.motor_gear_ratio, dtype=float)
            for joint in model_description.joints
        }
        joint_motor_viscous_friction = {
            joint.index: jnp.array(joint.motor_viscous_friction, dtype=float)
            for joint in model_description.joints
        }

        # Transform between model's root and model's base link
        # (this is just the pose of the base link in the SDF description)
        base_link = model_description.links_dict[model_description.link_names()[0]]
        R_H_B = model_description.transform(name=base_link.name)
        tree_transform_0 = se3.SE3.from_matrix(matrix=R_H_B).adjoint()

        # Helper to compute the transform pre(i)_H_λ(i).
        # Given a joint 'i', it is the coordinate transform between its predecessor
        # frame [pre(i)] and the frame of its parent link [λ(i)].
        prei_H_λi = lambda j: model_description.relative_transform(
            relative_to=j.name, name=j.parent.name
        )

        # Compute the tree transforms: pre(i)_X_λ(i).
        # Given a joint 'i', it is the coordinate transform between its predecessor
        # frame [pre(i)] and the frame of its parent link [λ(i)].
        tree_transforms_dict = {
            0: tree_transform_0,
            **{
                j.index: se3.SE3.from_matrix(matrix=prei_H_λi(j)).adjoint()
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
            _joint_friction_static=joint_friction_static,
            _joint_friction_viscous=joint_friction_viscous,
            _joint_limit_spring=joint_limit_spring,
            _joint_limit_damper=joint_limit_damper,
            _joint_motor_gear_ratio=joint_motor_gear_ratio,
            _joint_motor_inertia=joint_motor_inertia,
            _joint_motor_viscous_friction=joint_motor_viscous_friction,
            gravity=jnp.hstack([gravity.squeeze(), np.zeros(3)]),
            is_floating_base=True,
            gc=GroundContact.build_from(model_description=model_description),
            description=model_description,
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

    def set_gravity(self, gravity: jtp.Vector) -> None:
        gravity = gravity.squeeze()

        if gravity.size == 3:
            self.gravity = jnp.hstack([gravity, 0, 0, 0])

        elif gravity.size == 6:
            self.gravity = gravity

        else:
            raise ValueError(gravity.shape)

    @property
    def parent(self) -> jtp.Vector:
        return self.parent_array()

    def parent_array(self) -> jtp.Vector:
        """Returns λ(i)"""
        return jnp.array([-1] + list(self._parent_array_dict.values()), dtype=int)

    def support_body_array(self, body_index: jtp.Int) -> jtp.Vector:
        """Returns κ(i)"""

        κ_bool = self.support_body_array_bool(body_index=body_index)
        return jnp.array(jnp.where(κ_bool)[0], dtype=int)

    def support_body_array_bool(self, body_index: jtp.Int) -> jtp.Vector:
        active_link = body_index
        κ_bool = jnp.zeros(self.NB, dtype=bool)

        for i in np.flip(np.arange(start=0, stop=self.NB)):
            κ_bool, active_link = jax.lax.cond(
                pred=(i == active_link),
                false_fun=lambda: (κ_bool, active_link),
                true_fun=lambda: (
                    κ_bool.at[active_link].set(True),
                    self.parent[active_link],
                ),
            )

        return κ_bool

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

        if not_tracing(q):
            if q.shape[0] != self.dofs():
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

        if not_tracing(var=q):
            if q.shape[0] != self.dofs():
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

    def __repr__(self) -> str:
        attributes = [
            f"dofs: {self.dofs()},",
            f"links: {self.NB},",
            f"floating_base: {self.is_floating_base},",
        ]
        attributes_string = "\n    ".join(attributes)

        return f"{type(self).__name__}(\n    {attributes_string}\n)"
