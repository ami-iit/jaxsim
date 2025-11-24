from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from jaxsim import logging
from jaxsim.logging import jaxsim_warn

from ..kinematic_graph import KinematicGraph, KinematicGraphTransforms, RootPose
from .joint import JointDescription
from .link import LinkDescription


@dataclasses.dataclass(frozen=True, eq=False, unsafe_hash=False)
class ModelDescription(KinematicGraph):
    """
    Intermediate representation representing the kinematic graph of a robot model.

    Attributes:
        name: The name of the model.
        fixed_base: Whether the model is either fixed-base or floating-base.
        collision_shapes: List of collision shapes associated with the model.
    """

    name: str = None

    fixed_base: bool = True

    collision_shapes: tuple = dataclasses.field(default_factory=list, repr=False)

    @staticmethod
    def build_model_from(
        name: str,
        links: list[LinkDescription],
        joints: list[JointDescription],
        frames: list[LinkDescription] | None = None,
        collisions: tuple = (),
        fixed_base: bool = False,
        base_link_name: str | None = None,
        considered_joints: Sequence[str] | None = None,
        model_pose: RootPose = RootPose(),
    ) -> ModelDescription:
        """
        Build a model description from provided components.

        Args:
            name: The name of the model.
            links: List of link descriptions.
            joints: List of joint descriptions.
            frames: List of frame descriptions.
            collisions: List of collision shapes associated with the model.
            fixed_base: Indicates whether the model has a fixed base.
            base_link_name: Name of the base link (i.e. the root of the kinematic tree).
            considered_joints: List of joint names to consider (by default all joints).
            model_pose: Pose of the model's root (by default an identity transform).

        Returns:
            A ModelDescription instance representing the model.
        """

        # Create the full kinematic graph.
        kinematic_graph = KinematicGraph.build_from(
            links=links,
            joints=joints,
            frames=frames,
            root_link_name=base_link_name,
            root_pose=model_pose,
        )

        # Reduce the graph if needed.
        if considered_joints is not None:
            kinematic_graph = kinematic_graph.reduce(
                considered_joints=considered_joints
            )

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=kinematic_graph)

        # Container of the final model's collision shapes.
        final_collisions: list = []

        # Move and express the collision shapes of removed links to the resulting
        # lumped link that replace the combination of the removed link and its parent.
        for collision_shape in collisions:

            # Get the parent link of the collision shape.
            # Note that this link could have been lumped and we need to find the
            # link in which it was lumped into.
            parent_link_of_shape = collision_shape.parent_link

            # If it is part of the (reduced) graph, add it as it is...
            if parent_link_of_shape in kinematic_graph.link_names():
                final_collisions.append(collision_shape)
                continue

            # ... otherwise look for the frame
            if parent_link_of_shape not in kinematic_graph.frame_names():
                msg = "Parent frame '{}' of collision shape not found, ignoring shape"
                logging.info(msg.format(parent_link_of_shape))
                continue

            # Find the link that is part of the (reduced) model in which the
            # collision shape's parent was lumped into.
            real_parent_link_name = kinematic_graph.frames_dict[
                parent_link_of_shape
            ].parent_name

            # Get the transform from the real parent link to the removed link
            # that still exists as a frame.
            parent_H_frame = fk.relative_transform(
                relative_to=real_parent_link_name,
                name=parent_link_of_shape,
            )

            # Transform the collision shape's pose to the new parent link frame.
            # The collision shape was defined w.r.t. the removed link (now a frame).
            # Now we need to express it w.r.t. the link that absorbed the removed link.
            # Compose the transforms: parent_H_shape = parent_H_frame @ frame_H_shape
            parent_H_shape = parent_H_frame @ collision_shape.transform

            # Create a new collision shape with updated pose and parent link
            new_collision_shape = dataclasses.replace(
                collision_shape,
                transform=parent_H_shape,
                parent_link=real_parent_link_name,
            )

            final_collisions.append(new_collision_shape)

        # Build the model
        model = ModelDescription(
            name=name,
            root_pose=kinematic_graph.root_pose,
            fixed_base=fixed_base,
            collision_shapes=tuple(final_collisions),
            root=kinematic_graph.root,
            joints=kinematic_graph.joints,
            frames=kinematic_graph.frames,
            _joints_removed=kinematic_graph.joints_removed,
        )

        # Check that the root link of kinematic graph is the desired base link.
        assert kinematic_graph.root.name == base_link_name, kinematic_graph.root.name

        return model

    def reduce(self, considered_joints: Sequence[str]) -> ModelDescription:
        """
        Reduce the model by removing specified joints.

        Args:
            considered_joints: Sequence of joint names to consider.

        Returns:
            A `ModelDescription` instance that only includes the considered joints.
        """

        jaxsim_warn(
            "The joint order in the model description is not preserved when reducing "
            "the model. Consider using the `names_to_indices` method to get the correct "
            "order of the joints, or use the `joint_names()` method to inspect the internal joint ordering."
        )

        if len(set(considered_joints) - set(self.joint_names())) != 0:
            extra_joints = set(considered_joints) - set(self.joint_names())
            msg = f"Found joints not part of the model: {extra_joints}"
            raise ValueError(msg)

        reduced_model_description = ModelDescription.build_model_from(
            name=self.name,
            links=list(self.links_dict.values()),
            joints=self.joints,
            frames=self.frames,
            collisions=tuple(self.collision_shapes),
            fixed_base=self.fixed_base,
            base_link_name=next(iter(self)).name,
            model_pose=self.root_pose,
            considered_joints=considered_joints,
        )

        # Include the unconnected/removed joints from the original model.
        for joint in self.joints_removed:
            reduced_model_description.joints_removed.append(joint)

        return reduced_model_description

    def __eq__(self, other: ModelDescription) -> bool:

        if not isinstance(other, ModelDescription):
            return False

        if not (
            self.name == other.name
            and self.fixed_base == other.fixed_base
            and self.root == other.root
            and self.joints == other.joints
            and self.frames == other.frames
            and self.root_pose == other.root_pose
        ):
            return False

        return True

    def __hash__(self) -> int:

        return hash(
            (
                hash(self.name),
                hash(self.fixed_base),
                hash(self.root),
                hash(tuple(self.joints)),
                hash(tuple(self.frames)),
                hash(self.root_pose),
            )
        )
