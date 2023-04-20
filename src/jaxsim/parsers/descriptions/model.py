import dataclasses
import itertools
from typing import List

import numpy.typing as npt

import jaxsim.logging as logging

from ..kinematic_graph import KinematicGraph, RootPose
from .collision import CollidablePoint, CollisionShape
from .joint import JointDescription
from .link import LinkDescription


@dataclasses.dataclass(frozen=True)
class ModelDescription(KinematicGraph):
    name: str = None
    fixed_base: bool = True
    collision_shapes: List[CollisionShape] = dataclasses.field(default_factory=list)

    @staticmethod
    def build_model_from(
        name: str,
        links: List[LinkDescription],
        joints: List[JointDescription],
        collisions: List[CollisionShape] = (),
        fixed_base: bool = False,
        base_link_name: str = None,
        considered_joints: List[str] = None,
        model_pose: RootPose = RootPose(),
    ) -> "ModelDescription":
        # Create the full kinematic graph
        kinematic_graph = KinematicGraph.build_from(
            links=links,
            joints=joints,
            root_link_name=base_link_name,
            root_pose=model_pose,
        )

        # Reduce the graph if needed
        if considered_joints is not None:
            kinematic_graph = kinematic_graph.reduce(
                considered_joints=considered_joints
            )

        # Store here the final model collisions
        final_collisions: List[CollisionShape] = []

        # Move and express the collision shapes of the removed link to the lumped link
        for collision_shape in collisions:
            # Get all the collidable points of the shape
            coll_points = [cp for cp in collision_shape.collidable_points]

            # Assume they have an unique parent link
            if not len(set({cp.parent_link.name for cp in coll_points})) == 1:
                msg = "Collision shape not currently supported (multiple parent links)"
                raise RuntimeError(msg)

            # Get the parent link of the collision shape.
            # Note that this link could have been lumped and we need to find the
            # link in which it was lumped into.
            parent_link_of_shape = collision_shape.collidable_points[0].parent_link

            # If it is part of the (reduced) graph, add it as it is...
            if parent_link_of_shape.name in kinematic_graph.link_names():
                final_collisions.append(collision_shape)
                continue

            # ... otherwise look for the frame
            if parent_link_of_shape.name not in kinematic_graph.frame_names():
                msg = "Parent frame '{}' of collision shape not found, ignoring shape"
                logging.info(msg.format(parent_link_of_shape.name))
                continue

            # Create a new collision shape
            new_collision_shape = CollisionShape(collidable_points=[])
            final_collisions.append(new_collision_shape)

            # If the frame was found, update the collidable points' pose and add them
            #  to the new collision shape
            for cp in collision_shape.collidable_points:
                # Find the link that is part of the (reduced) model in which the
                # collision shape's parent was lumped into
                real_parent_link_of_shape = kinematic_graph.frames_dict[
                    parent_link_of_shape.name
                ].parent

                # Change the link associated to the collidable point, updating their
                # relative pose
                moved_cp = cp.change_link(
                    new_link=real_parent_link_of_shape,
                    new_H_old=kinematic_graph.relative_transform(
                        relative_to=real_parent_link_of_shape.name,
                        name=cp.parent_link.name,
                    ),
                )

                # Store the updated collision
                new_collision_shape.collidable_points.append(moved_cp)

        # Build the model
        model = ModelDescription(
            name=name,
            root_pose=kinematic_graph.root_pose,
            fixed_base=fixed_base,
            collision_shapes=final_collisions,
            root=kinematic_graph.root,
            joints=kinematic_graph.joints,
            frames=kinematic_graph.frames,
        )
        assert kinematic_graph.root.name == base_link_name, kinematic_graph.root.name

        return model

    def reduce(self, considered_joints: List[str]) -> "ModelDescription":
        msg = "The model reduction logic assumes that removed joints have zero angles"
        logging.info(msg=msg)

        if len(set(considered_joints) - set(self.joint_names())) != 0:
            extra_joints = set(considered_joints) - set(self.joint_names())
            msg = f"Found joints not part of the model: {extra_joints}"
            raise ValueError(msg)

        return ModelDescription.build_model_from(
            name=self.name,
            links=list(self.links_dict.values()),
            joints=self.joints,
            collisions=self.collision_shapes,
            fixed_base=self.fixed_base,
            base_link_name=list(iter(self))[0].name,
            model_pose=self.root_pose,
            considered_joints=considered_joints,
        )

    def update_collision_shape_of_link(self, link_name: str, enabled: bool) -> None:
        if link_name not in self.link_names():
            raise ValueError(link_name)

        for point in self.collision_shape_of_link(
            link_name=link_name
        ).collidable_points:
            point.enabled = enabled

    def collision_shape_of_link(self, link_name: str) -> CollisionShape:
        if link_name not in self.link_names():
            raise ValueError(link_name)

        return CollisionShape(
            collidable_points=[
                point
                for shape in self.collision_shapes
                for point in shape.collidable_points
                if point.parent_link.name == link_name
            ]
        )

    def all_enabled_collidable_points(self) -> List[CollidablePoint]:
        # Get iterator of all collidable points
        all_collidable_points = itertools.chain.from_iterable(
            [shape.collidable_points for shape in self.collision_shapes]
        )

        # Return enabled collidable points
        return [cp for cp in all_collidable_points if cp.enabled]
