import dataclasses
from pathlib import Path
from typing import Dict, List, NamedTuple, Union

import jax.numpy as jnp
import numpy as np
import pysdf
from scipy.spatial.transform.rotation import Rotation as R

from jaxsim import logging
from jaxsim.parsers import descriptions, kinematic_graph

from . import utils as utils


class SDFData(NamedTuple):

    model_name: str

    fixed_base: bool
    base_link_name: str

    link_descriptions: List[descriptions.LinkDescription]
    joint_descriptions: List[descriptions.JointDescription]
    collision_shapes: List[descriptions.CollisionShape]

    sdf_tree: pysdf.Model = None
    model_pose: kinematic_graph.RootPose = kinematic_graph.RootPose()


def extract_data_from_sdf(
    sdf: Union[Path, str],
) -> SDFData:

    if isinstance(sdf, str) and len(sdf) < 500 and Path(sdf).is_file():
        sdf = Path(sdf)

    # Get the SDF string
    sdf_string = sdf if isinstance(sdf, str) else sdf.read_text()

    # Parse the tree
    sdf_tree = pysdf.SDF.from_xml(sdf_string=sdf_string, remove_blank_text=True)

    # Detect whether the model is fixed base by checking joints with world parent exist.
    # This link is a special link used to specify that the model's base should be fixed.
    fixed_base = len([j for j in sdf_tree.model.joints if j.parent == "world"]) > 0

    # Base link of the model. We take the first link in the SDF description.
    base_link_name = sdf_tree.model.links[0].name

    # Pose of the model
    if sdf_tree.model.pose is None:
        model_pose = kinematic_graph.RootPose()

    else:
        w_H_m = utils.from_sdf_pose(pose=sdf_tree.model.pose)
        xyzw_to_wxyz = np.array([3, 0, 1, 2])
        w_quat_m = R.from_matrix(w_H_m[0:3, 0:3]).as_quat()[xyzw_to_wxyz]
        model_pose = kinematic_graph.RootPose(
            root_position=w_H_m[0:3, 3],
            root_quaternion=w_quat_m,
        )

    # ===========
    # Parse links
    # ===========

    # Parse the links (unconnected)
    links = [
        descriptions.LinkDescription(
            name=l.name,
            mass=jnp.float32(l.inertial.mass),
            inertia=utils.from_sdf_inertial(inertial=l.inertial),
            pose=utils.from_sdf_pose(pose=l.pose) if l.pose is not None else np.eye(4),
        )
        for l in sdf_tree.model.links
        if l.inertial.mass > 0
    ]

    # Create a dictionary to find easily links
    links_dict: Dict[str, descriptions.LinkDescription] = {l.name: l for l in links}

    # =========================
    # Process fixed-base models
    # =========================

    # In this case, we need to get the pose of the joint that connects the base link
    # to the world and combine their pose
    if fixed_base:

        world_link = descriptions.LinkDescription(
            name="world", mass=0, inertia=np.zeros(shape=(6, 6))
        )

        # Gather joints connecting fixed-base models to the world.
        # TODO: the pose of this joint could be expressed wrt any arbitrary frame,
        #       here we assume is expressed wrt the model. This also means that the
        #       default model pose matches the pose of the fake "world" link.
        joints_with_world_parent = [
            descriptions.JointDescription(
                name=j.name,
                parent=world_link,
                child=links_dict[j.child],
                jtype=utils.axis_to_jtype(axis=j.axis, type=j.type),
                axis=utils.from_sdf_string_list(string_list=j.axis.xyz.text)
                if j.axis is not None
                and j.axis.xyz is not None
                and j.axis.xyz.text is not None
                else None,
                pose=utils.from_sdf_pose(pose=j.pose)
                if j.pose is not None
                else np.eye(4),
            )
            for j in sdf_tree.model.joints
            if j.type == "fixed"
            and j.parent == "world"
            and j.child in links_dict.keys()
            and j.pose.relative_to == "__model__"
        ]

        logging.debug(
            f"Found joints connecting to world: {[j.name for j in joints_with_world_parent]}"
        )

        if len(joints_with_world_parent) != 1:
            msg = "Found more/less than one joint connecting a fixed-base model to the world"
            raise ValueError(msg + f": {[j.name for j in joints_with_world_parent]}")

        msg = "Combining the pose of base link '{}' with the pose of joint '{}'"
        logging.info(
            msg.format(
                joints_with_world_parent[0].child.name, joints_with_world_parent[0].name
            )
        )

        # Combine the pose of the base link (child of the found fixed joint)
        # with the pose of the fixed joint connecting with the world.
        # Note: we assume it's a fixed joint and ignore any joint angle.
        links_dict[joints_with_world_parent[0].child.name].mutable(
            validate=False
        ).pose = (
            joints_with_world_parent[0].pose
            @ links_dict[joints_with_world_parent[0].child.name].pose
        )

    # ============
    # Parse joints
    # ============

    # Check that all joint poses are expressed w.r.t. their parent link
    for j in sdf_tree.model.joints:

        if j.pose is None:
            continue

        if j.parent == "world":

            if j.pose.relative_to == "__model__":
                continue

            raise ValueError("Pose of fixed joint connecting to 'world' link not valid")

        if j.pose.relative_to != j.parent:
            msg = "Pose of joint '{}' is not expressed wrt its parent link '{}'"
            raise ValueError(msg.format(j.name, j.parent))

    # Parse the joints
    joints = [
        descriptions.JointDescription(
            name=j.name,
            parent=links_dict[j.parent],
            child=links_dict[j.child],
            jtype=utils.axis_to_jtype(axis=j.axis, type=j.type),
            axis=utils.from_sdf_string_list(j.axis.xyz.text)
            if j.axis is not None
            and j.axis.xyz is not None
            and j.axis.xyz.text is not None
            else None,
            pose=utils.from_sdf_pose(pose=j.pose) if j.pose is not None else np.eye(4),
            initial_position=0.0,
            position_limit=(
                float(j.axis.limit.lower)
                if j.axis is not None and j.axis.limit is not None
                else np.finfo(float).min,
                float(j.axis.limit.upper)
                if j.axis is not None and j.axis.limit is not None
                else np.finfo(float).max,
            ),
        )
        for j in sdf_tree.model.joints
        if j.type in {"revolute", "prismatic", "fixed"}
        and j.parent != "world"
        and j.child in links_dict.keys()
    ]

    # Create a dictionary to find the parent joint of the links
    joint_dict = {j.child.name: j.name for j in joints}

    # Check that all the link poses are expressed wrt their parent joint
    for l in sdf_tree.model.links:

        if l.name not in links_dict:
            continue

        if l.pose is None:
            continue

        if l.name == base_link_name:
            continue

        if l.name not in joint_dict:
            raise ValueError(f"Failed to find parent joint of link '{l.name}'")

        if l.pose.relative_to != joint_dict[l.name]:
            msg = "Pose of link '{}' is not expressed wrt its parent joint '{}'"
            raise ValueError(msg.format(l.name, joint_dict[l.name]))

    # ================
    # Parse collisions
    # ================

    # Initialize the collision shapes
    collisions: List[descriptions.CollisionShape] = []

    # Parse the collisions
    for link in sdf_tree.model.links:
        for collision in link.colliders:

            if collision.geometry.box.to_xml() != "<box/>":

                box_collision = utils.create_box_collision(
                    collision=collision,
                    link_description=links_dict[link.name],
                )

                collisions.append(box_collision)

            if collision.geometry.sphere.to_xml() != "<sphere/>":

                sphere_collision = utils.create_sphere_collision(
                    collision=collision,
                    link_description=links_dict[link.name],
                )

                collisions.append(sphere_collision)

    return SDFData(
        model_name=sdf_tree.model.name,
        link_descriptions=links,
        joint_descriptions=joints,
        collision_shapes=collisions,
        fixed_base=fixed_base,
        base_link_name=base_link_name,
        model_pose=model_pose,
        sdf_tree=sdf_tree.model,
    )


def build_model_from_sdf(sdf: Union[Path, str]) -> descriptions.ModelDescription:

    # Parse data from the SDF
    sdf_data = extract_data_from_sdf(sdf=sdf)

    # Build the model description.
    # Note: if the model is fixed-base, the fixed joint between world and the first
    #       link is removed and the pose of the first link is updated.
    model = descriptions.ModelDescription.build_model_from(
        name=sdf_data.model_name,
        links=sdf_data.link_descriptions,
        joints=sdf_data.joint_descriptions,
        collisions=sdf_data.collision_shapes,
        fixed_base=sdf_data.fixed_base,
        base_link_name=sdf_data.base_link_name,
        model_pose=sdf_data.model_pose,
        considered_joints=[
            j.name
            for j in sdf_data.joint_descriptions
            if j.jtype is not descriptions.JointType.F
        ],
    )

    # Store the parsed SDF tree as extra info
    model = dataclasses.replace(model, extra_info=dict(sdf_tree=sdf_data.sdf_tree))

    return model
