import dataclasses
import os
import pathlib
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import sdformat

from jaxsim import logging
from jaxsim.math import Quaternion
from jaxsim.parsers import descriptions, kinematic_graph

from . import utils


class SDFData(NamedTuple):
    """
    Data extracted fr        for i in range(sdf_model.joint_count()):
            j = sdf_model.joint_by_index(i)m an SDF resource useful to build a JaxSim model.
    """

    model_name: str

    fixed_base: bool
    base_link_name: str

    link_descriptions: list[descriptions.LinkDescription]
    joint_descriptions: list[descriptions.JointDescription]
    frame_descriptions: list[descriptions.LinkDescription]
    collision_shapes: list[descriptions.CollisionShape]

    sdf_model: sdformat.Model | None = None
    model_pose: kinematic_graph.RootPose = kinematic_graph.RootPose()


def extract_model_data(
    model_description: pathlib.Path | str | sdformat.Model | sdformat.Root,
    model_name: str | None = None,
    is_urdf: bool | None = None,
) -> SDFData:
    """
    Extract data from an SDF/URDF resource useful to build a JaxSim model.

    Args:
        model_description:
            A path to an SDF/URDF file, a string containing its content, or
            a pre-parsed/pre-built sdformat model.
        model_name: The name of the model to extract from the SDF resource.
        is_urdf:
            Whether to force parsing the resource as a URDF file. Automatically
            detected if not provided.

    Returns:
        The extracted model data.
    """

    match model_description:
        case sdformat.Model():
            sdf_model = model_description
        case sdformat.Root() | str() | pathlib.Path():
            if isinstance(model_description, sdformat.Root):
                root = model_description
            else:
                root = sdformat.Root()
                if isinstance(model_description, (str, pathlib.Path)):
                    # Check if it's a file path or string content
                    if isinstance(model_description, pathlib.Path) or (
                        isinstance(model_description, str)
                        and os.path.isfile(model_description)
                    ):
                        # It's a file path
                        try:
                            root.load(str(model_description))
                        except sdformat.SDFErrorsException as e:
                            raise RuntimeError(f"Failed to load SDF file: {e}") from e
                    else:
                        # It's string content
                        try:
                            root.load_sdf_string(model_description)
                        except sdformat.SDFErrorsException as e:
                            raise RuntimeError(
                                f"Failed to parse SDF string: {e}"
                            ) from e

            # Get the model from the root
            if root.model() is not None:
                sdf_model = root.model()
            elif root.world_count() > 0:
                # Try to get model from world
                world = root.world_by_index(0)
                if world.model_count() > 0:
                    sdf_model = world.model_by_index(0)
                else:
                    raise RuntimeError("No models found in SDF resource")
            else:
                raise RuntimeError("No models found in SDF resource")

            # If model_name is specified, try to find it
            if model_name is not None:
                if root.model() and root.model().name() == model_name:
                    sdf_model = root.model()
                else:
                    # Search in worlds
                    found = False
                    for i in range(root.world_count()):
                        world = root.world_by_index(i)
                        for j in range(world.model_count()):
                            model = world.model_by_index(j)
                            if model.name() == model_name:
                                sdf_model = model
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        raise RuntimeError(
                            f"Model '{model_name}' not found in SDF resource"
                        )

    # Log model name.
    logging.info(msg=f"Found model '{sdf_model.name()}' in SDF resource")

    # Jaxsim supports only models compatible with URDF, i.e. those having all links
    # directly attached to their parent joint without additional roto-translations.
    # Note: The real SDFormat API handles frame conventions differently than ROD
    # TODO: Implement frame convention handling if needed

    # Log type of base link.
    canonical_link = sdf_model.canonical_link_name()

    # If canonical link is empty, use the first link as the base link
    if not canonical_link and sdf_model.link_count() > 0:
        canonical_link = sdf_model.link_by_index(0).name()
        logging.info(
            f"No canonical link specified, using first link '{canonical_link}' as base"
        )

    logging.debug(
        msg=f"Model '{sdf_model.name()}' has canonical link '{canonical_link}'"
    )

    # Log detected base link.
    logging.debug(msg=f"Considering '{canonical_link}' as base link")

    # Pose of the model
    # TODO: Extract pose from SDFormat model if needed
    model_pose = kinematic_graph.RootPose()

    # ===========
    # Parse links
    # ===========

    # Parse the links (unconnected).
    links = []
    for i in range(sdf_model.link_count()):
        l = sdf_model.link_by_index(i)
        mass = float(l.inertial().mass_matrix().mass())

        # Always include the canonical link, even if it has zero mass
        # For other links, only include them if they have positive mass
        if mass > 0 or l.name() == canonical_link:
            # Extract link pose from SDFormat
            link_pose = utils.extract_pose_from_sdf_element(l)

            links.append(
                descriptions.LinkDescription(
                    name=l.name(),
                    mass=mass,
                    inertia=utils.from_sdf_inertial(inertial=l.inertial()),
                    pose=link_pose,
                )
            )

    # Create a dictionary to find easily links.
    links_dict: dict[str, descriptions.LinkDescription] = {l.name: l for l in links}

    # ============
    # Parse frames
    # ============

    # Parse the frames (unconnected).
    frames = []

    # Create a dictionary of custom frames declared in the model.
    custom_frames_dict: dict[str, descriptions.LinkDescription] = {}
    for i in range(sdf_model.frame_count()):
        f = sdf_model.frame_by_index(i)
        if f.attached_to() in links_dict:
            # Convert pose to transformation matrix
            pose = f.raw_pose()
            pos = pose.pos()
            rot = pose.rot()

            # Create transformation matrix from position and quaternion
            import scipy.spatial.transform

            r = scipy.spatial.transform.Rotation.from_quat(
                [rot.x(), rot.y(), rot.z(), rot.w()]
            )
            transform = jnp.eye(4)
            transform = transform.at[:3, :3].set(r.as_matrix())
            transform = transform.at[:3, 3].set([pos.x(), pos.y(), pos.z()])

            frame_link = descriptions.LinkDescription(
                name=f.name(),
                mass=jnp.array(0.0, dtype=float),
                inertia=jnp.zeros(shape=(3, 3)),
                parent_name=f.attached_to(),
                pose=transform,
            )
            frames.append(frame_link)
            custom_frames_dict[f.name()] = frame_link

    # =========================
    # Process fixed-base models
    # =========================

    # In this case, we need to get the pose of the joint that connects the base link
    # to the world and combine their pose.
    if sdf_model.static():
        # Create a massless word link
        world_link = descriptions.LinkDescription(
            name="world", mass=0, inertia=np.zeros(shape=(6, 6))
        )

        # Gather joints connecting fixed-base models to the world.
        # TODO: the pose of this joint could be expressed wrt any arbitrary frame,
        #       here we assume is expressed wrt the model. This also means that the
        #       default model pose matches the pose of the fake "world" link.
        joints_with_world_parent = [
            descriptions.JointDescription(
                name=j.name(),
                parent=world_link,
                child=links_dict[j.child_name()],
                jtype=utils.joint_to_joint_type(joint=j),
                axis=None,  # TODO: Extract axis properly for fixed joints
                pose=np.eye(4),  # TODO: Extract proper pose from SDFormat joint
            )
            for j in [
                sdf_model.joint_by_index(i) for i in range(sdf_model.joint_count())
            ]
            if j.type() == sdformat.JointType.FIXED
            and j.parent_name() == "world"
            and j.child_name() in links_dict
        ]

        logging.debug(
            f"Found joints connecting to world: {[j.name for j in joints_with_world_parent]}"
        )

        if len(joints_with_world_parent) != 1:
            msg = "Found more/less than one joint connecting a fixed-base model to the world"
            raise ValueError(msg + f": {[j.name for j in joints_with_world_parent]}")

        base_link_name = joints_with_world_parent[0].child.name

        msg = "Combining the pose of base link '{}' with the pose of joint '{}'"
        logging.debug(msg.format(base_link_name, joints_with_world_parent[0].name))

        # Combine the pose of the base link (child of the found fixed joint)
        # with the pose of the fixed joint connecting with the world.
        # Note: we assume it's a fixed joint and ignore any joint angle.
        links_dict[base_link_name].mutable(validate=False).pose = (
            joints_with_world_parent[0].pose @ links_dict[base_link_name].pose
        )

    # ============
    # Parse joints
    # ============

    # Check that all joint poses are expressed w.r.t. their parent link.
    for joint_idx in range(sdf_model.joint_count()):
        j = sdf_model.joint_by_index(joint_idx)
        # SDFormat poses are handled differently, skip complex pose validation for now
        continue

    # Parse the joints.
    joints = []
    for joint_idx in range(sdf_model.joint_count()):
        j = sdf_model.joint_by_index(joint_idx)

        # Filter joint types
        joint_type_map = {
            sdformat.JointType.REVOLUTE: "revolute",
            sdformat.JointType.CONTINUOUS: "continuous",
            sdformat.JointType.PRISMATIC: "prismatic",
            sdformat.JointType.FIXED: "fixed",
        }

        if (
            j.type() not in joint_type_map
            or j.parent_name() == "world"
            or j.child_name() not in links_dict
        ):
            continue

        # Extract axis information
        axis = None
        position_limit_lower = jnp.finfo(float).min
        position_limit_upper = jnp.finfo(float).max
        friction_static = 0.0
        friction_viscous = 0.0
        position_limit_damper = float(
            os.environ.get("JAXSIM_JOINT_POSITION_LIMIT_DAMPER", 0.0)
        )
        position_limit_spring = float(
            os.environ.get("JAXSIM_JOINT_POSITION_LIMIT_SPRING", 0.0)
        )

        # Extract axis and joint parameters (SDFormat joints have a single axis at index 0)
        try:
            axis_obj = j.axis(0)
            if axis_obj is not None:
                axis_xyz = axis_obj.xyz()
                axis = np.array([axis_xyz.x(), axis_xyz.y(), axis_xyz.z()], dtype=float)

                # Extract limits and dynamics
                position_limit_lower = (
                    float(axis_obj.lower())
                    if axis_obj.lower() != -float("inf")
                    else jnp.finfo(float).min
                )
                position_limit_upper = (
                    float(axis_obj.upper())
                    if axis_obj.upper() != float("inf")
                    else jnp.finfo(float).max
                )
                friction_static = float(axis_obj.friction())
                friction_viscous = float(axis_obj.damping())
        except (AttributeError, RuntimeError, IndexError):
            # Some joint types may not have axis information
            pass

        # Extract joint pose from SDFormat
        joint_pose = utils.extract_pose_from_sdf_element(j)

        joint_desc = descriptions.JointDescription(
            name=j.name(),
            parent=links_dict[j.parent_name()],
            child=links_dict[j.child_name()],
            jtype=utils.joint_to_joint_type(joint=j),
            axis=axis,
            pose=joint_pose,
            initial_position=0.0,
            position_limit=(position_limit_lower, position_limit_upper),
            friction_static=friction_static,
            friction_viscous=friction_viscous,
            position_limit_damper=position_limit_damper,
            position_limit_spring=position_limit_spring,
        )
        joints.append(joint_desc)

    # Create a dictionary to find the parent joint of the links.
    joint_dict = {j.child.name: j.name for j in joints}

    # Check that all the link poses are expressed wrt their parent joint.
    for link_idx in range(sdf_model.link_count()):
        l = sdf_model.link_by_index(link_idx)
        if l.name() not in links_dict:
            continue

        # SDFormat poses are handled differently, skip complex pose validation for now
        # TODO: Implement proper pose validation if needed
        continue

        if l.name() not in joint_dict:
            raise ValueError(f"Failed to find parent joint of link '{l.name()}'")

        # TODO: Implement proper pose checking with SDFormat API
        # if l.pose.relative_to != joint_dict[l.name()]:
        #     msg = "Pose of link '{}' is not expressed wrt its parent joint '{}'"
        #     raise ValueError(msg.format(l.name(), joint_dict[l.name()]))

    # ================
    # Parse collisions
    # ================

    # Initialize the collision shapes
    collisions: list[descriptions.CollisionShape] = []

    # Parse the collisions
    for link_idx in range(sdf_model.link_count()):
        link = sdf_model.link_by_index(link_idx)
        for collision_idx in range(link.collision_count()):
            collision = link.collision_by_index(collision_idx)

            if collision.geometry().type() == sdformat.GeometryType.BOX:
                box_collision = utils.create_box_collision(
                    collision=collision,
                    link_description=links_dict[link.name()],
                )
                collisions.append(box_collision)

            if collision.geometry().type() == sdformat.GeometryType.SPHERE:
                sphere_collision = utils.create_sphere_collision(
                    collision=collision,
                    link_description=links_dict[link.name()],
                )
                collisions.append(sphere_collision)

            if collision.geometry().type() == sdformat.GeometryType.MESH and int(
                os.environ.get("JAXSIM_COLLISION_MESH_ENABLED", "0")
            ):
                logging.warning("Mesh collision support is still experimental.")
                mesh_collision = utils.create_mesh_collision(
                    collision=collision,
                    link_description=links_dict[link.name()],
                    method=utils.meshes.extract_points_vertices,
                )
                collisions.append(mesh_collision)

    return SDFData(
        model_name=sdf_model.name(),
        link_descriptions=links,
        joint_descriptions=joints,
        frame_descriptions=frames,
        collision_shapes=collisions,
        fixed_base=sdf_model.static(),
        base_link_name=canonical_link,
        model_pose=model_pose,
        sdf_model=sdf_model,
    )


def build_model_description(
    model_description: pathlib.Path | str | sdformat.Model,
    is_urdf: bool | None = None,
) -> descriptions.ModelDescription:
    """
    Build a model description from an SDF/URDF resource.

    Args:
        model_description: A path to an SDF/URDF file, a string containing its content,
          or a pre-parsed/pre-built rod model.
        is_urdf: Whether the force parsing the resource as a URDF file. Automatically
            detected if not provided.

    Returns:
        The parsed model description.
    """

    # Parse data from the SDF assuming it contains a single model.
    sdf_data = extract_model_data(
        model_description=model_description, model_name=None, is_urdf=is_urdf
    )

    # Build the intermediate representation used for building a JaxSim model.
    # This process, beyond other operations, removes the fixed joints.
    # Note: if the model is fixed-base, the fixed joint between world and the first
    #       link is removed and the pose of the first link is updated.
    #
    # The whole process is:
    # URDF/SDF ⟶ rod.Model ⟶ ModelDescription ⟶ JaxSimModel.
    graph = descriptions.ModelDescription.build_model_from(
        name=sdf_data.model_name,
        links=sdf_data.link_descriptions,
        joints=sdf_data.joint_descriptions,
        frames=sdf_data.frame_descriptions,
        collisions=sdf_data.collision_shapes,
        fixed_base=sdf_data.fixed_base,
        base_link_name=sdf_data.base_link_name,
        model_pose=sdf_data.model_pose,
        considered_joints=[
            j.name
            for j in sdf_data.joint_descriptions
            if j.jtype is not descriptions.JointType.Fixed
        ],
    )

    # Store the parsed SDF tree as extra info
    graph = dataclasses.replace(graph, _extra_info={"sdf_model": sdf_data.sdf_model})

    return graph
