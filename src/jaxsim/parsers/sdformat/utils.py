import os
import pathlib
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import trimesh
import sdformat

import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import Adjoint, Inertia
from jaxsim.parsers import descriptions
from jaxsim.parsers.sdformat import meshes


def resolve_local_uri(uri: str) -> str:
    """
    Resolve local URI (simple implementation).

    Args:
        uri: The URI to resolve.

    Returns:
        The resolved file path.
    """
    if uri.startswith("file://"):
        return uri[7:]  # Remove file:// prefix
    elif uri.startswith("package://"):
        # This would need proper ROS package resolution
        return uri.replace("package://", "")
    return uri


MeshMappingMethod = TypeVar("MeshMappingMethod", bound=Callable[..., npt.NDArray])


def from_sdf_inertial(inertial) -> jtp.Matrix:
    """
    Extract the 6D inertia matrix from an SDF inertial element.

    Args:
        inertial: The SDF inertial element.

    Returns:
        The 6D inertia matrix of the link expressed in the link frame.
    """

    # Extract the mass
    m = inertial.mass_matrix().mass()

    # Extract the moment of inertia matrix
    moi = inertial.mass_matrix().moi()

    # Build the 3x3 inertia matrix expressed in the CoM.
    I_CoM = np.array(
        [
            [moi(0, 0), moi(0, 1), moi(0, 2)],
            [moi(1, 0), moi(1, 1), moi(1, 2)],
            [moi(2, 0), moi(2, 1), moi(2, 2)],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM.
    M_CoM = Inertia.to_sixd(mass=m, com=np.zeros(3), I=I_CoM)

    # TODO: Extract pose from semantic pose if needed
    # For now, assume inertial frame is at link frame
    L_H_CoM = np.eye(4)

    # We need its inverse.
    CoM_X_L = Adjoint.from_transform(transform=L_H_CoM, inverse=True)

    # Express the CoM inertia matrix in the link frame L.
    M_L = CoM_X_L.T @ M_CoM @ CoM_X_L

    return M_L.astype(dtype=float)


def extract_pose_from_sdf_element(element) -> np.ndarray:
    """
    Extract the 4x4 transformation matrix from an SDF element's pose.

    Args:
        element: The SDF element (Link, Joint, etc.) with a pose.

    Returns:
        The 4x4 transformation matrix.
    """
    try:
        # Get the raw pose from the element
        raw_pose = element.raw_pose()

        # Extract position and rotation
        pos = raw_pose.pos()
        rot = raw_pose.rot()

        # Create transformation matrix
        transform = np.eye(4)

        # Set translation
        transform[0:3, 3] = [pos.x(), pos.y(), pos.z()]

        # Set rotation (quaternion to rotation matrix)
        # SDFormat uses w, x, y, z order
        import scipy.spatial.transform

        quat = [rot.x(), rot.y(), rot.z(), rot.w()]  # scipy expects x, y, z, w
        rotation_matrix = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
        transform[0:3, 0:3] = rotation_matrix

        return transform

    except (AttributeError, RuntimeError):
        # If pose extraction fails, return identity matrix
        return np.eye(4)


def joint_to_joint_type(joint) -> int:
    """
    Extract the joint type from an SDF joint.

    Args:
        joint: The parsed SDF joint.

    Returns:
        The integer corresponding to the joint type.
    """

    joint_type = joint.type()

    if joint_type == sdformat.JointType.FIXED:
        return descriptions.JointType.Fixed
    elif joint_type in [sdformat.JointType.REVOLUTE, sdformat.JointType.CONTINUOUS]:
        return descriptions.JointType.Revolute
    elif joint_type == sdformat.JointType.PRISMATIC:
        return descriptions.JointType.Prismatic
    else:
        raise ValueError(f"Joint type not supported: {joint_type}")


def create_box_collision(
    collision, link_description: descriptions.LinkDescription
) -> descriptions.BoxCollision:
    """
    Create a box collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The box collision description.
    """

    geometry = collision.geometry()
    if geometry.type() != sdformat.GeometryType.BOX:
        raise ValueError(f"Expected box geometry, got {geometry.type()}")

    box = geometry.box_shape()
    size = box.size()
    x, y, z = size.x(), size.y(), size.z()

    center = np.array([x / 2, y / 2, z / 2])

    # Define the bottom corners.
    bottom_corners = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0]])

    # Conditionally add the top corners based on the environment variable.
    top_corners = (
        np.array([[0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
        if os.environ.get("JAXSIM_COLLISION_USE_BOTTOM_ONLY", "0").lower()
        in {
            "false",
            "0",
        }
        else []
    )

    # Combine and shift by the center
    box_corners = np.vstack([bottom_corners, *top_corners]) - center

    # Extract collision pose transformation from SDFormat collision
    H = extract_pose_from_sdf_element(collision)

    center_wrt_link = (H @ np.hstack([center, 1.0]))[0:-1]
    box_corners_wrt_link = (
        H @ np.hstack([box_corners, np.vstack([1.0] * box_corners.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=np.array(corner),
            enabled=True,
        )
        for corner in box_corners_wrt_link.T
    ]

    return descriptions.BoxCollision(
        collidable_points=collidable_points, center=center_wrt_link
    )


def create_sphere_collision(
    collision, link_description: descriptions.LinkDescription
) -> descriptions.SphereCollision:
    """
    Create a sphere collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The sphere collision description.
    """

    # From https://stackoverflow.com/a/26127012
    def fibonacci_sphere(samples: int) -> npt.NDArray:
        # Get the golden ratio in radians.
        phi = np.pi * (3.0 - np.sqrt(5.0))

        # Generate the points.
        points = [
            np.array(
                [
                    np.cos(phi * i)
                    * np.sqrt(1 - (y := 1 - 2 * i / (samples - 1)) ** 2),
                    y,
                    np.sin(phi * i) * np.sqrt(1 - y**2),
                ]
            )
            for i in range(samples)
        ]

        # Filter to keep only the bottom half if required.
        if os.environ.get("JAXSIM_COLLISION_USE_BOTTOM_ONLY", "0").lower() in {
            "true",
            "1",
        }:
            # Keep only the points with z <= 0.
            points = [point for point in points if point[2] <= 0]

        return np.vstack(points)

    geometry = collision.geometry()
    if geometry.type() != sdformat.GeometryType.SPHERE:
        raise ValueError(f"Expected sphere geometry, got {geometry.type()}")

    sphere = geometry.sphere_shape()
    r = sphere.radius()

    sphere_points = r * fibonacci_sphere(
        samples=int(os.getenv(key="JAXSIM_COLLISION_SPHERE_POINTS", default="50"))
    )

    # Extract collision pose transformation from SDFormat collision
    H = extract_pose_from_sdf_element(collision)

    center_wrt_link = (H @ np.hstack([0, 0, 0, 1.0]))[0:-1]

    sphere_points_wrt_link = (
        H @ np.hstack([sphere_points, np.vstack([1.0] * sphere_points.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=np.array(point),
            enabled=True,
        )
        for point in sphere_points_wrt_link.T
    ]

    return descriptions.SphereCollision(
        collidable_points=collidable_points, center=center_wrt_link
    )


def create_mesh_collision(
    collision,
    link_description: descriptions.LinkDescription,
    method: MeshMappingMethod = None,
) -> descriptions.MeshCollision:
    """
    Create a mesh collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.
        method: The method to use for mesh wrapping.

    Returns:
        The mesh collision description.
    """

    geometry = collision.geometry()
    if geometry.type() != sdformat.GeometryType.MESH:
        raise ValueError(f"Expected mesh geometry, got {geometry.type()}")

    mesh_geom = geometry.mesh_shape()
    file = pathlib.Path(resolve_local_uri(uri=mesh_geom.uri()))
    file_type = file.suffix.replace(".", "")
    mesh = trimesh.load_mesh(file, file_type=file_type)

    if mesh.is_empty:
        raise RuntimeError(f"Failed to process '{file}' with trimesh")

    scale = mesh_geom.scale()
    mesh.apply_scale([scale.x(), scale.y(), scale.z()])
    logging.info(
        msg=f"Loading mesh {mesh_geom.uri()} with scale [{scale.x()}, {scale.y()}, {scale.z()}], file type '{file_type}'"
    )

    if method is None:
        method = meshes.VertexExtraction()
        logging.debug("Using default Vertex Extraction method for mesh wrapping")
    else:
        logging.debug(f"Using method {method} for mesh wrapping")

    points = method(mesh=mesh)
    logging.debug(f"Extracted {len(points)} points from mesh")

    # Extract collision pose transformation from SDFormat collision
    W_H_L = extract_pose_from_sdf_element(collision)

    # Extract translation from transformation matrix
    W_p_L = W_H_L[:3, 3]
    mesh_points_wrt_link = points @ W_H_L[:3, :3].T + W_p_L
    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=point,
            enabled=True,
        )
        for point in mesh_points_wrt_link
    ]

    return descriptions.MeshCollision(collidable_points=collidable_points, center=W_p_L)
