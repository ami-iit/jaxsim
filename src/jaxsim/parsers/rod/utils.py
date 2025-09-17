import numpy as np
import rod

import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Inertia
from jaxsim.parsers import descriptions


def from_sdf_inertial(inertial: rod.Inertial) -> jtp.Matrix:
    """
    Extract the 6D inertia matrix from an SDF inertial element.

    Args:
        inertial: The SDF inertial element.

    Returns:
        The 6D inertia matrix of the link expressed in the link frame.
    """

    # Extract the "mass" element.
    m = inertial.mass

    # Extract the "inertia" element.
    inertia_element = inertial.inertia

    ixx = inertia_element.ixx
    iyy = inertia_element.iyy
    izz = inertia_element.izz
    ixy = inertia_element.ixy if inertia_element.ixy is not None else 0.0
    ixz = inertia_element.ixz if inertia_element.ixz is not None else 0.0
    iyz = inertia_element.iyz if inertia_element.iyz is not None else 0.0

    # Build the 3x3 inertia matrix expressed in the CoM.
    I_CoM = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM.
    M_CoM = Inertia.to_sixd(mass=m, com=np.zeros(3), I=I_CoM)

    # Compute the transform from the inertial frame (CoM) to the link frame.
    L_H_CoM = inertial.pose.transform() if inertial.pose is not None else np.eye(4)

    # We need its inverse.
    CoM_X_L = Adjoint.from_transform(transform=L_H_CoM, inverse=True)

    # Express the CoM inertia matrix in the link frame L.
    M_L = CoM_X_L.T @ M_CoM @ CoM_X_L

    return M_L.astype(dtype=float)


def joint_to_joint_type(joint: rod.Joint) -> int:
    """
    Extract the joint type from an SDF joint.

    Args:
        joint: The parsed SDF joint.

    Returns:
        The integer corresponding to the joint type.
    """

    axis = joint.axis
    joint_type = joint.type

    if joint_type == "fixed":
        return descriptions.JointType.Fixed

    if not (axis.xyz is not None and axis.xyz.xyz is not None):
        raise ValueError("Failed to read axis xyz data")

    # Make sure that the axis is a unary vector.
    axis_xyz = np.array(axis.xyz.xyz).astype(float)
    axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)

    if joint_type in {"revolute", "continuous"}:
        return descriptions.JointType.Revolute

    if joint_type == "prismatic":
        return descriptions.JointType.Prismatic

    raise ValueError("Joint not supported", axis_xyz, joint_type)


def create_box_collision(
    collision: rod.Collision, link_description: descriptions.LinkDescription
) -> descriptions.BoxCollision:
    """
    Create a box collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The box collision description.
    """

    x, y, z = collision.geometry.box.size

    center = np.array([x / 2, y / 2, z / 2])

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([center, 1.0]))[0:-1]

    return descriptions.BoxCollision(
        size=np.array([x, y, z]),
        center=center_wrt_link,
        parent_link=link_description.name,
    )


def create_sphere_collision(
    collision: rod.Collision, link_description: descriptions.LinkDescription
) -> descriptions.SphereCollision:
    """
    Create a sphere collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The sphere collision description.
    """

    r = collision.geometry.sphere.radius

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([0, 0, 0, 1.0]))[0:-1]

    return descriptions.SphereCollision(
        size=np.array([r] * 3),
        center=center_wrt_link,
        parent_link=link_description.name,
    )


def create_cylinder_collision(
    collision: rod.Collision, link_description: descriptions.LinkDescription
) -> descriptions.CylinderCollision:
    """
    Create a cylinder collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The cylinder collision description.
    """

    r = collision.geometry.cylinder.radius
    l = collision.geometry.cylinder.length

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([0, 0, 0, 1.0]))[0:-1]

    return descriptions.CylinderCollision(
        size=np.array([r, l, 0]),
        center=center_wrt_link,
        parent_link=link_description.name,
    )
