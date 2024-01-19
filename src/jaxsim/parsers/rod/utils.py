import os
from typing import Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import rod

from jaxsim.parsers import descriptions


def from_sdf_inertial(inertial: rod.Inertial) -> npt.NDArray:
    """
    Extract the 6D inertia matrix from an SDF inertial element.

    Args:
        inertial: The SDF inertial element.

    Returns:
        The 6D inertia matrix of the link expressed in the link frame.
    """

    from jaxsim.math.inertia import Inertia
    from jaxsim.sixd import se3

    # Extract the "mass" element
    m = inertial.mass

    # Extract the "inertia" element
    inertia_element = inertial.inertia

    ixx = inertia_element.ixx
    iyy = inertia_element.iyy
    izz = inertia_element.izz
    ixy = inertia_element.ixy if inertia_element.ixy is not None else 0.0
    ixz = inertia_element.ixz if inertia_element.ixz is not None else 0.0
    iyz = inertia_element.iyz if inertia_element.iyz is not None else 0.0

    # Build the 3x3 inertia matrix expressed in the CoM
    I_CoM = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM
    M_CoM = Inertia.to_sixd(mass=m, com=np.zeros(3), I=I_CoM)

    # Compute the transform from the inertial frame (CoM) to the link frame
    L_H_CoM = inertial.pose.transform() if inertial.pose is not None else np.eye(4)

    # We need its inverse
    CoM_H_L = se3.SE3.from_matrix(matrix=L_H_CoM).inverse()
    CoM_X_L: npt.NDArray = CoM_H_L.adjoint()

    # Express the CoM inertia matrix in the link frame L
    M_L = CoM_X_L.T @ M_CoM @ CoM_X_L

    return jnp.array(M_L)


def axis_to_jtype(
    axis: rod.Axis, type: str
) -> Union[descriptions.JointType, descriptions.JointDescriptor]:
    """
    Convert an SDF axis to a joint type.

    Args:
        axis: The SDF axis.
        type: The SDF joint type.

    Returns:
        The corresponding joint type description.
    """

    if type == "fixed":
        return descriptions.JointType.F

    if not (axis.xyz is not None and axis.xyz.xyz is not None):
        raise ValueError("Failed to read axis xyz data")

    axis_xyz = np.array(axis.xyz.xyz)

    if np.allclose(axis_xyz, [1, 0, 0]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Rx

    if np.allclose(axis_xyz, [0, 1, 0]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Ry

    if np.allclose(axis_xyz, [0, 0, 1]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Rz

    if np.allclose(axis_xyz, [1, 0, 0]) and type == "prismatic":
        return descriptions.JointType.Px

    if np.allclose(axis_xyz, [0, 1, 0]) and type == "prismatic":
        return descriptions.JointType.Py

    if np.allclose(axis_xyz, [0, 0, 1]) and type == "prismatic":
        return descriptions.JointType.Pz

    if type == "revolute":
        return descriptions.JointGenericAxis(
            code=descriptions.JointType.R, axis=np.array(axis_xyz, dtype=float)
        )

    if type == "prismatic":
        return descriptions.JointGenericAxis(
            code=descriptions.JointType.P, axis=np.array(axis_xyz, dtype=float)
        )

    raise ValueError("Joint not supported", axis_xyz, type)


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

    box_corners = (
        np.vstack(
            [
                np.array([0, 0, 0]),
                np.array([x, 0, 0]),
                np.array([x, y, 0]),
                np.array([0, y, 0]),
                np.array([0, 0, z]),
                np.array([x, 0, z]),
                np.array([x, y, z]),
                np.array([0, y, z]),
            ]
        )
        - center
    )

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([center, 1.0]))[0:-1]
    box_corners_wrt_link = (
        H @ np.hstack([box_corners, np.vstack([1.0] * box_corners.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=corner,
            enabled=True,
        )
        for corner in box_corners_wrt_link.T
    ]

    return descriptions.BoxCollision(
        collidable_points=collidable_points, center=center_wrt_link
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

    # From https://stackoverflow.com/a/26127012
    def fibonacci_sphere(samples: int) -> npt.NDArray:
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append(np.array([x, y, z]))

        return np.vstack(points)

    r = collision.geometry.sphere.radius
    sphere_points = r * fibonacci_sphere(
        samples=int(os.getenv(key="JAXSIM_COLLISION_SPHERE_POINTS", default="250"))
    )

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([0, 0, 0, 1.0]))[0:-1]

    sphere_points_wrt_link = (
        H @ np.hstack([sphere_points, np.vstack([1.0] * sphere_points.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=point,
            enabled=True,
        )
        for point in sphere_points_wrt_link.T
    ]

    return descriptions.SphereCollision(
        collidable_points=collidable_points, center=center_wrt_link
    )
