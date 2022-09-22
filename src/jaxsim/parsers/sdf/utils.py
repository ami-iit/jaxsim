from typing import Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pysdf
from scipy.spatial.transform import Rotation as R

from jaxsim.parsers import descriptions


def from_sdf_string_list(string_list: str, epsilon: float = 1e-06) -> npt.NDArray:

    lst = np.array(string_list.split(" "), dtype=float)
    lst[np.abs(lst) < epsilon] = 0
    return lst


def from_sdf_pose(pose: pysdf.Pose) -> npt.NDArray:

    # Transform euler to DCM matrix (sequence of extrinsic rotations, i.e. all angles
    # consider a fixed reference frame)
    DCM = R.from_euler(
        seq="xyz", angles=pose.orientation, degrees=pose.degrees
    ).as_matrix()

    return np.vstack(
        [
            np.hstack([DCM, np.vstack(pose.position)]),
            np.array([0, 0, 0, 1]),
        ]
    )


def from_sdf_inertial(inertial: pysdf.Link.Inertial) -> npt.NDArray:

    from jaxsim.math.inertia import Inertia
    from jaxsim.sixd import se3, so3

    # Extract the "mass" element
    m = inertial.mass

    # Extract the "inertia" element
    inertia_element = inertial.inertia

    # Build the 3x3 inertia matrix expressed in the CoM
    I_com = np.array(
        [
            [inertia_element.ixx, inertia_element.ixy, inertia_element.ixz],
            [inertia_element.ixy, inertia_element.iyy, inertia_element.iyz],
            [inertia_element.ixz, inertia_element.iyz, inertia_element.izz],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM
    I_generalized = Inertia.to_sixd(mass=m, com=np.zeros(3), I=I_com)

    # Transform euler to DCM matrix (sequence of extrinsic rotations, i.e. all angles
    # consider a fixed reference frame)
    l_R_com = so3.SO3.from_matrix(
        R.from_euler(
            seq="xyz", angles=inertial.pose.orientation, degrees=inertial.pose.degrees
        ).as_matrix()
    )

    # Compute the transform from the inertial frame (CoM) to the link frame
    l_H_com = se3.SE3.from_rotation_and_translation(
        rotation=l_R_com, translation=np.array(inertial.pose.position)
    )

    # We need its inverse
    com_H_l = l_H_com.inverse()
    com_X_l = com_H_l.adjoint()

    # Express the CoM inertia matrix in the link frame
    I_expressed_in_link_frame = com_X_l.T @ I_generalized @ com_X_l

    return jnp.array(I_expressed_in_link_frame)


def axis_to_jtype(
    axis: pysdf.Joint.Axis, type: str
) -> Union[descriptions.JointType, descriptions.JointDescriptor]:

    if type == "fixed":
        return descriptions.JointType.F

    if not (axis.xyz is not None and axis.xyz.text is not None):
        raise ValueError("Failed to read axis xyz data")

    axis_xyz = from_sdf_string_list(axis.xyz.text)

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
    collision: pysdf.Collision, link_description: descriptions.LinkDescription
) -> descriptions.BoxCollision:

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

    H = from_sdf_pose(pose=collision.pose)

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
    collision: pysdf.Collision, link_description: descriptions.LinkDescription
) -> descriptions.SphereCollision:

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
    sphere_points = r * fibonacci_sphere(samples=250)

    H = from_sdf_pose(pose=collision.pose)

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
