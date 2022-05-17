from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from jaxsim.parsers import descriptions


def from_sdf_string_list(string_list: str, epsilon: float = 1e-03) -> npt.NDArray:

    lst = np.array(string_list.split(" "), dtype=float)
    lst[np.abs(lst) < epsilon] = 0
    return lst


def from_sdf_pose(pose: str) -> npt.NDArray:

    # Parse the pose string
    pose_array = from_sdf_string_list(pose)
    assert pose_array.size == 6

    # Extract position and Euler angles
    position, rpy = pose_array[0:3], pose_array[3:6]

    # Transform euler to DCM matrix (sequence of extrinsic rotations, i.e. all angles
    # consider a fixed reference frame)
    DCM = R.from_euler(seq="xyz", angles=rpy).as_matrix()

    return np.vstack(
        [
            np.hstack([DCM, np.vstack(position)]),
            np.array([0, 0, 0, 1]),
        ]
    )


def flip_velocity_serialization(array: npt.NDArray) -> npt.NDArray:

    _np = np if isinstance(array, np.ndarray) else jnp
    array_in = array.squeeze()

    if array_in.shape[0] % 2 != 0:
        raise ValueError(array_in.shape)

    # array is a 6d vector
    if len(array_in.shape) == 1:

        half1, half2 = _np.split(array_in, 2)
        out = _np.hstack([half2, half1]).reshape(array.shape)
        return out

    # array is a 6x6 matrix
    else:

        # This will fail if the matrix is not 2D
        n_rows, n_cols = array.shape

        # Must be a square 2D matrix
        if not n_rows == n_cols:
            raise ValueError(array_in.shape)

        A, B = _np.split(array_in[0 : int(n_rows / 2), :], 2, axis=1)
        C, D = _np.split(array_in[int(n_rows / 2) :, :], 2, axis=1)

        return _np.block([[D, C], [B, A]])


def from_sdf_inertial(inertial_sdf_element) -> npt.NDArray:

    from jaxsim.sixd import se3, so3

    # Parse the inertial pose
    inertial_pose = from_sdf_string_list(
        string_list=inertial_sdf_element.pose, epsilon=1e-9
    )

    # Extract position and rpy
    inertial_position, inertial_rpy = np.split(inertial_pose, 2)

    # Extract the "mass" element
    m = inertial_sdf_element.mass

    # Extract the "inertia" element
    inertia_element = inertial_sdf_element.inertia

    # Build the 3x3 inertia matrix expressed in the CoM
    I_com = np.array(
        [
            [inertia_element.ixx, inertia_element.ixy, inertia_element.ixz],
            [inertia_element.ixy, inertia_element.iyy, inertia_element.iyz],
            [inertia_element.ixz, inertia_element.iyz, inertia_element.izz],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM
    I_generalized = np.vstack(
        [
            np.hstack([m * np.eye(3), np.zeros(shape=(3, 3))]),
            np.hstack([np.zeros(shape=(3, 3)), I_com]),
        ]
    )

    # Transform euler to DCM matrix (sequence of extrinsic rotations, i.e. all angles
    # consider a fixed reference frame)
    l_R_com = so3.SO3.from_matrix(
        R.from_euler(seq="xyz", angles=inertial_rpy).as_matrix()
    )

    # Compute the transform from the inertial frame (CoM) to the link frame
    l_H_com = se3.SE3.from_rotation_and_translation(
        rotation=l_R_com, translation=inertial_position
    )

    # We need its inverse
    com_H_l = l_H_com.inverse()

    # Express the CoM inertia matrix in the link frame
    I_expressed_in_link_frame = com_H_l.adjoint().T @ I_generalized @ com_H_l.adjoint()

    # Convert lin-ang serialization to ang-lin used by featherstone
    I_link = flip_velocity_serialization(I_expressed_in_link_frame)

    return jnp.array(I_link)


def axis_to_jtype(
    axis: Optional[npt.NDArray], type: str
) -> Union[descriptions.JointType, descriptions.JointDescriptor]:

    if type == "fixed":
        return descriptions.JointType.F

    if np.allclose(axis, [1, 0, 0]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Rx

    if np.allclose(axis, [0, 1, 0]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Ry

    if np.allclose(axis, [0, 0, 1]) and type in {"revolute", "continuous"}:
        return descriptions.JointType.Rz

    if np.allclose(axis, [1, 0, 0]) and type == "prismatic":
        return descriptions.JointType.Px

    if np.allclose(axis, [0, 1, 0]) and type == "prismatic":
        return descriptions.JointType.Py

    if np.allclose(axis, [0, 0, 1]) and type == "prismatic":
        return descriptions.JointType.Pz

    if type == "revolute":
        return descriptions.JointGenericAxis(
            code=descriptions.JointType.R, axis=np.array(axis, dtype=float)
        )

    if type == "prismatic":
        return descriptions.JointGenericAxis(
            code=descriptions.JointType.P, axis=np.array(axis, dtype=float)
        )

    raise ValueError("Joint not supported", axis, type)


def create_box_collision(
    collision_sdf_element, link_description: descriptions.LinkDescription
) -> descriptions.BoxCollision:

    x, y, z = from_sdf_string_list(collision_sdf_element.geometry.box.size)

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

    H = from_sdf_pose(pose=collision_sdf_element.pose.value)

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
