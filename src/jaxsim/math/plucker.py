from typing import Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Plucker:
    @staticmethod
    def from_rot_and_trans(dcm: jtp.Matrix, translation: jtp.Vector) -> jtp.Matrix:
        """
        Computes the Plücker matrix from a rotation matrix and a translation vector.

        Args:
            dcm: A 3x3 rotation matrix.
            translation: A 3x1 translation vector.

        Returns:
            A 6x6 Plücker matrix.
        """
        R = dcm

        X = jnp.block(
            [
                [R, -R @ Skew.wedge(vector=translation)],
                [jnp.zeros(shape=(3, 3)), R],
            ]
        )

        return X

    @staticmethod
    def to_rot_and_trans(adjoint: jtp.Matrix) -> Tuple[jtp.Matrix, jtp.Vector]:
        """
        Computes the rotation matrix and translation vector from a Plücker matrix.

        Args:
            adjoint: A 6x6 Plücker matrix.

        Returns:
            A tuple containing the 3x3 rotation matrix and the 3x1 translation vector.
        """
        X = adjoint

        R = X[0:3, 0:3]
        p = -Skew.vee(R.T @ X[0:3, 3:6])

        return R, p

    @staticmethod
    def from_transform(transform: jtp.Matrix) -> jtp.Matrix:
        """
        Computes the Plücker matrix from a homogeneous transformation matrix.

        Args:
            transform: A 4x4 homogeneous transformation matrix.

        Returns:
            A 6x6 Plücker matrix.
        """
        H = transform

        R = H[0:3, 0:3]
        p = H[0:3, 3]

        X = jnp.block(
            [
                [R, Skew.wedge(vector=p) @ R],
                [jnp.zeros(shape=(3, 3)), R],
            ]
        )

        return X

    @staticmethod
    def to_transform(adjoint: jtp.Matrix) -> jtp.Matrix:
        """
        Computes the homogeneous transformation matrix from a Plücker matrix.

        Args:
            adjoint: A 6x6 Plücker matrix.

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        X = adjoint

        R = X[0:3, 0:3]
        o_x_R = X[0:3, 3:6]

        H = jnp.vstack(
            [
                jnp.hstack([R, Skew.vee(matrix=o_x_R @ R.T)]),
                [0, 0, 0, 1],
            ]
        )

        return H
