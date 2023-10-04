import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.sixd import so3

from .quaternion import Quaternion
from .skew import Skew


class Adjoint:
    @staticmethod
    def from_quaternion_and_translation(
        quaternion: jtp.Vector = jnp.array([1.0, 0, 0, 0]),
        translation: jtp.Vector = jnp.zeros(3),
        inverse: bool = False,
        normalize_quaternion: bool = False,
    ) -> jtp.Matrix:
        """
        Create an adjoint matrix from a quaternion and a translation.

        Args:
            quaternion (jtp.Vector): A quaternion vector (4D) representing orientation.
            translation (jtp.Vector): A translation vector (3D).
            inverse (bool): Whether to compute the inverse adjoint. Default is False.
            normalize_quaternion (bool): Whether to normalize the quaternion before creating the adjoint.
                                         Default is False.

        Returns:
            jtp.Matrix: The adjoint matrix.
        """
        assert quaternion.size == 4
        assert translation.size == 3

        Q_sixd = so3.SO3.from_quaternion_xyzw(xyzw=Quaternion.to_xyzw(quaternion))
        Q_sixd = Q_sixd if not normalize_quaternion else Q_sixd.normalize()

        return Adjoint.from_rotation_and_translation(
            rotation=Q_sixd.as_matrix(), translation=translation, inverse=inverse
        )

    @staticmethod
    def from_rotation_and_translation(
        rotation: jtp.Matrix = jnp.eye(3),
        translation: jtp.Vector = jnp.zeros(3),
        inverse: bool = False,
    ) -> jtp.Matrix:
        """
        Create an adjoint matrix from a rotation matrix and a translation vector.

        Args:
            rotation (jtp.Matrix): A 3x3 rotation matrix.
            translation (jtp.Vector): A translation vector (3D).
            inverse (bool): Whether to compute the inverse adjoint. Default is False.

        Returns:
            jtp.Matrix: The adjoint matrix.
        """
        assert rotation.shape == (3, 3)
        assert translation.size == 3

        A_R_B = rotation.squeeze()
        A_o_B = translation.squeeze()

        if not inverse:
            X = A_X_B = jnp.vstack(
                [
                    jnp.block([A_R_B, Skew.wedge(A_o_B) @ A_R_B]),
                    jnp.block([jnp.zeros(shape=(3, 3)), A_R_B]),
                ]
            )
        else:
            X = B_X_A = jnp.vstack(
                [
                    jnp.block([A_R_B.T, -A_R_B.T @ Skew.wedge(A_o_B)]),
                    jnp.block([jnp.zeros(shape=(3, 3)), A_R_B.T]),
                ]
            )

        return X

    @staticmethod
    def to_transform(adjoint: jtp.Matrix) -> jtp.Matrix:
        """
        Convert an adjoint matrix to a transformation matrix.

        Args:
            adjoint (jtp.Matrix): The adjoint matrix (6x6).

        Returns:
            jtp.Matrix: The transformation matrix (4x4).
        """
        X = adjoint.squeeze()
        assert X.shape == (6, 6)

        R = X[0:3, 0:3]
        o_x_R = X[0:3, 3:6]

        H = jnp.vstack(
            [
                jnp.block([R, Skew.vee(matrix=o_x_R @ R.T)]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

        return H

    @staticmethod
    def inverse(adjoint: jtp.Matrix) -> jtp.Matrix:
        """
        Compute the inverse of an adjoint matrix.

        Args:
            adjoint (jtp.Matrix): The adjoint matrix.

        Returns:
            jtp.Matrix: The inverse adjoint matrix.
        """
        A_X_B = adjoint
        A_H_B = Adjoint.to_transform(adjoint=A_X_B)

        A_R_B = A_H_B[0:3, 0:3]
        A_o_B = A_H_B[0:3, 3]

        return Adjoint.from_rotation_and_translation(
            rotation=A_R_B, translation=A_o_B, inverse=True
        )
