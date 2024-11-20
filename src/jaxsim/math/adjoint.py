import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

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

        Q_sixd = jaxlie.SO3(wxyz=quaternion)
        Q_sixd = Q_sixd if not normalize_quaternion else Q_sixd.normalize()

        return Adjoint.from_rotation_and_translation(
            rotation=Q_sixd.as_matrix(), translation=translation, inverse=inverse
        )

    @staticmethod
    def from_transform(transform: jtp.MatrixLike, inverse: bool = False) -> jtp.Matrix:
        """
        Create an adjoint matrix from a transformation matrix.

        Args:
            transform: A 4x4 transformation matrix.
            inverse: Whether to compute the inverse adjoint.

        Returns:
            The 6x6 adjoint matrix.
        """

        A_H_B = jnp.reshape(transform, (-1, 4, 4))

        return (
            jaxlie.SE3.from_matrix(matrix=A_H_B).adjoint()
            if not inverse
            else jaxlie.SE3.from_matrix(matrix=A_H_B).inverse().adjoint()
        ).reshape(transform.shape[:-2] + (6, 6))

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

        A_R_B = rotation.squeeze()
        A_o_B = translation.squeeze()

        if not inverse:
            X = A_X_B = jnp.vstack(  # noqa: F841
                [
                    jnp.block([A_R_B, Skew.wedge(A_o_B) @ A_R_B]),
                    jnp.block([jnp.zeros(shape=(3, 3)), A_R_B]),
                ]
            )
        else:
            X = B_X_A = jnp.vstack(  # noqa: F841
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
        X = adjoint.reshape(-1, 6, 6)

        R = X[..., 0:3, 0:3]
        o_x_R = X[..., 0:3, 3:6]

        H = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        R,
                        Skew.vee(matrix=o_x_R @ R.transpose(0, 2, 1))[
                            ..., :, jnp.newaxis
                        ],
                    ],
                    axis=-1,
                ),
                jnp.zeros((X.shape[0], 1, 4)).at[:, :, 3].set(1),
            ],
            axis=-2,
        ).reshape(adjoint.shape[:-2] + (4, 4))

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
        A_X_B = adjoint.reshape(-1, 6, 6)

        A_R_B_T = jnp.swapaxes(A_X_B[..., 0:3, 0:3], -2, -1)
        A_T_B = A_X_B[..., 0:3, 3:6]

        return jnp.concatenate(
            [
                jnp.concatenate(
                    [A_R_B_T, -A_R_B_T @ A_T_B @ A_R_B_T],
                    axis=-1,
                ),
                jnp.concatenate([jnp.zeros_like(A_R_B_T), A_R_B_T], axis=-1),
            ],
            axis=-2,
        ).reshape(adjoint.shape)
