import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp


class Transform:

    @staticmethod
    def from_quaternion_and_translation(
        quaternion: jtp.VectorLike = jnp.array([1.0, 0, 0, 0]),
        translation: jtp.VectorLike = jnp.zeros(3),
        inverse: jtp.BoolLike = False,
        normalize_quaternion: jtp.BoolLike = False,
    ) -> jtp.Matrix:
        """
        Create a transformation matrix from a quaternion and a translation.

        Args:
            quaternion: A 4D vector representing a SO(3) orientation.
            translation: A 3D vector representing a translation.
            inverse: Whether to compute the inverse transformation.
            normalize_quaternion:
                Whether to normalize the quaternion before creating the transformation.

        Returns:
            The 4x4 transformation matrix representing the SE(3) transformation.
        """

        W_Q_B = jnp.array(quaternion).astype(float)
        W_p_B = jnp.array(translation).astype(float)

        assert W_p_B.size == 3
        assert W_Q_B.size == 4

        A_R_B = jaxlie.SO3(wxyz=W_Q_B)
        A_R_B = A_R_B if not normalize_quaternion else A_R_B.normalize()

        A_H_B = jaxlie.SE3.from_rotation_and_translation(
            rotation=A_R_B, translation=W_p_B
        )

        return A_H_B.as_matrix() if not inverse else A_H_B.inverse().as_matrix()

    @staticmethod
    def from_rotation_and_translation(
        rotation: jtp.MatrixLike = jnp.eye(3),
        translation: jtp.VectorLike = jnp.zeros(3),
        inverse: jtp.BoolLike = False,
    ) -> jtp.Matrix:
        """
        Create a transformation matrix from a rotation matrix and a translation vector.

        Args:
            rotation: A 3x3 rotation matrix representing a SO(3) orientation.
            translation: A 3D vector representing a translation.
            inverse: Whether to compute the inverse transformation.

        Returns:
            The 4x4 transformation matrix representing the SE(3) transformation.
        """

        A_R_B = jnp.array(rotation).astype(float)
        W_p_B = jnp.array(translation).astype(float)

        assert W_p_B.size == 3
        assert A_R_B.shape == (3, 3)

        A_H_B = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.from_matrix(A_R_B), translation=W_p_B
        )

        return A_H_B.as_matrix() if not inverse else A_H_B.inverse().as_matrix()

    @staticmethod
    def inverse(transform: jtp.MatrixLike) -> jtp.Matrix:
        """
        Compute the inverse transformation matrix.

        Args:
            transform: A 4x4 transformation matrix.

        Returns:
            The 4x4 inverse transformation matrix.
        """

        A_H_B = jnp.array(transform).astype(float)
        assert A_H_B.shape == (4, 4)

        return jaxlie.SE3.from_matrix(matrix=A_H_B).inverse().as_matrix()
