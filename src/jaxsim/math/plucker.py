from typing import Tuple

import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Plucker:
    @staticmethod
    def from_rot_and_trans(dcm: jtp.Matrix, translation: jtp.Vector) -> jtp.Matrix:
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
        X = adjoint

        R = X[0:3, 0:3]
        p = -Skew.vee(R.T @ X[0:3, 3:6])

        return R, p

    @staticmethod
    def from_transform(transform: jtp.Matrix) -> jtp.Matrix:
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
