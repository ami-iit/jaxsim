import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Quaternion:
    @staticmethod
    def to_xyzw(wxyz: jtp.Vector) -> jtp.Vector:

        return wxyz.squeeze()[jnp.array([1, 2, 3, 0])]

    @staticmethod
    def to_wxyz(xyzw: jtp.Vector) -> jtp.Vector:

        return xyzw.squeeze()[jnp.array([3, 0, 1, 2])]

    @staticmethod
    def to_dcm(quaternion: jtp.Vector) -> jtp.Matrix:

        q = quaternion / jnp.linalg.norm(quaternion)

        q0s = q[0] * q[0]
        q1s = q[1] * q[1]
        q2s = q[2] * q[2]
        q3s = q[3] * q[3]
        q01 = q[0] * q[1]
        q02 = q[0] * q[2]
        q03 = q[0] * q[3]
        q12 = q[1] * q[2]
        q13 = q[3] * q[1]
        q23 = q[2] * q[3]

        R = 2 * jnp.array(
            [
                [q0s + q1s - 0.5, q12 + q03, q13 - q02],
                [q12 - q03, q0s + q2s - 0.5, q23 + q01],
                [q13 + q02, q23 - q01, q0s + q3s - 0.5],
            ]
        )

        return R.squeeze()

    @staticmethod
    def from_dcm(dcm: jtp.Matrix) -> jtp.Vector:

        R = dcm.squeeze()

        tr = jnp.trace(R)
        v = -Skew.vee(R)

        q = jnp.vstack([(tr + 1) / 2.0, v])

        return jnp.vstack(q) / jnp.linalg.norm(q)

    @staticmethod
    def derivative(
        quaternion: jtp.Vector,
        omega: jtp.Vector,
        omega_in_body_fixed: bool = False,
        K: float = 0.1,
    ) -> jtp.Vector:

        w = omega.squeeze()
        qw, qx, qy, qz = quaternion.squeeze()

        if omega_in_body_fixed:

            Q = jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, -qz, qy],
                    [qy, qz, qw, -qx],
                    [qz, -qy, qx, qw],
                ]
            )

        else:

            Q = jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, qz, -qy],
                    [qy, -qz, qw, qx],
                    [qz, qy, -qx, qw],
                ]
            )

        qd = 0.5 * (
            Q
            @ jnp.hstack(
                [K * jnp.linalg.norm(w) * (1 - jnp.linalg.norm(quaternion)), w]
            )
        )

        return jnp.vstack(qd)
