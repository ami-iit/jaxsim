import jax.lax
import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.sixd import so3


class Quaternion:
    @staticmethod
    def to_xyzw(wxyz: jtp.Vector) -> jtp.Vector:
        return wxyz.squeeze()[jnp.array([1, 2, 3, 0])]

    @staticmethod
    def to_wxyz(xyzw: jtp.Vector) -> jtp.Vector:
        return xyzw.squeeze()[jnp.array([3, 0, 1, 2])]

    @staticmethod
    def to_dcm(quaternion: jtp.Vector) -> jtp.Matrix:
        return so3.SO3.from_quaternion_xyzw(
            xyzw=Quaternion.to_xyzw(quaternion)
        ).as_matrix()

    @staticmethod
    def from_dcm(dcm: jtp.Matrix) -> jtp.Vector:
        return Quaternion.to_wxyz(
            xyzw=so3.SO3.from_matrix(matrix=dcm).as_quaternion_xyzw()
        )

    @staticmethod
    def derivative(
        quaternion: jtp.Vector,
        omega: jtp.Vector,
        omega_in_body_fixed: bool = False,
        K: float = 0.1,
    ) -> jtp.Vector:
        w = omega.squeeze()
        quaternion = quaternion.squeeze()

        def Q_body(q: jtp.Vector) -> jtp.Matrix:
            qw, qx, qy, qz = q

            return jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, -qz, qy],
                    [qy, qz, qw, -qx],
                    [qz, -qy, qx, qw],
                ]
            )

        def Q_inertial(q: jtp.Vector) -> jtp.Matrix:
            qw, qx, qy, qz = q

            return jnp.array(
                [
                    [qw, -qx, -qy, -qz],
                    [qx, qw, qz, -qy],
                    [qy, -qz, qw, qx],
                    [qz, qy, -qx, qw],
                ]
            )

        Q = jax.lax.cond(
            pred=omega_in_body_fixed,
            true_fun=Q_body,
            false_fun=Q_inertial,
            operand=quaternion,
        )

        qd = 0.5 * (
            Q
            @ jnp.hstack(
                [K * jnp.linalg.norm(w) * (1 - jnp.linalg.norm(quaternion)), w]
            )
        )

        return jnp.vstack(qd)
