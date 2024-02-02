from typing import Tuple, Union

import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import JointDescriptor, JointGenericAxis, JointType

from .adjoint import Adjoint
from .rotation import Rotation


def jcalc(
    jtyp: Union[JointType, JointDescriptor], q: jtp.Float
) -> Tuple[jtp.Matrix, jtp.Vector]:
    """
    Compute the spatial transformation matrix and motion subspace vector for a joint.

    Args:
        jtyp (Union[JointType, JointDescriptor]): The type or descriptor of the joint.
        q (jtp.Float): The joint configuration parameter.

    Returns:
        Tuple[jtp.Matrix, jtp.Vector]: A tuple containing the spatial transformation matrix (6x6) and the motion subspace vector (6x1).

    Raises:
        ValueError: If the joint type or descriptor is not recognized.
    """
    if isinstance(jtyp, JointType):
        code = jtyp
    elif isinstance(jtyp, JointDescriptor):
        code = jtyp.code
    else:
        raise ValueError(jtyp)

    match code:
        case JointType.F:
            raise ValueError("Fixed joints shouldn't be here")

        case JointType.R:
            jtyp: JointGenericAxis

            Xj = Adjoint.from_rotation_and_translation(
                rotation=Rotation.from_axis_angle(vector=q * jtyp.axis), inverse=True
            )

            S = jnp.vstack(jnp.hstack([jnp.zeros(3), jtyp.axis.squeeze()]))

        case JointType.P:
            jtyp: JointGenericAxis

            Xj = Adjoint.from_rotation_and_translation(
                translation=jnp.array(q * jtyp.axis), inverse=True
            )

            S = jnp.vstack(jnp.hstack([jtyp.axis.squeeze(), jnp.zeros(3)]))

        case JointType.Rx:
            Xj = Adjoint.from_rotation_and_translation(
                rotation=Rotation.x(theta=q), inverse=True
            )

            S = jnp.vstack([0, 0, 0, 1.0, 0, 0])

        case JointType.Ry:
            Xj = Adjoint.from_rotation_and_translation(
                rotation=Rotation.y(theta=q), inverse=True
            )

            S = jnp.vstack([0, 0, 0, 0, 1.0, 0])

        case JointType.Rz:
            Xj = Adjoint.from_rotation_and_translation(
                rotation=Rotation.z(theta=q), inverse=True
            )

            S = jnp.vstack([0, 0, 0, 0, 0, 1.0])

        case JointType.Px:
            Xj = Adjoint.from_rotation_and_translation(
                translation=jnp.array([q, 0.0, 0.0]), inverse=True
            )

            S = jnp.vstack([1.0, 0, 0, 0, 0, 0])

        case JointType.Py:
            Xj = Adjoint.from_rotation_and_translation(
                translation=jnp.array([0.0, q, 0.0]), inverse=True
            )

            S = jnp.vstack([0, 1.0, 0, 0, 0, 0])

        case JointType.Pz:
            Xj = Adjoint.from_rotation_and_translation(
                translation=jnp.array([0.0, 0.0, q]), inverse=True
            )

            S = jnp.vstack([0, 0, 1.0, 0, 0, 0])

        case _:
            raise ValueError(code)

    return Xj, S
