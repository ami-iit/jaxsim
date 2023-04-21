from typing import Tuple, Union

import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import JointDescriptor, JointGenericAxis, JointType

from .adjoint import Adjoint
from .rotation import Rotation


def jcalc(
    jtyp: Union[JointType, JointDescriptor], q: jtp.Float
) -> Tuple[jtp.Matrix, jtp.Vector]:
    if isinstance(jtyp, JointType):
        code = jtyp
    elif isinstance(jtyp, JointDescriptor):
        code = jtyp.code
    else:
        raise ValueError(jtyp)

    if code is JointType.F:
        raise ValueError("Fixed joints shouldn't be here")

    elif code is JointType.R:
        jtyp: JointGenericAxis

        Xj = Adjoint.from_rotation_and_translation(
            rotation=Rotation.from_axis_angle(vector=(q * jtyp.axis)), inverse=True
        )

        S = jnp.vstack(jnp.hstack([jnp.zeros(3), jtyp.axis.squeeze()]))

    elif code is JointType.P:
        jtyp: JointGenericAxis

        Xj = Adjoint.from_rotation_and_translation(
            translation=jnp.array(q * jtyp.axis), inverse=True
        )

        S = jnp.vstack(jnp.hstack([jtyp.axis.squeeze(), jnp.zeros(3)]))

    elif code is JointType.Rx:
        Xj = Adjoint.from_rotation_and_translation(
            rotation=Rotation.x(theta=q), inverse=True
        )

        S = jnp.vstack([0, 0, 0, 1.0, 0, 0])

    elif code is JointType.Ry:
        Xj = Adjoint.from_rotation_and_translation(
            rotation=Rotation.y(theta=q), inverse=True
        )

        S = jnp.vstack([0, 0, 0, 0, 1.0, 0])

    elif code is JointType.Rz:
        Xj = Adjoint.from_rotation_and_translation(
            rotation=Rotation.z(theta=q), inverse=True
        )

        S = jnp.vstack([0, 0, 0, 0, 0, 1.0])

    elif code is JointType.Px:
        Xj = Adjoint.from_rotation_and_translation(
            translation=jnp.array([q, 0.0, 0.0]), inverse=True
        )

        S = jnp.vstack([1.0, 0, 0, 0, 0, 0])

    elif code is JointType.Py:
        Xj = Adjoint.from_rotation_and_translation(
            translation=jnp.array([0.0, q, 0.0]), inverse=True
        )

        S = jnp.vstack([0, 1.0, 0, 0, 0, 0])

    elif code is JointType.Pz:
        Xj = Adjoint.from_rotation_and_translation(
            translation=jnp.array([0.0, 0.0, q]), inverse=True
        )

        S = jnp.vstack([0, 0, 1.0, 0, 0, 0])

    else:
        raise ValueError(code)

    return Xj, S
