from typing import Tuple, Union

import jax.numpy as jnp

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import JointDescriptor, JointGenericAxis, JointType

from .adjoint import Adjoint
from .plucker import Plucker
from .rotation import Rotation


def jcalc(
    jtyp: Union[JointType, JointDescriptor], q: float
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
        Xj = Plucker.from_rot_and_trans(
            dcm=Rotation.from_axis_angle(vector=(q * jtyp.axis)),
            translation=jnp.zeros(3),
        )
        S = jnp.vstack(jnp.hstack([jtyp.axis.squeeze(), jnp.zeros(3)]))

    elif code is JointType.P:

        jtyp: JointGenericAxis
        Xj = Adjoint.translate(direction=(q * jtyp.axis))
        S = jnp.vstack(jnp.hstack([jnp.zeros(3), jtyp.axis.squeeze()]))

    elif code is JointType.Rx:

        Xj = Adjoint.rotate_x(theta=q)
        S = jnp.vstack([1.0, 0, 0, 0, 0, 0])

    elif code is JointType.Ry:

        Xj = Adjoint.rotate_y(theta=q)
        S = jnp.vstack([0, 1.0, 0, 0, 0, 0])

    elif code is JointType.Rz:

        Xj = Adjoint.rotate_z(theta=q)
        S = jnp.vstack([0, 0, 1.0, 0, 0, 0])

    elif code is JointType.Px:

        Xj = Adjoint.translate(direction=jnp.hstack([q, 0.0, 0.0]))
        S = jnp.vstack([0, 0, 0, 1.0, 0, 0])

    elif code is JointType.Py:

        Xj = Adjoint.translate(direction=jnp.hstack([0.0, q, 0.0]))
        S = jnp.vstack([0, 0, 0, 0, 1.0, 0])

    elif code is JointType.Pz:

        Xj = Adjoint.translate(direction=jnp.hstack([0.0, 0.0, q]))
        S = jnp.vstack([0, 0, 0, 0, 0, 1.0])

    else:
        raise ValueError(code)

    return Xj, S
