import abc
import contextlib
import dataclasses
import functools
from typing import ClassVar, ContextManager

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.math import Adjoint
from jaxsim.utils import JaxsimDataclass, Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@dataclasses.dataclass(frozen=True)
class VelRepr:
    """
    Enumeration of all supported 6D velocity representations.
    """

    Body: ClassVar[int] = 0
    Mixed: ClassVar[int] = 1
    Inertial: ClassVar[int] = 2


@jax_dataclasses.pytree_dataclass
class ModelDataWithVelocityRepresentation(JaxsimDataclass, abc.ABC):
    """
    Base class for model data structures with velocity representation.
    """

    velocity_representation: jtp.VelRepr = dataclasses.field(
        default=VelRepr.Inertial, kw_only=True
    )

    @contextlib.contextmanager
    def switch_velocity_representation(
        self, velocity_representation: jtp.VelRepr
    ) -> ContextManager[Self]:
        """
        Context manager to temporarily switch the velocity representation.

        Args:
            velocity_representation: The new velocity representation.

        Yields:
            The same object with the new velocity representation.
        """

        original_representation = self.velocity_representation

        try:

            # First, we replace the velocity representation.
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = velocity_representation

            # Then, we yield the data with changed representation.
            # We run this in a mutable context with restoration so that any exception
            # occurring, we restore the original object in case it was modified.
            with self.mutable_context(
                mutability=self.mutability(), restore_after_exception=True
            ):
                yield self

        finally:
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = original_representation

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["is_force"])
    def inertial_to_other_representation(
        array: jtp.Array,
        other_representation: jtp.VelRepr,
        transform: jtp.Matrix,
        *,
        is_force: bool,
    ) -> jtp.Array:
        r"""
        Convert a 6D quantity from inertial-fixed to another representation.

        Args:
            array: The 6D quantity to convert.
            other_representation: The representation to convert to.
            transform:
                The :math:`W \mathbf{H}_O` transform, where :math:`O` is the
                reference frame of the other representation.
            is_force: Whether the quantity is a 6D force or a 6D velocity.

        Returns:
            The 6D quantity in the other representation.
        """

        W_array = array.squeeze()
        W_H_O = transform.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size, 6)

        if W_H_O.shape != (4, 4):
            raise ValueError(W_H_O.shape, (4, 4))

        def to_inertial():
            return W_array

        def to_body():
            if not is_force:
                O_Xv_W = Adjoint.from_transform(transform=W_H_O, inverse=True)
                O_array = O_Xv_W @ W_array
            else:
                O_Xf_W = Adjoint.from_transform(transform=W_H_O).T
                O_array = O_Xf_W @ W_array
            return O_array

        def to_mixed():
            W_p_O = W_H_O[0:3, 3]
            W_H_OW = jnp.eye(4).at[0:3, 3].set(W_p_O)
            if not is_force:
                OW_Xv_W = Adjoint.from_transform(transform=W_H_OW, inverse=True)
                OW_array = OW_Xv_W @ W_array
            else:
                OW_Xf_W = Adjoint.from_transform(transform=W_H_OW).T
                OW_array = OW_Xf_W @ W_array
            return OW_array

        return jax.lax.switch(
            index=other_representation,
            branches=(
                to_body,  # VelRepr.Body
                to_mixed,  # VelRepr.Mixed
                to_inertial,  # VelRepr.Inertial
            ),
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["is_force"])
    def other_representation_to_inertial(
        array: jtp.Array,
        other_representation: jtp.VelRepr,
        transform: jtp.Matrix,
        *,
        is_force: bool,
    ) -> jtp.Array:
        r"""
        Convert a 6D quantity from another representation to inertial-fixed.

        Args:
            array: The 6D quantity to convert.
            other_representation: The representation to convert from.
            transform:
                The `math:W \mathbf{H}_O` transform, where `math:O` is the
                reference frame of the other representation.
            is_force: Whether the quantity is a 6D force or a 6D velocity.

        Returns:
            The 6D quantity in the inertial-fixed representation.
        """

        W_array = array.squeeze()
        W_H_O = transform.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size, 6)

        if W_H_O.shape != (4, 4):
            raise ValueError(W_H_O.shape, (4, 4))

        def from_inertial():
            W_array = array
            return W_array

        def from_body():
            O_array = array

            if not is_force:
                W_Xv_O = Adjoint.from_transform(W_H_O)
                W_array = W_Xv_O @ O_array

            else:
                W_Xf_O = Adjoint.from_transform(transform=W_H_O, inverse=True).T
                W_array = W_Xf_O @ O_array

            return W_array

        def from_mixed():
            BW_array = array
            W_p_O = W_H_O[0:3, 3]
            W_H_OW = jnp.eye(4).at[0:3, 3].set(W_p_O)

            if not is_force:
                W_Xv_BW = Adjoint.from_transform(W_H_OW)
                W_array = W_Xv_BW @ BW_array

            else:
                W_Xf_BW = Adjoint.from_transform(transform=W_H_OW, inverse=True).T
                W_array = W_Xf_BW @ BW_array

            return W_array

        return jax.lax.switch(
            index=other_representation,
            branches=(
                from_body,  # VelRepr.Body
                from_mixed,  # VelRepr.Mixed
                from_inertial,  # VelRepr.Inertial
            ),
        )
