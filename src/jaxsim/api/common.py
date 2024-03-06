import abc
import contextlib
import dataclasses
import functools
from typing import ContextManager

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.high_level.common import VelRepr
from jaxsim.utils import JaxsimDataclass, Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class ModelDataWithVelocityRepresentation(JaxsimDataclass, abc.ABC):
    """
    Base class for model data structures with velocity representation.
    """

    velocity_representation: Static[VelRepr] = dataclasses.field(
        default=VelRepr.Inertial, kw_only=True
    )

    @contextlib.contextmanager
    def switch_velocity_representation(
        self, velocity_representation: VelRepr
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

            # First, we replace the velocity representation
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = velocity_representation

            # Then, we yield the data with changed representation.
            # We run this in a mutable context with restoration so that any exception
            # occurring, we restore the original object in case it was modified.
            with self.mutable_context(
                mutability=self._mutability(), restore_after_exception=True
            ):
                yield self

        finally:
            with self.mutable_context(
                mutability=Mutability.MUTABLE_NO_VALIDATION,
                restore_after_exception=True,
            ):
                self.velocity_representation = original_representation

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def inertial_to_other_representation(
        array: jtp.Array,
        other_representation: VelRepr,
        transform: jtp.Matrix,
        is_force: bool = False,
    ) -> jtp.Array:
        r"""
        Convert a 6D quantity from inertial-fixed to another representation.

        Args:
            array: The 6D quantity to convert.
            other_representation: The representation to convert to.
            transform:
                The `math:W \mathbf{H}_O` transform, where `math:O` is the
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

        match other_representation:

            case VelRepr.Inertial:
                return W_array

            case VelRepr.Body:

                if not is_force:
                    O_Xv_W = jaxlie.SE3.from_matrix(W_H_O).inverse().adjoint()
                    O_array = O_Xv_W @ W_array

                else:
                    O_Xf_W = jaxlie.SE3.from_matrix(W_H_O).adjoint().T
                    O_array = O_Xf_W @ W_array

                return O_array

            case VelRepr.Mixed:
                W_p_O = W_H_O[0:3, 3]
                W_H_OW = jnp.eye(4).at[0:3, 3].set(W_p_O)

                if not is_force:
                    OW_Xv_W = jaxlie.SE3.from_matrix(W_H_OW).inverse().adjoint()
                    OW_array = OW_Xv_W @ W_array

                else:
                    OW_Xf_W = jaxlie.SE3.from_matrix(W_H_OW).adjoint().transpose()
                    OW_array = OW_Xf_W @ W_array

                return OW_array

            case _:
                raise ValueError(other_representation)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def other_representation_to_inertial(
        array: jtp.Array,
        other_representation: VelRepr,
        transform: jtp.Matrix,
        is_force: bool = False,
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

        match other_representation:
            case VelRepr.Inertial:
                W_array = array
                return W_array

            case VelRepr.Body:
                O_array = array

                if not is_force:
                    W_Xv_O: jtp.Array = jaxlie.SE3.from_matrix(W_H_O).adjoint()
                    W_array = W_Xv_O @ O_array

                else:
                    W_Xf_O = jaxlie.SE3.from_matrix(W_H_O).inverse().adjoint().T
                    W_array = W_Xf_O @ O_array

                return W_array

            case VelRepr.Mixed:
                BW_array = array
                W_p_O = W_H_O[0:3, 3]
                W_H_OW = jnp.eye(4).at[0:3, 3].set(W_p_O)

                if not is_force:
                    W_Xv_BW: jtp.Array = jaxlie.SE3.from_matrix(W_H_OW).adjoint()
                    W_array = W_Xv_BW @ BW_array

                else:
                    W_Xf_BW = jaxlie.SE3.from_matrix(W_H_OW).inverse().adjoint().T
                    W_array = W_Xf_BW @ BW_array

                return W_array

            case _:
                raise ValueError(other_representation)
