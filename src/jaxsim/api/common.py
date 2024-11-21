import abc
import contextlib
import dataclasses
import enum
import functools
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.math import Adjoint
from jaxsim.utils import JaxsimDataclass, Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


_P = ParamSpec("_P")
_R = TypeVar("_R")


def named_scope(fn, name: str | None = None) -> Callable[_P, _R]:
    """Apply a JAX named scope to a function for improved profiling and clarity."""

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with jax.named_scope(name or fn.__name__):
            return fn(*args, **kwargs)

    return wrapper


@enum.unique
class VelRepr(enum.IntEnum):
    """
    Enumeration of all supported 6D velocity representations.
    """

    Body = enum.auto()
    Mixed = enum.auto()
    Inertial = enum.auto()


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
    ) -> Iterator[Self]:
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
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def inertial_to_other_representation(
        array: jtp.Array,
        other_representation: VelRepr,
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

        W_array = array.reshape(-1, 6)
        W_H_O = transform.reshape(-1, 4, 4)

        match other_representation:

            case VelRepr.Inertial:
                return W_array.reshape(array.shape[:-1] + (6,))

            case VelRepr.Body:

                if not is_force:
                    O_Xv_W = Adjoint.from_transform(transform=W_H_O, inverse=True)
                    O_array = jnp.einsum("bij,bj->bi", O_Xv_W, W_array)

                else:
                    O_Xf_W = Adjoint.from_transform(transform=W_H_O)
                    O_array = jnp.einsum(
                        "bij,bj->bi", O_Xf_W.transpose(0, 2, 1), W_array
                    )

                return O_array.reshape(array.shape[:-1] + (6,))

            case VelRepr.Mixed:
                W_p_O = W_H_O[:, 0:3, 3]
                W_H_OW = (
                    jnp.array([jnp.eye(4)] * W_H_O.shape[0]).at[:, 0:3, 3].set(W_p_O)
                )

                if not is_force:
                    OW_Xv_W = Adjoint.from_transform(transform=W_H_OW, inverse=True)
                    OW_array = jnp.einsum("bij,bj->bi", OW_Xv_W, W_array)

                else:
                    OW_Xf_W = Adjoint.from_transform(transform=W_H_OW)
                    OW_array = jnp.einsum(
                        "bij,bj->bi", OW_Xf_W.transpose(0, 2, 1), W_array
                    )

                return OW_array.reshape(array.shape[:-1] + (6,))

            case _:
                raise ValueError(other_representation)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["other_representation", "is_force"])
    def other_representation_to_inertial(
        array: jtp.Array,
        other_representation: VelRepr,
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

        O_array = array.reshape(-1, 6)
        W_H_O = transform.reshape(-1, 4, 4)

        match other_representation:
            case VelRepr.Inertial:
                return O_array.reshape(array.shape[:-1] + (6,))

            case VelRepr.Body:

                if not is_force:
                    W_Xv_O = Adjoint.from_transform(W_H_O)
                    W_array = jnp.einsum("bij,bj->bi", W_Xv_O, O_array)

                else:
                    W_Xf_O = Adjoint.from_transform(transform=W_H_O, inverse=True)
                    W_array = jnp.einsum(
                        "bij,bj->bi", W_Xf_O.transpose(0, 2, 1), O_array
                    )

                return W_array.reshape(array.shape[:-1] + (6,))

            case VelRepr.Mixed:

                W_p_O = W_H_O[:, 0:3, 3]
                W_H_OW = (
                    jnp.array([jnp.eye(4)] * W_H_O.shape[0]).at[:, 0:3, 3].set(W_p_O)
                )

                if not is_force:
                    W_Xv_BW = Adjoint.from_transform(W_H_OW)
                    W_array = jnp.einsum("bij,bj->bi", W_Xv_BW, O_array)

                else:
                    W_Xf_BW = Adjoint.from_transform(transform=W_H_OW, inverse=True)
                    W_array = jnp.einsum(
                        "bij,bj->bi", W_Xf_BW.transpose(0, 2, 1), O_array
                    )

                return W_array.reshape(array.shape[:-1] + (6,))

            case _:
                raise ValueError(other_representation)
