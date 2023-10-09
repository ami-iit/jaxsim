import dataclasses
import functools
from typing import Any, Tuple

import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.parsers
import jaxsim.typing as jtp
from jaxsim.utils import Vmappable, not_tracing, oop


@jax_dataclasses.pytree_dataclass
class Joint(Vmappable):
    """
    High-level class to operate in r/o on a single joint of a simulated model.
    """

    joint_description: Static[jaxsim.parsers.descriptions.JointDescription]

    _parent_model: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def parent_model(self) -> "jaxsim.high_level.model.Model":
        """"""

        return self._parent_model

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def valid(self) -> jtp.Bool:
        """"""

        return jnp.array(self.parent_model is not None, dtype=bool)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def index(self) -> jtp.Int:
        """"""

        return jnp.array(self.joint_description.index, dtype=int)

    @functools.partial(oop.jax_tf.method_ro)
    def dofs(self) -> jtp.Int:
        """"""

        return jnp.array(1, dtype=int)

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def name(self) -> str:
        """"""

        return self.joint_description.name

    @functools.partial(oop.jax_tf.method_ro)
    def position(self, dof: int = None) -> jtp.Float:
        """"""

        dof = dof if dof is not None else 0

        return jnp.array(
            self.parent_model.joint_positions(joint_names=(self.name(),))[dof],
            dtype=float,
        )

    @functools.partial(oop.jax_tf.method_ro)
    def velocity(self, dof: int = None) -> jtp.Float:
        """"""

        dof = dof if dof is not None else 0

        return jnp.array(
            self.parent_model.joint_velocities(joint_names=(self.name(),))[dof],
            dtype=float,
        )

    @functools.partial(oop.jax_tf.method_ro)
    def acceleration(self, dof: int = None) -> jtp.Float:
        """"""

        dof = dof if dof is not None else 0

        return jnp.array(
            self.parent_model.joint_accelerations(joint_names=[self.name()])[dof],
            dtype=float,
        )

    @functools.partial(oop.jax_tf.method_ro)
    def force(self, dof: int = None) -> jtp.Float:
        """"""

        dof = dof if dof is not None else 0

        return jnp.array(
            self.parent_model.joint_generalized_forces(joint_names=(self.name(),))[dof],
            dtype=float,
        )

    @functools.partial(oop.jax_tf.method_ro)
    def position_limit(self, dof: int = None) -> Tuple[jtp.Float, jtp.Float]:
        """"""

        dof = dof if dof is not None else 0

        if not_tracing(dof) and dof != 0:
            msg = "Only joints with 1 DoF are currently supported"
            raise ValueError(msg)

        low, high = self.joint_description.position_limit

        return jnp.array(low, dtype=float), jnp.array(high, dtype=float)

    # =================
    # Multi-DoF methods
    # =================

    @functools.partial(oop.jax_tf.method_ro)
    def joint_position(self) -> jtp.Vector:
        """"""

        return self.parent_model.joint_positions(joint_names=(self.name(),))

    @functools.partial(oop.jax_tf.method_ro)
    def joint_velocity(self) -> jtp.Vector:
        """"""

        return self.parent_model.joint_velocities(joint_names=(self.name(),))

    @functools.partial(oop.jax_tf.method_ro)
    def joint_acceleration(self) -> jtp.Vector:
        """"""

        return self.parent_model.joint_accelerations(joint_names=[self.name()])

    @functools.partial(oop.jax_tf.method_ro)
    def joint_force(self) -> jtp.Vector:
        """"""

        return self.parent_model.joint_generalized_forces(joint_names=(self.name(),))
