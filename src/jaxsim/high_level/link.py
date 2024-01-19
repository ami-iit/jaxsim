import dataclasses
import functools
from typing import Any

import jax.lax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
from jax_dataclasses import Static

import jaxsim.parsers
import jaxsim.typing as jtp
from jaxsim import sixd
from jaxsim.physics.algos.jacobian import jacobian
from jaxsim.utils import Vmappable, oop

from .common import VelRepr


@jax_dataclasses.pytree_dataclass
class Link(Vmappable):
    """
    High-level class to operate in r/o on a single link of a simulated model.
    """

    link_description: Static[jaxsim.parsers.descriptions.LinkDescription]

    _parent_model: Any = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    @property
    def parent_model(self) -> "jaxsim.high_level.model.Model":
        """"""

        return self._parent_model

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def valid(self) -> jtp.Bool:
        """"""

        return jnp.array(self.parent_model is not None, dtype=bool)

    # ==========
    # Properties
    # ==========

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def name(self) -> str:
        """"""

        return self.link_description.name

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def index(self) -> jtp.Int:
        """"""

        return jnp.array(self.link_description.index, dtype=int)

    # ========
    # Dynamics
    # ========

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def mass(self) -> jtp.Float:
        """"""

        return jnp.array(self.link_description.mass, dtype=float)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def spatial_inertia(self) -> jtp.Matrix:
        """"""

        return jnp.array(self.link_description.inertia, dtype=float)

    @functools.partial(oop.jax_tf.method_ro, vmap_in_axes=(0, None))
    def com_position(self, in_link_frame: bool = True) -> jtp.Vector:
        """"""

        from jaxsim.math.inertia import Inertia

        _, L_p_CoM, _ = Inertia.to_params(M=self.spatial_inertia())

        def com_in_link_frame():
            return L_p_CoM.squeeze()

        def com_in_inertial_frame():
            W_H_L = self.transform()
            W_p̃_CoM = W_H_L @ jnp.hstack([L_p_CoM.squeeze(), 1])

            return W_p̃_CoM[0:3].squeeze()

        return jax.lax.select(
            pred=in_link_frame,
            on_true=com_in_link_frame(),
            on_false=com_in_inertial_frame(),
        )

    # ==========
    # Kinematics
    # ==========

    @functools.partial(oop.jax_tf.method_ro)
    def position(self) -> jtp.Vector:
        """"""

        return self.transform()[0:3, 3]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["dcm"])
    def orientation(self, dcm: bool = False) -> jtp.Vector:
        """"""

        R = self.transform()[0:3, 0:3]

        to_wxyz = np.array([3, 0, 1, 2])
        return R if dcm else sixd.so3.SO3.from_matrix(R).as_quaternion_xyzw()[to_wxyz]

    @functools.partial(oop.jax_tf.method_ro)
    def transform(self) -> jtp.Matrix:
        """"""

        return self.parent_model.forward_kinematics()[self.index()]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["vel_repr"])
    def velocity(self, vel_repr: VelRepr | None = None) -> jtp.Vector:
        """"""

        v_WL = (
            self.jacobian(output_vel_repr=vel_repr)
            @ self.parent_model.generalized_velocity()
        )

        return v_WL

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["vel_repr"])
    def linear_velocity(self, vel_repr: VelRepr | None = None) -> jtp.Vector:
        """"""

        return self.velocity(vel_repr=vel_repr)[0:3]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["vel_repr"])
    def angular_velocity(self, vel_repr: VelRepr | None = None) -> jtp.Vector:
        """"""

        return self.velocity(vel_repr=vel_repr)[3:6]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["output_vel_repr"])
    def jacobian(self, output_vel_repr: VelRepr | None = None) -> jtp.Matrix:
        """"""

        if output_vel_repr is None:
            output_vel_repr = self.parent_model.velocity_representation

        # Compute the doubly left-trivialized free-floating jacobian
        L_J_WL_B = jacobian(
            model=self.parent_model.physics_model,
            body_index=self.index(),
            q=self.parent_model.data.model_state.joint_positions,
        )

        if self.parent_model.velocity_representation is VelRepr.Body:
            L_J_WL_target = L_J_WL_B

        elif self.parent_model.velocity_representation is VelRepr.Inertial:
            dofs = self.parent_model.dofs()
            W_H_B = self.parent_model.base_transform()

            B_X_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, dofs))

            B_T_W = jnp.vstack(
                [
                    jnp.block([B_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(dofs)]),
                ]
            )

            L_J_WL_target = L_J_WL_B @ B_T_W

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            dofs = self.parent_model.dofs()
            W_H_B = self.parent_model.base_transform()
            BW_H_B = jnp.array(W_H_B).at[0:3, 3].set(jnp.zeros(3))

            B_X_BW = sixd.se3.SE3.from_matrix(BW_H_B).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, dofs))

            B_T_BW = jnp.vstack(
                [
                    jnp.block([B_X_BW, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(dofs)]),
                ]
            )

            L_J_WL_target = L_J_WL_B @ B_T_BW

        else:
            raise ValueError(self.parent_model.velocity_representation)

        if output_vel_repr is VelRepr.Body:
            return L_J_WL_target

        elif output_vel_repr is VelRepr.Inertial:
            W_H_L = self.transform()
            W_X_L = sixd.se3.SE3.from_matrix(W_H_L).adjoint()
            return W_X_L @ L_J_WL_target

        elif output_vel_repr is VelRepr.Mixed:
            W_H_L = self.transform()
            LW_H_L = jnp.array(W_H_L).at[0:3, 3].set(jnp.zeros(3))
            LW_X_L = sixd.se3.SE3.from_matrix(LW_H_L).adjoint()
            return LW_X_L @ L_J_WL_target

        else:
            raise ValueError(output_vel_repr)

    @functools.partial(oop.jax_tf.method_ro)
    def external_force(self) -> jtp.Vector:
        """
        Return the active external force acting on the link.

        This external force is a user input and is not computed by the physics engine.
        During the simulation, this external force is summed to other terms like those
        related to enforce contact constraints.

        Returns:
            The active external 6D force acting on the link in the active representation.
        """

        # Get the external force stored in the inertial representation
        W_f_ext = self.parent_model.data.model_input.f_ext[self.index()]

        # Express it in the active representation
        if self.parent_model.velocity_representation is VelRepr.Inertial:
            f_ext = W_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Body:
            W_H_L = self.transform()
            W_X_L = sixd.se3.SE3.from_matrix(W_H_L).adjoint()

            f_ext = L_f_ext = W_X_L.transpose() @ W_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            W_p_L = self.transform()[0:3, 3]
            W_H_LW = jnp.eye(4).at[0:3, 3].set(W_p_L)
            W_X_LW = sixd.se3.SE3.from_matrix(W_H_LW).adjoint()

            f_ext = LW_f_ext = W_X_LW.transpose() @ W_f_ext

        else:
            raise ValueError(self.parent_model.velocity_representation)

        return f_ext

    @functools.partial(oop.jax_tf.method_ro)
    def in_contact(self) -> jtp.Bool:
        """"""

        return self.parent_model.in_contact()[self.index()]
