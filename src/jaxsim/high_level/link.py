import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim.high_level
import jaxsim.parsers.descriptions as descriptions
import jaxsim.sixd as sixd
import jaxsim.typing as jtp
from jaxsim.physics.algos.jacobian import jacobian
from jaxsim.utils import JaxsimDataclass

from .common import VelRepr


@jax_dataclasses.pytree_dataclass
class Link(JaxsimDataclass):
    """
    High-level class to operate on a single link of a simulated model.
    """

    link_description: descriptions.LinkDescription = jax_dataclasses.static_field()
    parent_model: "jaxsim.high_level.model.Model" = jax_dataclasses.field(
        default=None, repr=False, compare=False
    )

    def valid(self) -> bool:
        return self.parent_model is not None

    # ==========
    # Properties
    # ==========

    def name(self) -> str:
        return self.link_description.name

    def index(self) -> int:
        return self.link_description.index

    # ========
    # Dynamics
    # ========

    def mass(self) -> jtp.Float:
        return self.link_description.mass

    def spatial_inertia(self) -> jtp.Matrix:
        return self.link_description.inertia

    def com_position(self, in_link_frame: bool = True) -> jtp.VectorJax:
        from jaxsim.math.inertia import Inertia

        _, L_p_CoM, _ = Inertia.to_params(M=self.spatial_inertia())

        if in_link_frame:
            return L_p_CoM.squeeze()

        W_H_L = self.transform()
        W_ph_CoM = W_H_L @ jnp.hstack([L_p_CoM.squeeze(), 1])

        return W_ph_CoM[0:3].squeeze()

    # ==========
    # Kinematics
    # ==========

    def position(self) -> jtp.Vector:
        return self.transform()[0:3, 3]

    def orientation(self, dcm: bool = False) -> jtp.Vector:
        R = self.transform()[0:3, 0:3]

        to_wxyz = np.array([3, 0, 1, 2])
        return R if dcm else sixd.so3.SO3.from_matrix(R).as_quaternion_xyzw()[to_wxyz]

    def transform(self) -> jtp.Matrix:
        return self.parent_model.forward_kinematics()[self.index()]

    def velocity(self, vel_repr: VelRepr = None) -> jtp.Vector:
        v_WL = (
            self.jacobian(output_vel_repr=vel_repr)
            @ self.parent_model.generalized_velocity()
        )
        return v_WL

    def linear_velocity(self, vel_repr: VelRepr = None) -> jtp.Vector:
        return self.velocity(vel_repr=vel_repr)[0:3]

    def angular_velocity(self, vel_repr: VelRepr = None) -> jtp.Vector:
        return self.velocity(vel_repr=vel_repr)[3:6]

    def jacobian(self, output_vel_repr: VelRepr = None) -> jtp.Matrix:
        if output_vel_repr is None:
            output_vel_repr = self.parent_model.velocity_representation

        # Return the doubly left-trivialized free-floating jacobian
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
            B_T_W = jnp.block([[B_X_W, zero_6n], [zero_6n.T, jnp.eye(dofs)]])
            L_J_WL_target = L_J_WL_B @ B_T_W

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            dofs = self.parent_model.dofs()
            W_H_B = self.parent_model.base_transform()
            BW_H_B = jnp.array(W_H_B).at[0:3, 3].set(jnp.zeros(3))

            B_X_BW = sixd.se3.SE3.from_matrix(BW_H_B).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, dofs))
            B_T_BW = jnp.block([[B_X_BW, zero_6n], [zero_6n.T, jnp.eye(dofs)]])

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

    def external_force(self) -> jtp.Vector:
        W_f_ext = self.parent_model.data.model_input.f_ext[self.index()]

        if self.parent_model.velocity_representation is VelRepr.Inertial:
            return W_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Body:
            W_H_B = self.parent_model.base_transform()
            W_X_B = sixd.se3.SE3.from_matrix(W_H_B).adjoint()

            return W_X_B.transpose() @ W_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            raise NotImplementedError

        else:
            raise ValueError(self.parent_model.velocity_representation)

    def add_external_force(
        self, force: jtp.Array = None, torque: jtp.Array = None
    ) -> None:
        force = force if force is not None else jnp.zeros(3)
        torque = torque if torque is not None else jnp.zeros(3)

        f_ext = jnp.hstack([force, torque])

        if self.parent_model.velocity_representation is VelRepr.Inertial:
            W_f_ext = f_ext

        elif self.parent_model.velocity_representation is VelRepr.Body:
            L_f_ext = f_ext
            W_H_L = self.transform()
            L_X_W = sixd.se3.SE3.from_matrix(W_H_L).inverse().adjoint()

            W_f_ext = L_X_W.transpose() @ L_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            LW_f_ext = f_ext

            W_p_L = self.transform()[0:3, 3]
            W_H_LW = jnp.eye(4).at[0:3, 3].set(W_p_L)
            LW_X_W = sixd.se3.SE3.from_matrix(W_H_LW).inverse().adjoint()

            W_f_ext = LW_X_W @ LW_f_ext

        else:
            raise ValueError(self.parent_model.velocity_representation)

        W_f_ext_current = self.parent_model.data.model_input.f_ext[self.index(), :]

        self.parent_model.data.model_input.f_ext = (
            self.parent_model.data.model_input.f_ext.at[self.index(), :].set(
                W_f_ext_current + W_f_ext
            )
        )

    def add_com_external_force(
        self, force: jtp.Array = None, torque: jtp.Array = None
    ) -> None:
        force = force if force is not None else jnp.zeros(3)
        torque = torque if torque is not None else jnp.zeros(3)

        f_ext = jnp.hstack([force, torque])

        if self.parent_model.velocity_representation is VelRepr.Inertial:
            W_f_ext = f_ext

        elif self.parent_model.velocity_representation is VelRepr.Body:
            GL_f_ext = f_ext

            W_H_L = self.transform()
            L_p_CoM = self.com_position(in_link_frame=True)
            L_H_GL = jnp.eye(4).at[0:3, 3].set(L_p_CoM)
            W_H_GL = W_H_L @ L_H_GL
            GL_X_W = sixd.se3.SE3.from_matrix(W_H_GL).inverse().adjoint()

            W_f_ext = GL_X_W.transpose() @ GL_f_ext

        elif self.parent_model.velocity_representation is VelRepr.Mixed:
            GW_f_ext = f_ext

            W_p_CoM = self.com_position(in_link_frame=False)
            W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            GW_X_W = sixd.se3.SE3.from_matrix(W_H_GW).inverse().adjoint()

            W_f_ext = GW_X_W.transpose() @ GW_f_ext

        else:
            raise ValueError(self.parent_model.velocity_representation)

        W_f_ext_current = self.parent_model.data.model_input.f_ext[self.index(), :]

        self.parent_model.data.model_input.f_ext = (
            self.parent_model.data.model_input.f_ext.at[self.index(), :].set(
                W_f_ext_current + W_f_ext
            )
        )

    def in_contact(self) -> jtp.Bool:
        return not jnp.allclose(self.external_force(), 0)
