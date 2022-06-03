import dataclasses
import pathlib
from typing import Dict, List, NamedTuple, Tuple, Union

import jax.experimental.ode
import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim
import jaxsim.physics.algos.aba
import jaxsim.physics.algos.crba
import jaxsim.physics.algos.rnea
import jaxsim.physics.model.physics_model
import jaxsim.physics.model.physics_model_state
import jaxsim.typing as jtp
from jaxsim import high_level, physics, sixd
from jaxsim.parsers.sdf.utils import flip_velocity_serialization
from jaxsim.physics.algos import soft_contacts
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.simulation import ode_data

from .common import VelRepr


class ModelData(NamedTuple):

    model_state: jaxsim.physics.model.physics_model_state.PhysicsModelState
    model_input: jaxsim.physics.model.physics_model_state.PhysicsModelInput
    contact_state: jaxsim.physics.algos.soft_contacts.SoftContactsState

    @staticmethod
    def zero(physics_model: physics.model.physics_model.PhysicsModel) -> "ModelData":

        return ModelData(
            model_state=jaxsim.physics.model.physics_model_state.PhysicsModelState.zero(
                physics_model=physics_model
            ),
            model_input=jaxsim.physics.model.physics_model_state.PhysicsModelInput.zero(
                physics_model=physics_model
            ),
            contact_state=jaxsim.physics.algos.soft_contacts.SoftContactsState.zero(
                physics_model=physics_model
            ),
        )


@jax_dataclasses.pytree_dataclass
class Model:

    model_name: str
    physics_model: physics.model.physics_model.PhysicsModel = dataclasses.field(
        repr=False
    )

    velocity_representation: VelRepr = dataclasses.field(default=VelRepr.Mixed)

    _links: Dict[str, "high_level.link.Link"] = dataclasses.field(
        default_factory=list, repr=False
    )
    _joints: Dict[str, "high_level.joint.Joint"] = dataclasses.field(
        default_factory=list, repr=False
    )

    data: ModelData = jax_dataclasses.field(default=None, repr=False)

    # ========================
    # Initialization and state
    # ========================

    @staticmethod
    def build_from_sdf(
        sdf: Union[str, pathlib.Path],
        model_name: str = None,
        vel_repr: VelRepr = VelRepr.Mixed,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: bool = False,
    ) -> "Model":

        import jaxsim.parsers.sdf

        if is_urdf:
            raise ValueError("Converting URDF to SDF is not yet supported")

        model_description = jaxsim.parsers.sdf.build_model_from_sdf(sdf=sdf)
        physics_model = jaxsim.physics.model.physics_model.PhysicsModel.build_from(
            model_description=model_description, gravity=gravity
        )

        return Model.build(
            physics_model=physics_model,
            model_name=model_name,
            vel_repr=vel_repr,
        )

    @staticmethod
    def build(
        physics_model: jaxsim.physics.model.physics_model.PhysicsModel,
        model_name: str = None,
        vel_repr: VelRepr = VelRepr.Mixed,
    ) -> "Model":

        model_name = (
            model_name if model_name is not None else physics_model.description.name
        )

        model = Model(
            physics_model=physics_model,
            model_name=model_name,
            velocity_representation=vel_repr,
            _links={
                l.name: high_level.link.Link(link_description=l)
                for l in sorted(
                    physics_model.description.links_dict.values(), key=lambda l: l.index
                )
            },
            _joints={
                j.name: high_level.joint.Joint(joint_description=j)
                for j in sorted(
                    physics_model.description.joints_dict.values(),
                    key=lambda j: j.index,
                )
            },
        ).zero()

        if not model.valid():
            raise RuntimeError

        return model

    def __post_init__(self):

        for l in self._links.values():
            l.parent_model = self

        for j in self._joints.values():
            j.parent_model = self

        object.__setattr__(
            self,
            "_links",
            {
                k: v
                for k, v in sorted(self._links.items(), key=lambda kv: kv[1].index())
            },
        )
        object.__setattr__(
            self,
            "_joints",
            {
                k: v
                for k, v in sorted(self._joints.items(), key=lambda kv: kv[1].index())
            },
        )

    def zero(self) -> "Model":

        return jax_dataclasses.replace(
            self, data=ModelData.zero(physics_model=self.physics_model)  # noqa
        )

    def update_data(
        self,
        data: ModelData = None,
        model_state: jaxsim.physics.model.physics_model_state.PhysicsModelState = None,
        model_input: jaxsim.physics.model.physics_model_state.PhysicsModelInput = None,
        contact_state: jaxsim.physics.algos.soft_contacts.SoftContactsState = None,
        validate: bool = True,
        **kwargs,
    ) -> "Model":

        data = data if data is not None else self.data
        model_state = model_state if model_state is not None else data.model_state
        model_input = model_input if model_input is not None else data.model_input
        contact_state = (
            contact_state if contact_state is not None else data.contact_state
        )

        def has_field(dc, field_name: str) -> bool:
            return field_name in {field.name for field in jax_dataclasses.fields(dc)}

        def replace_fields(dc, **kwargs):

            fields = {key: value for key, value in kwargs.items() if has_field(dc, key)}

            with jax_dataclasses.copy_and_mutate(dc, validate=validate) as updated_dc:
                _ = [updated_dc.__setattr__(k, v) for k, v in fields.items()]

            return updated_dc

        for key in kwargs.keys():
            if not any(
                [
                    has_field(model_state, key),
                    has_field(model_input, key),
                    has_field(contact_state, key),
                ]
            ):
                raise ValueError(f"Failed to replace invalid field '{key}'")

        model_state = replace_fields(model_state, **kwargs)
        model_input = replace_fields(model_input, **kwargs)
        contact_state = replace_fields(contact_state, **kwargs)

        update_data = data._replace(
            model_state=model_state,
            model_input=model_input,
            contact_state=contact_state,
        )

        with jax_dataclasses.copy_and_mutate(self) as model:

            model.data = update_data

        return model

    def change_representation(self, vel_repr: VelRepr) -> "Model":

        if self.velocity_representation is vel_repr:
            return self

        with jax_dataclasses.copy_and_mutate(self) as model:
            model.velocity_representation = vel_repr

        return model

    # ==========
    # Properties
    # ==========

    def valid(self) -> bool:

        valid = True
        valid = valid and all([l.valid for l in self.links()])
        valid = valid and all([j.valid for j in self.joints()])
        return valid

    def floating_base(self) -> bool:

        return self.physics_model.is_floating_base

    def dofs(self) -> int:

        return self.physics_model.dofs()

    def name(self) -> str:

        return self.model_name

    def nr_of_links(self) -> int:

        return len(self._links)

    def nr_of_joints(self) -> int:

        return len(self._joints)

    def total_mass(self) -> float:

        return jnp.sum(jnp.array([l.mass() for l in self.links()]))

    def get_link(self, link_name: str) -> high_level.link.Link:

        if link_name not in self.link_names():
            msg = f"Link '{link_name}' is not part of model '{self.name()}'"
            raise ValueError(msg)

        return self.links(link_names=[link_name])[0]

    def get_joint(self, joint_name: str) -> high_level.joint.Joint:

        if joint_name not in self.joint_names():
            msg = f"Joint '{joint_name}' is not part of model '{self.name()}'"
            raise ValueError(msg)

        return self.joints(joint_names=[joint_name])[0]

    def link_names(self) -> List[str]:

        return list(self._links.keys())

    def joint_names(self) -> List[str]:

        return list(self._joints.keys())

    def links(self, link_names: List[str] = None) -> List[high_level.link.Link]:

        if link_names is None:
            return list(self._links.values())

        return [self._links[name] for name in link_names]

    def joints(self, joint_names: List[str] = None) -> List[high_level.joint.Joint]:

        if joint_names is None:
            return list(self._joints.values())

        return [self._joints[name] for name in joint_names]

    # ==================
    # Vectorized methods
    # ==================

    def joint_positions(self, joint_names: List[str] = None) -> jtp.Vector:

        if self.dofs() == 0 and (joint_names is None or len(joint_names) == 0):
            return jnp.array([])

        return self.data.model_state.joint_positions[
            self._joint_indices(joint_names=joint_names)
        ]

    def joint_random_positions(
        self,
        joint_names: List[str] = None,
        key: jax.random.PRNGKeyArray = jax.random.PRNGKey(seed=0),
    ) -> jtp.Vector:

        if self.dofs() == 0 and (joint_names is None or len(joint_names) == 0):
            return jnp.array([])

        s_min, s_max = self.joint_limits(joint_names=joint_names)

        s_random = jax.random.uniform(
            minval=s_min,
            maxval=s_max,
            key=key,
            shape=s_min.shape,
        )

        return s_random

    def joint_velocities(self, joint_names: List[str] = None) -> jtp.Vector:

        if self.dofs() == 0 and (joint_names is None or len(joint_names) == 0):
            return jnp.array([])

        return self.data.model_state.joint_velocities[
            self._joint_indices(joint_names=joint_names)
        ]

    def joint_limits(
        self, joint_names: List[str] = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:

        if self.dofs() == 0 and (joint_names is None or len(joint_names) == 0):
            return jnp.array([])

        joint_names = joint_names if joint_names is not None else self.joint_names()

        s_min = jnp.array(
            [min(self.get_joint(name).position_limit()) for name in joint_names]
        )

        s_max = jnp.array(
            [max(self.get_joint(name).position_limit()) for name in joint_names]
        )

        return s_min, s_max

    # =========
    # Base link
    # =========

    def base_frame(self) -> str:

        return self.physics_model.description.root.name

    def base_transform(self) -> jtp.MatrixJax:

        world_H_base = jnp.eye(4)
        world_H_base = world_H_base.at[0:3, 3].set(self.base_position())

        world_H_base = world_H_base.at[0:3, 0:3].set(self.base_orientation(dcm=True))

        return world_H_base

    def base_position(self) -> jtp.Vector:

        return self.data.model_state.base_position.squeeze()

    def base_orientation(self, dcm: bool = False) -> jtp.Vector:

        to_xyzw = np.array([1, 2, 3, 0])

        return (
            self.data.model_state.base_quaternion
            if not dcm
            else sixd.so3.SO3.from_quaternion_xyzw(
                self.data.model_state.base_quaternion[to_xyzw]
            ).as_matrix()
        )

    def base_velocity(self) -> jtp.Vector:

        # Get the base velocity stored in the model's data (inertial representation)
        W_velocity_WB = jnp.hstack(
            [
                self.data.model_state.base_linear_velocity,
                self.data.model_state.base_angular_velocity,
            ]
        )

        if self.velocity_representation is VelRepr.Inertial:

            base_velocity = W_velocity_WB

        elif self.velocity_representation is VelRepr.Body:

            B_X_W = (
                sixd.se3.SE3.from_rotation_and_translation(
                    rotation=sixd.so3.SO3.from_matrix(self.base_orientation(dcm=True)),
                    translation=self.base_position(),
                )
                .inverse()
                .adjoint()
            )

            base_velocity = B_X_W @ W_velocity_WB

        elif self.velocity_representation is VelRepr.Mixed:

            BW_X_W = (
                sixd.se3.SE3.from_rotation_and_translation(
                    rotation=sixd.so3.SO3.identity(),
                    translation=self.base_position(),
                )
                .inverse()
                .adjoint()
            )

            base_velocity = BW_X_W @ W_velocity_WB

        else:
            raise ValueError(self.velocity_representation)

        return base_velocity.squeeze()

    def external_forces(self) -> jtp.Matrix:

        f_ext_anglin = self.data.model_input.f_ext
        W_f_ext = jnp.hstack([f_ext_anglin[:, 3:6], f_ext_anglin[:, 0:3]])

        if self.velocity_representation is VelRepr.Inertial:

            return W_f_ext

        elif self.velocity_representation.Body:

            W_X_B = sixd.se3.SE3.from_matrix(self.base_transform()).adjoint()
            B_f_ext = (W_X_B.transpose() @ W_f_ext.T).T
            return B_f_ext

        elif self.velocity_representation.Mixed:

            W_H_BW = jnp.array(self.base_transform()).at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
            BW_f_ext = (W_X_BW.transpose() @ W_f_ext.T).T
            raise BW_f_ext

        else:
            raise ValueError(self.velocity_representation)

    # ==================
    # Dynamic properties
    # ==================

    def generalized_position(self) -> Tuple[jtp.Matrix, jtp.Vector]:

        return self.base_transform(), self.joint_positions()

    def generalized_velocity(self) -> jtp.Vector:

        return jnp.hstack([self.base_velocity(), self.joint_velocities()])

    def generalized_jacobian(self, output_vel_repr: VelRepr = None) -> jtp.Matrix:

        return jnp.vstack(
            [
                self.get_link(link_name=link_name).jacobian(
                    output_vel_repr=output_vel_repr
                )
                for link_name in self.link_names()
            ]
        )

    def free_floating_mass_matrix(self) -> jtp.Matrix:

        M_body_anglin = jaxsim.physics.algos.crba.crba(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
        )

        Mbb = flip_velocity_serialization(M_body_anglin[0:6, 0:6])
        Mbs_ang = M_body_anglin[0:3, 6:]
        Mbs_lin = M_body_anglin[3:6, 6:]
        Msb_ang = M_body_anglin[6:, 0:3]
        Msb_lin = M_body_anglin[6:, 3:6]

        M_body_linang = jnp.zeros_like(M_body_anglin)
        M_body_linang = M_body_linang.at[0:6, 0:6].set(Mbb)
        M_body_linang = M_body_linang.at[0:6, 6:].set(jnp.vstack([Mbs_lin, Mbs_ang]))
        M_body_linang = M_body_linang.at[6:, 0:6].set(jnp.hstack([Msb_lin, Msb_ang]))
        M_body_linang = M_body_linang.at[6:, 6:].set(M_body_anglin[6:, 6:])

        # This is M in body-fixed velocity representation
        M_body = M_body_linang

        if self.velocity_representation is VelRepr.Body:
            return M_body

        elif self.velocity_representation is VelRepr.Inertial:

            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            W_X_B = sixd.se3.SE3.from_matrix(self.base_transform()).adjoint()
            T = jnp.block([[W_X_B, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])
            invT = jnp.linalg.inv(T)
            return invT.T @ M_body @ invT

        elif self.velocity_representation is VelRepr.Mixed:

            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            BW_H_B = self.base_transform().at[0:3, 3].set(jnp.zeros(3))
            BW_X_B = sixd.se3.SE3.from_matrix(BW_H_B).adjoint()
            T = jnp.block([[BW_X_B, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])
            invT = jnp.linalg.inv(T)
            return invT.T @ M_body @ invT

        else:
            raise ValueError(self.velocity_representation)

    def free_floating_generalized_forces(self) -> jtp.Vector:

        model = self.zero().update_data(
            joint_positions=self.data.model_state.joint_positions,
            joint_velocities=self.data.model_state.joint_velocities,
            base_quaternion=self.data.model_state.base_quaternion,
            base_position=self.data.model_state.base_position,
            base_linear_velocity=self.data.model_state.base_linear_velocity,
            base_angular_velocity=self.data.model_state.base_angular_velocity,
        )
        return jnp.hstack(model.inverse_dynamics())

    def free_floating_gravity_forces(self) -> jtp.Vector:

        model = self.zero().update_data(
            joint_positions=self.data.model_state.joint_positions,
            base_quaternion=self.data.model_state.base_quaternion,
            base_position=self.data.model_state.base_position,
        )

        return jnp.hstack(model.inverse_dynamics())

    def momentum(self) -> jtp.Vector:

        m = self.change_representation(vel_repr=VelRepr.Body)
        B_h = m.free_floating_mass_matrix()[0:6, :] @ m.generalized_velocity()

        if self.velocity_representation is VelRepr.Body:
            return B_h

        elif self.velocity_representation is VelRepr.Inertial:
            W_X_B = sixd.se3.SE3.from_matrix(self.base_transform()).adjoint()
            return W_X_B @ B_h

        elif self.velocity_representation is VelRepr.Mixed:
            BW_X_B = sixd.se3.SE3.from_matrix(
                self.base_transform().at[0:3, 0:3].set(jnp.eye(3))
            ).adjoint()
            return BW_X_B @ B_h

        else:
            raise ValueError(self.velocity_representation)

    # ==========
    # Algorithms
    # ==========

    def inverse_dynamics(
        self, joint_accelerations: jtp.Vector = None, a0: jtp.Vector = jnp.zeros(6)
    ) -> Tuple[jtp.Vector, jtp.Vector]:

        if self.velocity_representation is VelRepr.Mixed:
            msg = "This method has to be fixed in Mixed representation"
            raise ValueError(msg)

        joint_accelerations = jnp.vstack(
            joint_accelerations
            if joint_accelerations is not None
            else jnp.zeros_like(self.joint_positions())
        )

        if self.velocity_representation is VelRepr.Inertial:

            W_a0_WB_anglin = jnp.hstack([a0.squeeze()[3:6], a0.squeeze()[0:3]])

        elif self.velocity_representation is VelRepr.Body:

            B_a0_WB = a0.squeeze()
            W_X_B = sixd.se3.SE3.from_matrix(self.base_transform()).adjoint()
            W_a0_WB = W_X_B @ B_a0_WB
            W_a0_WB_anglin = jnp.hstack([W_a0_WB[3:6], W_a0_WB[0:3]])

        elif self.velocity_representation is VelRepr.Mixed:

            BW_a0_WB = a0.squeeze()
            W_H_B = self.base_transform()
            W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
            W_a0_WB = W_X_BW @ BW_a0_WB
            W_a0_WB_anglin = jnp.hstack([W_a0_WB[3:6], W_a0_WB[0:3]])

        else:
            raise ValueError(self.velocity_representation)

        f0_anglin, tau = jaxsim.physics.algos.rnea.rnea(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            qdd=joint_accelerations,
            a0fb=W_a0_WB_anglin,
            f_ext=self.data.model_input.f_ext,
        )

        tau = tau.squeeze()
        f0_anglin = f0_anglin.squeeze()

        id_inertial = jnp.hstack(
            list(reversed(jnp.split(f0_anglin, 2))) + [tau]
        ).squeeze()

        if self.velocity_representation is VelRepr.Inertial:

            return id_inertial[0:6], id_inertial[6:]

        elif self.velocity_representation is VelRepr.Body:

            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            B_X_W = sixd.se3.SE3.from_matrix(self.base_transform()).inverse().adjoint()
            T = jnp.block([[B_X_W, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])
            id_body = jnp.linalg.inv(T).T @ id_inertial
            return id_body[0:6], id_body[6:]

        elif self.velocity_representation is VelRepr.Mixed:

            W_H_B = self.base_transform()
            W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
            BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            T = jnp.block([[BW_X_W, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])
            id_mixed = jnp.linalg.inv(T).T @ id_inertial
            return id_mixed[0:6], id_mixed[6:]

        else:
            raise ValueError(self.velocity_representation)

    def forward_dynamics(
        self, tau: jtp.Vector = None, prefer_aba: float = True
    ) -> Union[jtp.Vector, jtp.Vector]:

        if prefer_aba:
            # Note: ABA does not work with closed kinematic chains (not yet supported)
            return self.forward_dynamics_aba(tau=tau)

        else:
            return self.forward_dynamics_crb(tau=tau)

    def forward_dynamics_aba(
        self, tau: jtp.Vector = None
    ) -> Union[jtp.Vector, jtp.Vector]:

        if (
            self.velocity_representation is not VelRepr.Inertial
            and (self.data.model_input.f_ext != 0).any()
        ):
            msg1 = "This method has to be fixed for Body and Mixed representation, "
            msg2 = "use forward_dynamics_crb instead."
            raise ValueError(msg1 + msg2)

        tau = jnp.vstack(
            tau if tau is not None else jnp.zeros_like(self.joint_positions())
        )

        xd_fb, sdd = jaxsim.physics.algos.aba.aba(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            tau=tau,
            f_ext=self.data.model_input.f_ext,
        )

        xd_fb = xd_fb.squeeze()
        sdd = sdd.squeeze()

        W_a_WB = jnp.hstack([xd_fb[10:13], xd_fb[7:10]])

        if self.velocity_representation is VelRepr.Inertial:

            return W_a_WB, sdd

        elif self.velocity_representation is VelRepr.Body:

            W_H_B = self.base_transform()
            B_X_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()

            B_a_WB = B_X_W @ W_a_WB
            return B_a_WB, sdd

        elif self.velocity_representation is VelRepr.Mixed:

            W_H_B = self.base_transform()
            W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
            BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()

            BW_a_WB = BW_X_W @ W_a_WB
            return BW_a_WB, sdd

        else:
            raise ValueError(self.velocity_representation)

    def forward_dynamics_crb(
        self, tau: jtp.Vector = None
    ) -> Union[jtp.Vector, jtp.Vector]:

        tau = (
            tau.squeeze() if tau is not None else jnp.zeros_like(self.joint_positions())
        )
        tau = jnp.array([tau]) if tau.shape == () else tau

        M = self.free_floating_mass_matrix()
        h = jnp.vstack(self.free_floating_generalized_forces())
        J = self.generalized_jacobian()
        f_ext = jnp.vstack(self.external_forces().flatten())
        S = jnp.block([jnp.zeros(shape=(self.dofs(), 6)), jnp.eye(self.dofs())]).T

        if self.floating_base():

            nu_dot = jnp.linalg.inv(M) @ (S @ jnp.vstack(tau) - h + J.T @ f_ext)
            sdd = nu_dot[6:]
            a_WB = nu_dot[0:6]

        else:

            sdd = jnp.linalg.inv(M[6:, 6:]) @ (
                jnp.vstack(tau) - h[6:] + (J.T @ f_ext)[6:]
            )
            a_WB = jnp.zeros(6)

        return a_WB.squeeze(), sdd.squeeze()

    # ==========
    # Kinematics
    # ==========

    def jacobian_momentum(self) -> jtp.Matrix:
        raise NotImplementedError

    def jacobian_centroidal_total_momentum(self) -> jtp.Matrix:
        raise NotImplementedError

    def jacobian_average_velocity(self) -> jtp.Vector:
        raise NotImplementedError

    def jacobian_centroidal_average_velocity(self) -> jtp.Vector:
        raise NotImplementedError

    # ======
    # Energy
    # ======

    def kinematic_energy(self) -> jtp.Float:

        m = self.change_representation(vel_repr=VelRepr.Body)
        nu = m.generalized_velocity()
        M = m.free_floating_mass_matrix()

        return 0.5 * nu.T @ M @ nu

    def potential_energy(self) -> jtp.Float:

        gravity = self.physics_model.gravity[3:6]

        return -jnp.sum(
            jnp.array(
                [
                    l.mass()
                    * jnp.hstack([gravity, 0])
                    @ l.transform()
                    @ jnp.hstack([l.com(), 1])
                    for l in self.links()
                ]
            )
        )

    # ==========
    # Reset data
    # ==========

    def reset_joint_positions(
        self, positions: jtp.Vector, joint_names: List[str] = None
    ) -> "Model":

        if positions.size == 0:
            return self

        if joint_names is None:
            joint_names = self.joint_names()

        if positions.size != len(joint_names):
            raise ValueError("Wrong arguments size", positions.size, len(joint_names))

        # TODO: joint position limits

        new_joint_positions = self.data.model_state.joint_positions.at[
            self._joint_indices(joint_names=joint_names)
        ].set(positions)

        return self.update_data(joint_positions=new_joint_positions)

    def reset_joint_velocities(
        self, velocities: jtp.Vector, joint_names: List[str] = None
    ) -> "Model":

        if velocities.size == 0:
            return self

        if joint_names is None:
            joint_names = self.joint_names()

        if velocities.size != len(joint_names):
            raise ValueError("Wrong arguments size", velocities.size, len(joint_names))

        # TODO: joint velocity limits

        new_joint_velocities = self.data.model_state.joint_velocities.at[
            self._joint_indices(joint_names=joint_names)
        ].set(velocities)

        return self.update_data(joint_velocities=new_joint_velocities)

    def reset_base_position(self, position: jtp.Vector) -> "Model":

        return self.update_data(base_position=position)

    def reset_base_orientation(
        self, orientation: jtp.Array, dcm: bool = False
    ) -> "Model":

        if dcm:
            to_wxyz = np.array([3, 0, 1, 2])
            orientation_xyzw = sixd.so3.SO3.from_matrix(
                orientation
            ).as_quaternion_xyzw()
            orientation = orientation_xyzw[to_wxyz]

        return self.update_data(base_quaternion=orientation)

    def reset_base_transform(self, transform: jtp.Matrix) -> "Model":

        if transform.shape != (4, 4):
            raise ValueError(transform.shape)

        model = self
        model = model.reset_base_position(position=transform[0:3, 3])
        model = model.reset_base_orientation(orientation=transform[0:3, 0:3], dcm=True)
        return model

    def reset_base_velocity(self, base_velocity: jtp.VectorJax) -> "Model":

        if not self.physics_model.is_floating_base:
            msg = "Changing the base velocity of a fixed-based model is not allowed"
            raise RuntimeError(msg)

        # Remove extra dimensions
        base_velocity = base_velocity.squeeze()

        # Check for a valid shape
        if base_velocity.shape != (6,):
            raise ValueError(base_velocity.shape)

        # Convert, if needed, to the representation used internally (VelRepr.Inertial)
        if self.velocity_representation is VelRepr.Inertial:

            base_velocity_inertial = base_velocity

        elif self.velocity_representation is VelRepr.Body:

            w_X_b = sixd.se3.SE3.from_rotation_and_translation(
                rotation=sixd.so3.SO3.from_matrix(self.base_orientation(dcm=True)),
                translation=self.base_position(),
            ).adjoint()

            base_velocity_inertial = w_X_b @ base_velocity

        elif self.velocity_representation is VelRepr.Mixed:

            w_X_bw = sixd.se3.SE3.from_rotation_and_translation(
                rotation=sixd.so3.SO3.identity(),
                translation=self.base_position(),
            ).adjoint()

            base_velocity_inertial = w_X_bw @ base_velocity

        else:
            raise ValueError(self.velocity_representation)

        return self.update_data(
            base_linear_velocity=base_velocity_inertial[0:3],
            base_angular_velocity=base_velocity_inertial[3:6],
        )

    # ===============
    # Private methods
    # ===============

    def _joint_indices(self, joint_names: List[str] = None) -> jtp.Vector:

        if joint_names is None:
            joint_names = self.joint_names()

        if set(joint_names) - set(self._joints.keys()) != set():
            raise ValueError("One or more joint names are not part of the model")

        # Note: joints share the same index as their child link, therefore the first
        #       joint has index=1. We need to subtract one to get the right entry of
        #       data stored in the PhysicsModelState arrays.
        joint_indices = [
            j.joint_description.index - 1 for j in self.joints(joint_names=joint_names)
        ]

        return np.array(joint_indices)
