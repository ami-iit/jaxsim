import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim
import jaxsim.physics.algos.aba
import jaxsim.physics.algos.crba
import jaxsim.physics.algos.forward_kinematics
import jaxsim.physics.algos.rnea
import jaxsim.physics.model.physics_model
import jaxsim.physics.model.physics_model_state
import jaxsim.typing as jtp
from jaxsim import high_level, logging, physics, sixd
from jaxsim.physics.algos import soft_contacts
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.simulation import ode_data, ode_integration
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim.utils import JaxsimDataclass, Mutability

from .common import VelRepr


@jax_dataclasses.pytree_dataclass
class ModelData(JaxsimDataclass):
    """
    Class used to store the model state and input at a given time.
    """

    model_state: jaxsim.physics.model.physics_model_state.PhysicsModelState
    model_input: jaxsim.physics.model.physics_model_state.PhysicsModelInput
    contact_state: jaxsim.physics.algos.soft_contacts.SoftContactsState

    @staticmethod
    def zero(physics_model: physics.model.physics_model.PhysicsModel) -> "ModelData":
        """
        Return a ModelData object with all fields set to zero and initialized with the right shape.

        Args:
            physics_model: The considered physics model.

        Returns:
            The zero ModelData object of the given physics model.
        """

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
class StepData(JaxsimDataclass):
    """
    Class used to store the data computed at each step of the simulation.
    """

    t0: float
    tf: float
    dt: float

    # Starting model data and real input (tau, f_ext) computed at t0
    t0_model_data: ModelData = dataclasses.field(repr=False)
    t0_model_input_real: jaxsim.physics.model.physics_model_state.PhysicsModelInput = (
        dataclasses.field(repr=False)
    )

    # ABA output
    t0_base_acceleration: jtp.Vector = dataclasses.field(repr=False)
    t0_joint_acceleration: jtp.Vector = dataclasses.field(repr=False)

    # (new ODEState)
    # Starting from t0_model_data, can be obtained by integrating the ABA output
    # and tangential_deformation_dot (which is fn of ode_state at t0)
    tf_model_state: jaxsim.physics.model.physics_model_state.PhysicsModelState = (
        dataclasses.field(repr=False)
    )
    tf_contact_state: jaxsim.physics.algos.soft_contacts.SoftContactsState = (
        dataclasses.field(repr=False)
    )

    aux: Dict[str, Any] = dataclasses.field(default_factory=dict)


@jax_dataclasses.pytree_dataclass
class Model(JaxsimDataclass):
    """
    High-level class to operate on a simulated model.
    """

    model_name: str = jax_dataclasses.static_field()
    physics_model: physics.model.physics_model.PhysicsModel = dataclasses.field(
        repr=False
    )

    velocity_representation: VelRepr = jax_dataclasses.static_field(
        default=VelRepr.Mixed
    )

    _links: Dict[str, "high_level.link.Link"] = jax_dataclasses.static_field(
        default_factory=list, repr=False
    )
    _joints: Dict[str, "high_level.joint.Joint"] = jax_dataclasses.static_field(
        default_factory=list, repr=False
    )

    data: ModelData = jax_dataclasses.field(default=None, repr=False)

    # ========================
    # Initialization and state
    # ========================

    @staticmethod
    def build_from_model_description(
        model_description: Union[str, pathlib.Path],
        model_name: Optional[str] = None,
        vel_repr: VelRepr = VelRepr.Mixed,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: Optional[bool] = None,
        considered_joints: Optional[List[str]] = None,
    ) -> "Model":
        """
        Build a Model object from a model description.

        Args:
            model_description: Either a path to a file or a string containing the URDF/SDF description.
            model_name: The optional name of the model that overrides the one in the description.
            vel_repr: The velocity representation to use.
            gravity: The 3D gravity vector.
            is_urdf: Whether the model description is a URDF or an SDF. This is
                automatically inferred if the model description is a path to a file.
            considered_joints: The list of joints to consider. If None, all joints are considered.

        Returns:
            The built Model object.
        """

        import jaxsim.parsers.rod

        # Parse the input resource (either a path to file or a string with the URDF/SDF)
        # and build the -intermediate- model description
        model_description = jaxsim.parsers.rod.build_model_description(
            model_description=model_description, is_urdf=is_urdf
        )

        # Lump links together if not all joints are considered.
        # Note: this procedure assigns a zero position to all joints not considered.
        if considered_joints is not None:
            model_description = model_description.reduce(
                considered_joints=considered_joints
            )

        # Create the physics model from the model description
        physics_model = jaxsim.physics.model.physics_model.PhysicsModel.build_from(
            model_description=model_description, gravity=gravity
        )

        # Build and return the high-level model
        return Model.build(
            physics_model=physics_model,
            model_name=model_name,
            vel_repr=vel_repr,
        )

    @staticmethod
    def build_from_sdf(
        sdf: Union[str, pathlib.Path],
        model_name: Optional[str] = None,
        vel_repr: VelRepr = VelRepr.Mixed,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: Optional[bool] = None,
        considered_joints: Optional[List[str]] = None,
    ) -> "Model":
        """
        Build a Model object from an SDF description.
        This is a deprecated method, use build_from_model_description instead.
        """

        msg = "Model.{} is deprecated, use Model.{} instead."
        logging.warning(
            msg=msg.format("build_from_sdf", "build_from_model_description")
        )

        return Model.build_from_model_description(
            model_description=sdf,
            model_name=model_name,
            vel_repr=vel_repr,
            gravity=gravity,
            is_urdf=is_urdf,
            considered_joints=considered_joints,
        )

    @staticmethod
    def build(
        physics_model: jaxsim.physics.model.physics_model.PhysicsModel,
        model_name: Optional[str] = None,
        vel_repr: VelRepr = VelRepr.Mixed,
    ) -> "Model":
        """
        Build a Model object from a physics model.

        Args:
            physics_model: The physics model.
            model_name: The optional name of the model that overrides the one in the physics model.
            vel_repr: The velocity representation to use.

        Returns:
            The built Model object.
        """

        # Set the model name (if not provided, use the one from the model description)
        model_name = (
            model_name if model_name is not None else physics_model.description.name
        )

        # Sort all the joints by their index
        sorted_links = {
            l.name: high_level.link.Link(link_description=l)
            for l in sorted(
                physics_model.description.links_dict.values(), key=lambda l: l.index
            )
        }

        # Sort all the joints by their index
        sorted_joints = {
            j.name: high_level.joint.Joint(joint_description=j)
            for j in sorted(
                physics_model.description.joints_dict.values(),
                key=lambda j: j.index,
            )
        }

        # Build the high-level model
        model = Model(
            physics_model=physics_model,
            model_name=model_name,
            velocity_representation=vel_repr,
            _links=sorted_links,
            _joints=sorted_joints,
        )

        # Zero the model data
        with model.editable(validate=False) as model:
            model.zero()

        # Check model validity
        if not model.valid():
            raise RuntimeError

        # Return the high-level model
        return model

    def __post_init__(self):
        """Post-init logic. Use the static methods to build high-level models."""

        original_mutability = self._mutability()
        self._set_mutability(Mutability.MUTABLE_NO_VALIDATION)

        for l in self._links.values():
            l.mutable(validate=False).parent_model = self

        for j in self._joints.values():
            j.mutable(validate=False).parent_model = self

        self._links: Dict[str, high_level.link.Link] = {
            k: v for k, v in sorted(self._links.items(), key=lambda kv: kv[1].index())
        }

        self._joints: Dict[str, high_level.joint.Joint] = {
            k: v for k, v in sorted(self._joints.items(), key=lambda kv: kv[1].index())
        }

        self._set_mutability(original_mutability)

    def reduce(self, considered_joints: List[str]) -> None:
        """
        Reduce the model by lumping together the links connected by removed joints.

        Args:
            considered_joints: The list of joints to consider.
        """

        # Reduce the model description
        reduced_model_description = self.physics_model.description.reduce(
            considered_joints=considered_joints
        )

        # Create the physics model from the reduced model description
        physics_model = jaxsim.physics.model.physics_model.PhysicsModel.build_from(
            model_description=reduced_model_description,
            gravity=self.physics_model.gravity[0:3],
        )

        # Build the reduced high-level model
        reduced_model = Model.build(
            physics_model=physics_model,
            model_name=self.name(),
            vel_repr=self.velocity_representation,
        )

        # Replace the current model with the reduced one
        original_mutability = self._mutability()
        self._set_mutability(mutability=self._mutability().MUTABLE_NO_VALIDATION)
        self.physics_model = reduced_model.physics_model
        self.data = reduced_model.data
        self._links = reduced_model._links
        self._joints = reduced_model._joints
        self._set_mutability(original_mutability)

    def zero(self) -> None:
        self.data = ModelData.zero(physics_model=self.physics_model)
        self.data._set_mutability(self._mutability())

    def zero_input(self) -> None:
        self.data.model_input = ModelData.zero(
            physics_model=self.physics_model
        ).model_input

        self.data._set_mutability(self._mutability())

    def zero_state(self) -> None:
        model_data_zero = ModelData.zero(physics_model=self.physics_model)
        self.data.model_state = model_data_zero.model_state
        self.data.contact_state = model_data_zero.contact_state

        self.data._set_mutability(self._mutability())

    def set_velocity_representation(self, vel_repr: VelRepr) -> None:
        if self.velocity_representation is vel_repr:
            return

        self.velocity_representation = vel_repr

    # ==========
    # Properties
    # ==========

    def valid(self) -> bool:
        valid = True
        valid = valid and all([l.valid() for l in self.links()])
        valid = valid and all([j.valid() for j in self.joints()])
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

    def total_mass(self) -> jtp.Float:
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

    def in_contact(
        self,
        link_names: Optional[List[str]] = None,
        terrain: Terrain = FlatTerrain(),
    ) -> jtp.Vector:
        """"""

        link_names = link_names if link_names is not None else self.link_names()

        if set(link_names) - set(self._links.keys()) != set():
            raise ValueError("One or more link names are not part of the model")

        from jaxsim.physics.algos.soft_contacts import collidable_points_pos_vel

        W_p_Ci, _ = collidable_points_pos_vel(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            xfb=self.data.model_state.xfb(),
        )

        terrain_height = jax.vmap(terrain.height)(W_p_Ci[0, :], W_p_Ci[1, :])

        below_terrain = W_p_Ci[2, :] <= terrain_height

        links_in_contact = jax.vmap(
            lambda link_index: jnp.where(
                self.physics_model.gc.body == link_index,
                below_terrain,
                jnp.zeros_like(below_terrain, dtype=bool),
            ).any()
        )(jnp.array([link.index() for link in self.links(link_names=link_names)]))

        return links_in_contact

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

    def joint_generalized_forces_targets(
        self, joint_names: List[str] = None
    ) -> jtp.Vector:
        if self.dofs() == 0 and (joint_names is None or len(joint_names) == 0):
            return jnp.array([])

        return self.data.model_input.tau[self._joint_indices(joint_names=joint_names)]

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

    def base_transform(self) -> jtp.MatrixJax:
        return jnp.block(
            [
                [self.base_orientation(dcm=True), jnp.vstack(self.base_position())],
                [0, 0, 0, 1],
            ]
        )

    def base_velocity(self) -> jtp.Vector:
        W_v_WB = jnp.hstack(
            [
                self.data.model_state.base_linear_velocity,
                self.data.model_state.base_angular_velocity,
            ]
        )

        return self.inertial_to_active_representation(array=W_v_WB)

    def external_forces(self) -> jtp.Matrix:
        W_f_ext = self.data.model_input.f_ext

        inertial_to_active = lambda f: self.inertial_to_active_representation(
            f, is_force=True
        )

        return jax.vmap(inertial_to_active, in_axes=0)(W_f_ext)

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
        M_body = jaxsim.physics.algos.crba.crba(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
        )

        if self.velocity_representation is VelRepr.Body:
            return M_body

        elif self.velocity_representation is VelRepr.Inertial:
            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            B_X_W = sixd.se3.SE3.from_matrix(self.base_transform()).inverse().adjoint()

            invT = jnp.block([[B_X_W, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])

            return invT.T @ M_body @ invT

        elif self.velocity_representation is VelRepr.Mixed:
            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            W_H_BW = self.base_transform().at[0:3, 3].set(jnp.zeros(3))
            BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()

            invT = jnp.block([[BW_X_W, zero_6n], [zero_6n.T, jnp.eye(self.dofs())]])

            return invT.T @ M_body @ invT

        else:
            raise ValueError(self.velocity_representation)

    def free_floating_generalized_forces(self) -> jtp.Vector:
        model_state = self.data.model_state
        model = self.copy().mutable(validate=True)

        model.zero()
        model.data.model_state.base_position = model_state.base_position
        model.data.model_state.base_quaternion = model_state.base_quaternion
        model.data.model_state.joint_positions = model_state.joint_positions
        model.data.model_state.base_linear_velocity = model_state.base_linear_velocity
        model.data.model_state.base_angular_velocity = model_state.base_angular_velocity
        model.data.model_state.joint_velocities = model_state.joint_velocities

        return jnp.hstack(model.inverse_dynamics())

    def free_floating_gravity_forces(self) -> jtp.Vector:
        model_state = self.data.model_state
        model = self.copy().mutable(validate=True)

        model.zero()
        model.data.model_state.base_position = model_state.base_position
        model.data.model_state.base_quaternion = model_state.base_quaternion
        model.data.model_state.joint_positions = model_state.joint_positions

        return jnp.hstack(model.inverse_dynamics())

    def momentum(self) -> jtp.Vector:
        with self.editable(validate=True) as m:
            m.set_velocity_representation(vel_repr=VelRepr.Body)

        # Compute the momentum in body-fixed velocity representation.
        # Note: the first 6 rows of the mass matrix define the jacobian of the
        #       floating-base momentum.
        B_h = m.free_floating_mass_matrix()[0:6, :] @ m.generalized_velocity()

        W_H_B = self.base_transform()
        B_X_W: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()

        W_h = B_X_W.T @ B_h
        return self.inertial_to_active_representation(array=W_h, is_force=True)

    # ==============================
    # Quantities related to the CoM
    # ==============================

    def com_position(self) -> jtp.Vector:
        m = self.total_mass()

        W_H_L = self.forward_kinematics()
        W_H_B = self.base_transform()
        B_H_W = jnp.linalg.inv(W_H_B)

        com_links = [
            (l.mass() * B_H_W @ W_H_L[l.index()] @ jnp.hstack([l.com(), 1]))
            for l in self.links()
        ]

        B_ph_CoM = (1 / m) * jnp.sum(jnp.array(com_links), axis=0)

        return (W_H_B @ B_ph_CoM)[0:3]

    # ==========
    # Algorithms
    # ==========

    def forward_kinematics(self) -> jtp.Array:
        W_H_i = jaxsim.physics.algos.forward_kinematics.forward_kinematics_model(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
            xfb=self.data.model_state.xfb(),
        )

        return W_H_i

    def inverse_dynamics(
        self,
        joint_accelerations: jtp.Vector = None,
        base_acceleration: jtp.Vector = jnp.zeros(6),
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        if (
            self.velocity_representation is VelRepr.Mixed
            and self.floating_base()
            and not jnp.allclose(self.data.model_state.base_angular_velocity, 0)
        ):
            msg = "This method has to be fixed in Mixed representation"
            raise ValueError(msg)

        # Build joint accelerations if not provided
        joint_accelerations = (
            joint_accelerations
            if joint_accelerations is not None
            else jnp.zeros_like(self.joint_positions())
        )

        if joint_accelerations.size != self.dofs():
            raise ValueError(joint_accelerations.size)

        if base_acceleration.size != 6:
            raise ValueError(base_acceleration.size)

        # Express base_acceleration in inertial representation
        W_a_WB = self.active_to_inertial_representation(array=base_acceleration)

        # Compute RNEA
        W_f_B, tau = jaxsim.physics.algos.rnea.rnea(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            qdd=joint_accelerations,
            a0fb=W_a_WB,
            f_ext=self.data.model_input.f_ext,
        )

        # Adjust shape
        tau = jnp.atleast_1d(tau.squeeze())

        # Express W_f_B in the active representation
        f_B = self.inertial_to_active_representation(array=W_f_B, is_force=True)

        return f_B, tau

    def forward_dynamics(
        self, tau: jtp.Vector = None, prefer_aba: bool = True
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        return (
            self.forward_dynamics_aba(tau=tau)
            if prefer_aba
            else self.forward_dynamics_crb(tau=tau)
        )

    def forward_dynamics_aba(
        self, tau: jtp.Vector = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        if (
            self.velocity_representation is not VelRepr.Inertial
            and (self.data.model_input.f_ext != 0).any()
        ):
            msg1 = "This method has to be fixed for Body and Mixed representation, "
            msg2 = "use forward_dynamics_crb instead."
            raise ValueError(msg1 + msg2)

        # Build joint torques if not provided
        tau = tau if tau is not None else jnp.zeros_like(self.joint_positions())

        # Compute ABA
        W_a_WB, sdd = jaxsim.physics.algos.aba.aba(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            tau=tau,
            f_ext=self.data.model_input.f_ext,
        )

        # Adjust shape
        sdd = jnp.atleast_1d(sdd.squeeze())

        # Express W_a_WB in the active representation
        a_WB = self.inertial_to_active_representation(array=W_a_WB)

        return a_WB, sdd

    def forward_dynamics_crb(
        self, tau: jtp.Vector = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        # Build joint torques if not provided
        tau = tau if tau is not None else jnp.zeros_like(self.joint_positions())
        tau = jnp.atleast_1d(tau.squeeze())
        tau = jnp.vstack(tau) if tau.size > 0 else jnp.empty(shape=(0, 1))

        # Compute terms of the floating-base EoM
        M = self.free_floating_mass_matrix()
        h = jnp.vstack(self.free_floating_generalized_forces())
        J = self.generalized_jacobian()
        f_ext = jnp.vstack(self.external_forces().flatten())
        S = jnp.block([jnp.zeros(shape=(self.dofs(), 6)), jnp.eye(self.dofs())]).T

        if self.floating_base():
            nu_dot = jnp.linalg.inv(M) @ (S @ tau - h + J.T @ f_ext)
            sdd = nu_dot[6:]
            a_WB = nu_dot[0:6]

        else:
            hss = h[6:]
            Jss = J[:, 6:]
            Mss = M[6:, 6:]

            a_WB = jnp.zeros(6)
            sdd = jnp.linalg.inv(Mss) @ (tau - hss + Jss.T @ f_ext)

        # Adjust shape and convert to lin-ang serialization
        a_WB = a_WB.squeeze()
        sdd = jnp.atleast_1d(sdd.squeeze())

        return a_WB, sdd

    # ======
    # Energy
    # ======

    def mechanical_energy(self) -> jtp.Float:
        K = self.kinetic_energy()
        U = self.potential_energy()

        return K + U

    def kinetic_energy(self) -> jtp.Float:
        with self.editable(validate=True) as m:
            m.set_velocity_representation(vel_repr=VelRepr.Body)

        nu = m.generalized_velocity()
        M = m.free_floating_mass_matrix()

        return 0.5 * nu.T @ M @ nu

    def potential_energy(self) -> jtp.Float:
        m = self.total_mass()
        W_p_CoM = jnp.hstack([self.com_position(), 1])
        gravity = self.physics_model.gravity[3:6].squeeze()

        return -(m * jnp.hstack([gravity, 0]) @ W_p_CoM)

    # ===========
    # Set targets
    # ===========

    def set_joint_generalized_force_targets(
        self, forces: jtp.Vector, joint_names: List[str] = None
    ) -> None:
        if joint_names is None:
            joint_names = self.joint_names()

        if forces.size != len(joint_names):
            raise ValueError("Wrong arguments size", forces.size, len(joint_names))

        self.data.model_input.tau = self.data.model_input.tau.at[
            self._joint_indices(joint_names=joint_names)
        ].set(forces)

    # ==========
    # Reset data
    # ==========

    def reset_joint_positions(
        self, positions: jtp.Vector, joint_names: List[str] = None
    ) -> None:
        if joint_names is None:
            joint_names = self.joint_names()

        if positions.size != len(joint_names):
            raise ValueError("Wrong arguments size", positions.size, len(joint_names))

        if positions.size == 0:
            return

        # TODO: joint position limits

        self.data.model_state.joint_positions = (
            self.data.model_state.joint_positions.at[
                self._joint_indices(joint_names=joint_names)
            ].set(positions)
        )

    def reset_joint_velocities(
        self, velocities: jtp.Vector, joint_names: List[str] = None
    ) -> None:
        if joint_names is None:
            joint_names = self.joint_names()

        if velocities.size != len(joint_names):
            raise ValueError("Wrong arguments size", velocities.size, len(joint_names))

        if velocities.size == 0:
            return

        # TODO: joint velocity limits

        self.data.model_state.joint_velocities = (
            self.data.model_state.joint_velocities.at[
                self._joint_indices(joint_names=joint_names)
            ].set(velocities)
        )

    def reset_base_position(self, position: jtp.Vector) -> None:
        self.data.model_state.base_position = position

    def reset_base_orientation(self, orientation: jtp.Array, dcm: bool = False) -> None:
        if dcm:
            to_wxyz = np.array([3, 0, 1, 2])
            orientation_xyzw = sixd.so3.SO3.from_matrix(
                orientation
            ).as_quaternion_xyzw()
            orientation = orientation_xyzw[to_wxyz]

        self.data.model_state.base_quaternion = orientation

    def reset_base_transform(self, transform: jtp.Matrix) -> None:
        if transform.shape != (4, 4):
            raise ValueError(transform.shape)

        self.reset_base_position(position=transform[0:3, 3])
        self.reset_base_orientation(orientation=transform[0:3, 0:3], dcm=True)

    def reset_base_velocity(self, base_velocity: jtp.VectorJax) -> None:
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

        self.data.model_state.base_linear_velocity = base_velocity_inertial[0:3]
        self.data.model_state.base_angular_velocity = base_velocity_inertial[3:6]

    # ===========
    # Integration
    # ===========

    def integrate(
        self,
        t0: jtp.Float,
        tf: jtp.Float,
        sub_steps: int = 1,
        integrator_type: IntegratorType = IntegratorType.EulerForward,
        terrain: soft_contacts.Terrain = soft_contacts.FlatTerrain(),
        contact_parameters: soft_contacts.SoftContactsParams = soft_contacts.SoftContactsParams(),
        clear_inputs: bool = False,
    ) -> StepData:
        x0 = ode_integration.ode.ode_data.ODEState(
            physics_model=self.data.model_state,
            soft_contacts=self.data.contact_state,
        )

        ode_input = ode_integration.ode.ode_data.ODEInput(
            physics_model=self.data.model_input
        )

        if integrator_type is IntegratorType.EulerForward:
            integrator_fn = ode_integration.ode_integration_euler

        elif integrator_type is IntegratorType.EulerSemiImplicit:
            integrator_fn = ode_integration.ode_integration_euler_semi_implicit

        elif integrator_type is IntegratorType.RungeKutta4:
            integrator_fn = ode_integration.ode_integration_rk4

        else:
            raise ValueError(integrator_type)

        # Integrate the model dynamics
        ode_states, aux = integrator_fn(
            x0=x0,
            t=jnp.array([t0, tf], dtype=float),
            ode_input=ode_input,
            physics_model=self.physics_model,
            soft_contacts_params=contact_parameters,
            num_sub_steps=sub_steps,
            terrain=terrain,
            return_aux=True,
        )

        # Get quantities at t0
        t0_model_data = self.data
        t0_model_input = jax.tree_util.tree_map(
            lambda l: l[0],
            aux["ode_input"],
        )
        t0_model_input_real = jax.tree_util.tree_map(
            lambda l: l[0],
            aux["ode_input_real"],
        )
        t0_model_acceleration = jax.tree_util.tree_map(
            lambda l: l[0],
            aux["model_acceleration"],
        )

        # Get quantities at tf
        ode_states: ode_data.ODEState
        tf_model_state = jax.tree_util.tree_map(
            lambda l: l[-1], ode_states.physics_model
        )
        tf_contact_state = jax.tree_util.tree_map(
            lambda l: l[-1], ode_states.soft_contacts
        )

        # Clear user inputs (joint torques and external forces) if asked
        model_input = jax.lax.cond(
            pred=clear_inputs,
            false_fun=lambda: t0_model_input.physics_model,
            true_fun=lambda: jaxsim.physics.model.physics_model_state.PhysicsModelInput.zero(
                physics_model=self.physics_model
            ),
        )

        # Update model state
        self.data = ModelData(
            model_state=tf_model_state,
            contact_state=tf_contact_state,
            model_input=model_input,
        )
        self._set_mutability(self._mutability())

        return StepData(
            t0=t0,
            tf=tf,
            dt=(tf - t0),
            t0_model_data=t0_model_data,
            t0_model_input_real=t0_model_input_real.physics_model,
            t0_base_acceleration=t0_model_acceleration[0:6],
            t0_joint_acceleration=t0_model_acceleration[6:],
            tf_model_state=tf_model_state,
            tf_contact_state=tf_contact_state,
            aux={
                "t0": jax.tree_util.tree_map(
                    lambda l: l[0],
                    aux,
                ),
                "tf": jax.tree_util.tree_map(
                    lambda l: l[-1],
                    aux,
                ),
            },
        )

    # ===============
    # Private methods
    # ===============

    def inertial_to_active_representation(
        self, array: jtp.Array, is_force: bool = False
    ) -> jtp.Array:
        W_array = array.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size)

        if self.velocity_representation is VelRepr.Inertial:
            return W_array

        elif self.velocity_representation is VelRepr.Body:
            W_H_B = self.base_transform()

            if not is_force:
                B_X_W: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
                B_array = B_X_W @ W_array

            else:
                W_X_B: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).adjoint()
                B_array = W_X_B.T @ W_array

            return B_array

        elif self.velocity_representation is VelRepr.Mixed:
            W_H_BW = jnp.eye(4).at[0:3, 3].set(self.base_position())

            if not is_force:
                BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
                BW_array = BW_X_W @ W_array

            else:
                W_X_BW = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
                BW_array = W_X_BW.transpose() @ W_array

            return BW_array

        else:
            raise ValueError(self.velocity_representation)

    def active_to_inertial_representation(
        self, array: jtp.Array, is_force: bool = False
    ) -> jtp.Array:
        array = array.squeeze()

        if array.size != 6:
            raise ValueError(array.size)

        if self.velocity_representation is VelRepr.Inertial:
            W_array = array
            return W_array

        elif self.velocity_representation is VelRepr.Body:
            B_array = array
            W_H_B = self.base_transform()

            if not is_force:
                W_X_B: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).adjoint()
                W_array = W_X_B @ B_array

            else:
                B_X_W: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
                W_array = B_X_W.T @ B_array

            return W_array

        elif self.velocity_representation is VelRepr.Mixed:
            BW_array = array
            W_H_BW = jnp.eye(4).at[0:3, 3].set(self.base_position())

            if not is_force:
                W_X_BW: jtp.Array = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
                W_array = W_X_BW @ BW_array

            else:
                BW_X_W: jtp.Array = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
                W_array = BW_X_W.T @ BW_array

            return W_array

        else:
            raise ValueError(self.velocity_representation)

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
