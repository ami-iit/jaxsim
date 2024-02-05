import dataclasses
import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import rod
from jax_dataclasses import Static

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
from jaxsim.utils import JaxsimDataclass, Mutability, Vmappable, oop

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
class Model(Vmappable):
    """
    High-level class to operate on a simulated model.
    """

    model_name: Static[str]

    physics_model: physics.model.physics_model.PhysicsModel = dataclasses.field(
        repr=False
    )

    velocity_representation: Static[VelRepr] = dataclasses.field(default=VelRepr.Mixed)

    data: ModelData = dataclasses.field(default=None, repr=False)

    # ========================
    # Initialization and state
    # ========================

    @staticmethod
    def build_from_model_description(
        model_description: Union[str, pathlib.Path, rod.Model],
        model_name: str | None = None,
        vel_repr: VelRepr = VelRepr.Mixed,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: bool | None = None,
        considered_joints: List[str] | None = None,
    ) -> "Model":
        """
        Build a Model object from a model description.

        Args:
            model_description: A path to an SDF/URDF file, a string containing its content, or a pre-parsed/pre-built rod model.
            model_name: The optional name of the model that overrides the one in the description.
            vel_repr: The velocity representation to use.
            gravity: The 3D gravity vector.
            is_urdf: Whether the model description is a URDF or an SDF. This is automatically inferred if the model description is a path to a file.
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
        model_name: str | None = None,
        vel_repr: VelRepr = VelRepr.Mixed,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: bool | None = None,
        considered_joints: List[str] | None = None,
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
        model_name: str | None = None,
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

        # Build the high-level model
        model = Model(
            physics_model=physics_model,
            model_name=model_name,
            velocity_representation=vel_repr,
        )

        # Zero the model data
        with model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            model.zero()

        # Check model validity
        if not model.valid():
            raise RuntimeError("The model is not valid.")

        # Return the high-level model
        return model

    @functools.partial(oop.jax_tf.method_rw, jit=False, vmap=False, validate=False)
    def reduce(
        self, considered_joints: tuple[str, ...], keep_base_pose: bool = False
    ) -> None:
        """
        Reduce the model by lumping together the links connected by removed joints.

        Args:
            considered_joints: The sequence of joints to consider.
            keep_base_pose: A flag indicating whether to keep the base pose or not.
        """

        if self.vectorized:
            raise RuntimeError("Cannot reduce a vectorized model.")

        # Reduce the model description.
        # If considered_joints contains joints not existing in the model, the method
        # will raise an exception.
        reduced_model_description = self.physics_model.description.reduce(
            considered_joints=list(considered_joints)
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

        # Extract the base pose
        W_p_B = self.base_position()
        W_Q_B = self.base_orientation(dcm=False)

        # Replace the current model with the reduced model.
        # Since the structure of the PyTree changes, we disable validation.
        self.physics_model = reduced_model.physics_model
        self.data = reduced_model.data

        if keep_base_pose:
            self.reset_base_position(position=W_p_B)
            self.reset_base_orientation(orientation=W_Q_B, dcm=False)

    @functools.partial(oop.jax_tf.method_rw, jit=False)
    def zero(self) -> None:
        """"""

        self.data = ModelData.zero(physics_model=self.physics_model)

    @functools.partial(oop.jax_tf.method_rw, jit=False)
    def zero_input(self) -> None:
        """"""

        self.data.model_input = ModelData.zero(
            physics_model=self.physics_model
        ).model_input

    @functools.partial(oop.jax_tf.method_rw, jit=False)
    def zero_state(self) -> None:
        """"""

        model_data_zero = ModelData.zero(physics_model=self.physics_model)
        self.data.model_state = model_data_zero.model_state
        self.data.contact_state = model_data_zero.contact_state

    @functools.partial(oop.jax_tf.method_rw, jit=False, vmap=False)
    def set_velocity_representation(self, vel_repr: VelRepr) -> None:
        """"""

        if self.velocity_representation is vel_repr:
            return

        self.velocity_representation = vel_repr

    # ==========
    # Properties
    # ==========

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def valid(self) -> jtp.Bool:
        """"""

        valid = True
        valid = valid and all(l.valid() for l in self.links())
        valid = valid and all(j.valid() for j in self.joints())
        return jnp.array(valid, dtype=bool)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def floating_base(self) -> jtp.Bool:
        """"""

        return jnp.array(self.physics_model.is_floating_base, dtype=bool)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def dofs(self) -> jtp.Int:
        """"""

        return self.joint_positions().size

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def name(self) -> str:
        """"""

        return self.model_name

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def nr_of_links(self) -> jtp.Int:
        """"""

        return jnp.array(len(self.links()), dtype=int)

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def nr_of_joints(self) -> jtp.Int:
        """"""

        return jnp.array(len(self.joints()), dtype=int)

    @functools.partial(oop.jax_tf.method_ro)
    def total_mass(self) -> jtp.Float:
        """"""

        return jnp.sum(jnp.array([l.mass() for l in self.links()]), dtype=float)

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def get_link(self, link_name: str) -> high_level.link.Link:
        """"""

        if link_name not in self.link_names():
            msg = f"Link '{link_name}' is not part of model '{self.name()}'"
            raise ValueError(msg)

        return self.links(link_names=(link_name,))[0]

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def get_joint(self, joint_name: str) -> high_level.joint.Joint:
        """"""

        if joint_name not in self.joint_names():
            msg = f"Joint '{joint_name}' is not part of model '{self.name()}'"
            raise ValueError(msg)

        return self.joints(joint_names=(joint_name,))[0]

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def link_names(self) -> tuple[str, ...]:
        """"""

        return tuple(self.physics_model.description.links_dict.keys())

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def joint_names(self) -> tuple[str, ...]:
        """"""

        return tuple(self.physics_model.description.joints_dict.keys())

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def links(
        self, link_names: tuple[str, ...] | None = None
    ) -> tuple[high_level.link.Link, ...]:
        """"""

        all_links = {
            l.name: high_level.link.Link(
                link_description=l, _parent_model=self, batch_size=self.batch_size
            )
            for l in sorted(
                self.physics_model.description.links_dict.values(),
                key=lambda l: l.index,
            )
        }

        for l in all_links.values():
            l._set_mutability(self._mutability())

        if link_names is None:
            return tuple(all_links.values())

        return tuple(all_links[name] for name in link_names)

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def joints(
        self, joint_names: tuple[str, ...] | None = None
    ) -> tuple[high_level.joint.Joint, ...]:
        """"""

        all_joints = {
            j.name: high_level.joint.Joint(
                joint_description=j, _parent_model=self, batch_size=self.batch_size
            )
            for j in sorted(
                self.physics_model.description.joints_dict.values(),
                key=lambda j: j.index,
            )
        }

        for j in all_joints.values():
            j._set_mutability(self._mutability())

        if joint_names is None:
            return tuple(all_joints.values())

        return tuple(all_joints[name] for name in joint_names)

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["link_names", "terrain"])
    def in_contact(
        self,
        link_names: tuple[str, ...] | None = None,
        terrain: Terrain = FlatTerrain(),
    ) -> jtp.Vector:
        """"""

        link_names = link_names if link_names is not None else self.link_names()

        if set(link_names) - set(self.link_names()) != set():
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

    # =================
    # Multi-DoF methods
    # =================

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["joint_names"])
    def joint_positions(self, joint_names: tuple[str, ...] | None = None) -> jtp.Vector:
        """"""

        return self.data.model_state.joint_positions[
            self._joint_indices(joint_names=joint_names)
        ]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["joint_names"])
    def joint_random_positions(
        self,
        joint_names: tuple[str, ...] | None = None,
        key: jax.Array | None = None,
    ) -> jtp.Vector:
        """"""

        if key is None:
            key = jax.random.PRNGKey(seed=0)

        s_min, s_max = self.joint_limits(joint_names=joint_names)

        s_random = jax.random.uniform(
            minval=s_min,
            maxval=s_max,
            key=key,
            shape=s_min.shape,
        )

        return s_random

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["joint_names"])
    def joint_velocities(
        self, joint_names: tuple[str, ...] | None = None
    ) -> jtp.Vector:
        """"""

        return self.data.model_state.joint_velocities[
            self._joint_indices(joint_names=joint_names)
        ]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["joint_names"])
    def joint_generalized_forces_targets(
        self, joint_names: tuple[str, ...] | None = None
    ) -> jtp.Vector:
        """"""

        return self.data.model_input.tau[self._joint_indices(joint_names=joint_names)]

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["joint_names"])
    def joint_limits(
        self, joint_names: tuple[str, ...] | None = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """"""

        # Consider all joints if not specified otherwise
        joint_names = joint_names if joint_names is not None else self.joint_names()

        # Create a (Dofs, 2) matrix containing the joint limits
        limits = jnp.vstack(
            jnp.array([j.position_limit() for j in self.joints(joint_names)])
        )

        # Get the limits, reordering them in case low > high
        s_low = jnp.min(limits, axis=1)
        s_high = jnp.max(limits, axis=1)

        return s_low, s_high

    # =========
    # Base link
    # =========

    @functools.partial(oop.jax_tf.method_ro, jit=False, vmap=False)
    def base_frame(self) -> str:
        """"""

        return self.physics_model.description.root.name

    @functools.partial(oop.jax_tf.method_ro)
    def base_position(self) -> jtp.Vector:
        """"""

        return self.data.model_state.base_position.squeeze()

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["dcm"])
    def base_orientation(self, dcm: bool = False) -> jtp.Vector:
        """"""

        # Normalize the quaternion before using it.
        # Our integration logic has a Baumgarte stabilization term makes the quaternion
        # norm converge to 1, but it does not enforce to be 1 at all the time instants.
        base_unit_quaternion = (
            self.data.model_state.base_quaternion.squeeze()
            / jnp.linalg.norm(self.data.model_state.base_quaternion)
        )

        # wxyz -> xyzw
        to_xyzw = np.array([1, 2, 3, 0])

        return (
            base_unit_quaternion
            if not dcm
            else sixd.so3.SO3.from_quaternion_xyzw(
                base_unit_quaternion[to_xyzw]
            ).as_matrix()
        )

    @functools.partial(oop.jax_tf.method_ro)
    def base_transform(self) -> jtp.MatrixJax:
        """"""

        W_R_B = self.base_orientation(dcm=True)
        W_p_B = jnp.vstack(self.base_position())

        return jnp.vstack(
            [
                jnp.block([W_R_B, W_p_B]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

    @functools.partial(oop.jax_tf.method_ro)
    def base_velocity(self) -> jtp.Vector:
        """"""

        W_v_WB = jnp.hstack(
            [
                self.data.model_state.base_linear_velocity,
                self.data.model_state.base_angular_velocity,
            ]
        )

        return self.inertial_to_active_representation(array=W_v_WB)

    @functools.partial(oop.jax_tf.method_ro)
    def external_forces(self) -> jtp.Matrix:
        """
        Return the active external forces acting on the robot.

        The external forces are a user input and are not computed by the physics engine.
        During the simulation, these external forces are summed to other terms like
        the external forces due to the contact with the environment.

        Returns:
            A matrix of shape (n_links, 6) containing the external forces acting on the
            robot links. The forces are expressed in the active representation.
        """

        # Get the active external forces that are always stored internally
        # in Inertial representation
        W_f_ext = self.data.model_input.f_ext

        inertial_to_active = lambda f: self.inertial_to_active_representation(
            f, is_force=True
        )

        return jax.vmap(inertial_to_active, in_axes=0)(W_f_ext)

    # =======================
    # Single link r/w methods
    # =======================

    @functools.partial(
        oop.jax_tf.method_rw, jit=True, static_argnames=["link_name", "additive"]
    )
    def apply_external_force_to_link(
        self,
        link_name: str,
        force: jtp.Array | None = None,
        torque: jtp.Array | None = None,
        additive: bool = True,
    ) -> None:
        """"""

        # Get the target link with the correct mutability
        link = self.get_link(link_name=link_name)
        link._set_mutability(mutability=self._mutability())

        # Initialize zero force components if not set
        force = force if force is not None else jnp.zeros(3)
        torque = torque if torque is not None else jnp.zeros(3)

        # Build the target 6D force in the active representation
        f_ext = jnp.hstack([force, torque])

        # Convert the 6D force to the inertial representation
        if self.velocity_representation is VelRepr.Inertial:
            W_f_ext = f_ext

        elif self.velocity_representation is VelRepr.Body:
            L_f_ext = f_ext
            W_H_L = link.transform()
            L_X_W = sixd.se3.SE3.from_matrix(W_H_L).inverse().adjoint()

            W_f_ext = L_X_W.transpose() @ L_f_ext

        elif self.velocity_representation is VelRepr.Mixed:
            LW_f_ext = f_ext

            W_p_L = link.transform()[0:3, 3]
            W_H_LW = jnp.eye(4).at[0:3, 3].set(W_p_L)
            LW_X_W = sixd.se3.SE3.from_matrix(W_H_LW).inverse().adjoint()

            W_f_ext = LW_X_W.transpose() @ LW_f_ext

        else:
            raise ValueError(self.velocity_representation)

        # Obtain the new 6D force considering the 'additive' flag
        W_f_ext_current = self.data.model_input.f_ext[link.index(), :]
        new_force = W_f_ext_current + W_f_ext if additive else W_f_ext

        # Update the model data
        self.data.model_input.f_ext = self.data.model_input.f_ext.at[
            link.index(), :
        ].set(new_force)

    @functools.partial(
        oop.jax_tf.method_rw, jit=True, static_argnames=["link_name", "additive"]
    )
    def apply_external_force_to_link_com(
        self,
        link_name: str,
        force: jtp.Array | None = None,
        torque: jtp.Array | None = None,
        additive: bool = True,
    ) -> None:
        """"""

        # Get the target link with the correct mutability
        link = self.get_link(link_name=link_name)
        link._set_mutability(mutability=self._mutability())

        # Initialize zero force components if not set
        force = force if force is not None else jnp.zeros(3)
        torque = torque if torque is not None else jnp.zeros(3)

        # Build the target 6D force in the active representation
        f_ext = jnp.hstack([force, torque])

        # Convert the 6D force to the inertial representation
        if self.velocity_representation is VelRepr.Inertial:
            W_f_ext = f_ext

        elif self.velocity_representation is VelRepr.Body:
            GL_f_ext = f_ext

            W_H_L = link.transform()
            L_p_CoM = link.com_position(in_link_frame=True)
            L_H_GL = jnp.eye(4).at[0:3, 3].set(L_p_CoM)
            W_H_GL = W_H_L @ L_H_GL
            GL_X_W = sixd.se3.SE3.from_matrix(W_H_GL).inverse().adjoint()

            W_f_ext = GL_X_W.transpose() @ GL_f_ext

        elif self.velocity_representation is VelRepr.Mixed:
            GW_f_ext = f_ext

            W_p_CoM = link.com_position(in_link_frame=False)
            W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            GW_X_W = sixd.se3.SE3.from_matrix(W_H_GW).inverse().adjoint()

            W_f_ext = GW_X_W.transpose() @ GW_f_ext

        else:
            raise ValueError(self.velocity_representation)

        # Obtain the new 6D force considering the 'additive' flag
        W_f_ext_current = self.data.model_input.f_ext[link.index(), :]
        new_force = W_f_ext_current + W_f_ext if additive else W_f_ext

        # Update the model data
        self.data.model_input.f_ext = self.data.model_input.f_ext.at[
            link.index(), :
        ].set(new_force)

    # ================================================
    # Generalized methods and free-floating quantities
    # ================================================

    @functools.partial(oop.jax_tf.method_ro)
    def generalized_position(self) -> Tuple[jtp.Matrix, jtp.Vector]:
        """"""

        return self.base_transform(), self.joint_positions()

    @functools.partial(oop.jax_tf.method_ro)
    def generalized_velocity(self) -> jtp.Vector:
        """"""

        return jnp.hstack([self.base_velocity(), self.joint_velocities()])

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["output_vel_repr"])
    def generalized_free_floating_jacobian(
        self, output_vel_repr: VelRepr | None = None
    ) -> jtp.Matrix:
        """"""

        if output_vel_repr is None:
            output_vel_repr = self.velocity_representation

        # The body frame of the Link.jacobian method is the link frame L.
        # In this method, we want instead to use the base link B as body frame.
        # Therefore, we always get the link jacobian having Inertial as output
        # representation, and then we convert it to the desired output representation.
        if output_vel_repr is VelRepr.Inertial:
            to_output = lambda J: J

        elif output_vel_repr is VelRepr.Body:

            def to_output(W_J_Wi):
                W_H_B = self.base_transform()
                B_X_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
                return B_X_W @ W_J_Wi

        elif output_vel_repr is VelRepr.Mixed:

            def to_output(W_J_Wi):
                W_H_B = self.base_transform()
                W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
                BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
                return BW_X_W @ W_J_Wi

        else:
            raise ValueError(output_vel_repr)

        # Get the link jacobians in Inertial representation and convert them to the
        # target output representation in which the body frame is the base link B
        J_free_floating = jnp.vstack(
            [
                to_output(
                    self.get_link(link_name=link_name).jacobian(
                        output_vel_repr=VelRepr.Inertial
                    )
                )
                for link_name in self.link_names()
            ]
        )

        return J_free_floating

    @functools.partial(oop.jax_tf.method_ro)
    def free_floating_mass_matrix(self) -> jtp.Matrix:
        """"""

        M_body = jaxsim.physics.algos.crba.crba(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
        )

        if self.velocity_representation is VelRepr.Body:
            return M_body

        elif self.velocity_representation is VelRepr.Inertial:
            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            B_X_W = sixd.se3.SE3.from_matrix(self.base_transform()).inverse().adjoint()

            invT = jnp.vstack(
                [
                    jnp.block([B_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(self.dofs())]),
                ]
            )

            return invT.T @ M_body @ invT

        elif self.velocity_representation is VelRepr.Mixed:
            zero_6n = jnp.zeros(shape=(6, self.dofs()))
            W_H_BW = self.base_transform().at[0:3, 3].set(jnp.zeros(3))
            BW_X_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()

            invT = jnp.vstack(
                [
                    jnp.block([BW_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(self.dofs())]),
                ]
            )

            return invT.T @ M_body @ invT

        else:
            raise ValueError(self.velocity_representation)

    @functools.partial(oop.jax_tf.method_ro)
    def free_floating_bias_forces(self) -> jtp.Vector:
        """"""

        with self.editable(validate=True) as model:
            model.zero_input()

        return jnp.hstack(
            model.inverse_dynamics(
                base_acceleration=jnp.zeros(6), joint_accelerations=None
            )
        )

    @functools.partial(oop.jax_tf.method_ro)
    def free_floating_gravity_forces(self) -> jtp.Vector:
        """"""

        with self.editable(validate=True) as model:
            model.zero_input()
            model.data.model_state.joint_velocities = jnp.zeros_like(
                model.data.model_state.joint_velocities
            )
            model.data.model_state.base_linear_velocity = jnp.zeros_like(
                model.data.model_state.base_linear_velocity
            )
            model.data.model_state.base_angular_velocity = jnp.zeros_like(
                model.data.model_state.base_angular_velocity
            )

        return jnp.hstack(
            model.inverse_dynamics(
                base_acceleration=jnp.zeros(6), joint_accelerations=None
            )
        )

    @functools.partial(oop.jax_tf.method_ro)
    def momentum(self) -> jtp.Vector:
        """"""

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

    # ===========
    # CoM methods
    # ===========

    @functools.partial(oop.jax_tf.method_ro)
    def com_position(self) -> jtp.Vector:
        """"""

        m = self.total_mass()

        W_H_L = self.forward_kinematics()
        W_H_B = self.base_transform()
        B_H_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().as_matrix()

        com_links = [
            (
                l.mass()
                * B_H_W
                @ W_H_L[l.index()]
                @ jnp.hstack([l.com_position(in_link_frame=True), 1])
            )
            for l in self.links()
        ]

        B_ph_CoM = (1 / m) * jnp.sum(jnp.array(com_links), axis=0)

        return (W_H_B @ B_ph_CoM)[0:3]

    # ==========
    # Algorithms
    # ==========

    @functools.partial(oop.jax_tf.method_ro)
    def forward_kinematics(self) -> jtp.Array:
        """"""

        W_H_i = jaxsim.physics.algos.forward_kinematics.forward_kinematics_model(
            model=self.physics_model,
            q=self.data.model_state.joint_positions,
            xfb=self.data.model_state.xfb(),
        )

        return W_H_i

    @functools.partial(oop.jax_tf.method_ro)
    def inverse_dynamics(
        self,
        joint_accelerations: jtp.Vector | None = None,
        base_acceleration: jtp.Vector | None = None,
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """
        Compute inverse dynamics with the RNEA algorithm.

        Args:
            joint_accelerations: the joint accelerations to consider.
            base_acceleration:  the base acceleration in the active representation to consider.

        Returns:
            A tuple containing the 6D force in active representation applied to the base
            to obtain the considered base acceleration, and the joint torques to apply
            to obtain the considered joint accelerations.
        """

        # Build joint accelerations if not provided
        joint_accelerations = (
            joint_accelerations
            if joint_accelerations is not None
            else jnp.zeros_like(self.joint_positions())
        )

        # Build base acceleration if not provided
        base_acceleration = (
            base_acceleration if base_acceleration is not None else jnp.zeros(6)
        )

        if base_acceleration.size != 6:
            raise ValueError(base_acceleration.size)

        def to_inertial(C_vd_WB, W_H_C, C_v_WB, W_vl_WC):
            W_X_C = sixd.se3.SE3.from_matrix(W_H_C).adjoint()
            C_X_W = sixd.se3.SE3.from_matrix(W_H_C).inverse().adjoint()

            if self.velocity_representation != VelRepr.Mixed:
                return W_X_C @ C_vd_WB
            else:
                from jaxsim.math.cross import Cross

                C_v_WC = C_X_W @ jnp.hstack([W_vl_WC, jnp.zeros(3)])
                return W_X_C @ (C_vd_WB + Cross.vx(C_v_WC) @ C_v_WB)

        if self.velocity_representation is VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_vl_WC = W_vl_WW = jnp.zeros(3)

        elif self.velocity_representation is VelRepr.Body:
            W_H_C = W_H_B = self.base_transform()
            W_vl_WC = W_vl_WB = self.base_velocity()[0:3]

        elif self.velocity_representation is VelRepr.Mixed:
            W_H_B = self.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_vl_WC = W_vl_W_BW = self.base_velocity()[0:3]

        else:
            raise ValueError(self.velocity_representation)

        # We need to convert the derivative of the base acceleration to the Inertial
        # representation. In Mixed representation, this conversion is not a plain
        # transformation with just X, but it also involves a cross product in ℝ⁶.
        W_v̇_WB = to_inertial(
            C_vd_WB=base_acceleration,
            W_H_C=W_H_C,
            C_v_WB=self.base_velocity(),
            W_vl_WC=W_vl_WC,
        )

        # Compute RNEA
        W_f_B, tau = jaxsim.physics.algos.rnea.rnea(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            qdd=joint_accelerations,
            a0fb=W_v̇_WB,
            f_ext=self.data.model_input.f_ext,
        )

        # Adjust shape
        tau = jnp.atleast_1d(tau.squeeze())

        # Express W_f_B in the active representation
        f_B = self.inertial_to_active_representation(array=W_f_B, is_force=True)

        return f_B, tau

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["prefer_aba"])
    def forward_dynamics(
        self, tau: jtp.Vector | None = None, prefer_aba: float = True
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """"""

        return (
            self.forward_dynamics_aba(tau=tau)
            if prefer_aba
            else self.forward_dynamics_crb(tau=tau)
        )

    @functools.partial(oop.jax_tf.method_ro)
    def forward_dynamics_aba(
        self, tau: jtp.Vector | None = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """"""

        # Build joint torques if not provided
        tau = tau if tau is not None else jnp.zeros_like(self.joint_positions())

        # Compute ABA
        W_v̇_WB, s̈ = jaxsim.physics.algos.aba.aba(
            model=self.physics_model,
            xfb=self.data.model_state.xfb(),
            q=self.data.model_state.joint_positions,
            qd=self.data.model_state.joint_velocities,
            tau=tau,
            f_ext=self.data.model_input.f_ext,
        )

        def to_active(W_vd_WB, W_H_C, W_v_WB, W_vl_WC):
            C_X_W = sixd.se3.SE3.from_matrix(W_H_C).inverse().adjoint()

            if self.velocity_representation != VelRepr.Mixed:
                return C_X_W @ W_vd_WB
            else:
                from jaxsim.math.cross import Cross

                W_v_WC = jnp.hstack([W_vl_WC, jnp.zeros(3)])
                return C_X_W @ (W_vd_WB - Cross.vx(W_v_WC) @ W_v_WB)

        if self.velocity_representation is VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_vl_WC = W_vl_WW = jnp.zeros(3)

        elif self.velocity_representation is VelRepr.Body:
            W_H_C = W_H_B = self.base_transform()
            W_vl_WC = W_vl_WB = self.base_velocity()[0:3]

        elif self.velocity_representation is VelRepr.Mixed:
            W_H_B = self.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_vl_WC = W_vl_W_BW = self.base_velocity()[0:3]

        else:
            raise ValueError(self.velocity_representation)

        # We need to convert the derivative of the base acceleration to the active
        # representation. In Mixed representation, this conversion is not a plain
        # transformation with just X, but it also involves a cross product in ℝ⁶.
        C_v̇_WB = to_active(
            W_vd_WB=W_v̇_WB.squeeze(),
            W_H_C=W_H_C,
            W_v_WB=jnp.hstack(
                [
                    self.data.model_state.base_linear_velocity,
                    self.data.model_state.base_angular_velocity,
                ]
            ),
            W_vl_WC=W_vl_WC,
        )

        # Adjust shape
        s̈ = jnp.atleast_1d(s̈.squeeze())

        return C_v̇_WB, s̈

    @functools.partial(oop.jax_tf.method_ro)
    def forward_dynamics_crb(
        self, tau: jtp.Vector | None = None
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        """"""

        # Build joint torques if not provided
        τ = tau if tau is not None else jnp.zeros(shape=(self.dofs(),))
        τ = jnp.atleast_1d(τ.squeeze())
        τ = jnp.vstack(τ) if τ.size > 0 else jnp.empty(shape=(0, 1))

        # Extract motor parameters from the physics model
        GR = self.motor_gear_ratios()
        IM = self.motor_inertias()
        KV = jnp.diag(self.motor_viscous_frictions())

        # Compute auxiliary quantities
        Γ = jnp.diag(GR)
        K̅ᵥ = Γ.T @ KV @ Γ

        # Compute terms of the floating-base EoM
        M = self.free_floating_mass_matrix()
        h = jnp.vstack(self.free_floating_bias_forces())
        J = self.generalized_free_floating_jacobian()
        f_ext = jnp.vstack(self.external_forces().flatten())
        S = jnp.block([jnp.zeros(shape=(self.dofs(), 6)), jnp.eye(self.dofs())]).T

        # Configure the slice for motors
        sl_m = np.s_[M.shape[0] - self.dofs() :]

        # Add the motor related terms to the EoM
        M = M.at[sl_m, sl_m].set(M[sl_m, sl_m] + jnp.diag(Γ.T @ IM @ Γ))
        h = h.at[sl_m].set(h[sl_m] + K̅ᵥ @ self.joint_velocities()[:, None])
        S = S.at[sl_m].set(S[sl_m])

        # Compute the generalized acceleration by inverting the EoM
        ν̇ = jax.lax.select(
            pred=self.floating_base(),
            on_true=jnp.linalg.inv(M) @ ((S @ τ) - h + J.T @ f_ext),
            on_false=jnp.vstack(
                [
                    jnp.zeros(shape=(6, 1)),
                    jnp.linalg.inv(M[6:, 6:])
                    @ ((S @ τ)[6:] - h[6:] + J[:, 6:].T @ f_ext),
                ]
            ),
        ).squeeze()

        # Extract the base acceleration in the active representation.
        # Note that this is an apparent acceleration (relevant in Mixed representation),
        # therefore it cannot be always expressed in different frames with just a
        # 6D transformation X.
        v̇_WB = ν̇[0:6]

        # Extract the joint accelerations
        s̈ = jnp.atleast_1d(ν̇[6:])

        return v̇_WB, s̈

    # ======
    # Energy
    # ======

    @functools.partial(oop.jax_tf.method_ro)
    def mechanical_energy(self) -> jtp.Float:
        """"""

        K = self.kinetic_energy()
        U = self.potential_energy()

        return K + U

    @functools.partial(oop.jax_tf.method_ro)
    def kinetic_energy(self) -> jtp.Float:
        """"""

        with self.editable(validate=True) as m:
            m.set_velocity_representation(vel_repr=VelRepr.Body)

        nu = m.generalized_velocity()
        M = m.free_floating_mass_matrix()

        return 0.5 * nu.T @ M @ nu

    @functools.partial(oop.jax_tf.method_ro)
    def potential_energy(self) -> jtp.Float:
        """"""

        m = self.total_mass()
        W_p_CoM = jnp.hstack([self.com_position(), 1])
        gravity = self.physics_model.gravity[3:6].squeeze()

        return -(m * jnp.hstack([gravity, 0]) @ W_p_CoM)

    # ===========
    # Set targets
    # ===========

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["joint_names"])
    def set_joint_generalized_force_targets(
        self, forces: jtp.Vector, joint_names: tuple[str, ...] | None = None
    ) -> None:
        """"""

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

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["joint_names"])
    def reset_joint_positions(
        self, positions: jtp.Vector, joint_names: tuple[str, ...] | None = None
    ) -> None:
        """"""

        if joint_names is None:
            joint_names = self.joint_names()

        if positions.size != len(joint_names):
            raise ValueError("Wrong arguments size", positions.size, len(joint_names))

        if positions.size == 0:
            return

        # TODO: joint position limits

        self.data.model_state.joint_positions = jnp.atleast_1d(
            jnp.array(
                self.data.model_state.joint_positions.at[
                    self._joint_indices(joint_names=joint_names)
                ].set(positions),
                dtype=float,
            )
        )

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["joint_names"])
    def reset_joint_velocities(
        self, velocities: jtp.Vector, joint_names: tuple[str, ...] | None = None
    ) -> None:
        """"""

        if joint_names is None:
            joint_names = self.joint_names()

        if velocities.size != len(joint_names):
            raise ValueError("Wrong arguments size", velocities.size, len(joint_names))

        if velocities.size == 0:
            return

        # TODO: joint velocity limits

        self.data.model_state.joint_velocities = jnp.atleast_1d(
            jnp.array(
                self.data.model_state.joint_velocities.at[
                    self._joint_indices(joint_names=joint_names)
                ].set(velocities),
                dtype=float,
            )
        )

    @functools.partial(oop.jax_tf.method_rw)
    def reset_base_position(self, position: jtp.Vector) -> None:
        """"""

        self.data.model_state.base_position = jnp.array(position, dtype=float)

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["dcm"])
    def reset_base_orientation(self, orientation: jtp.Array, dcm: bool = False) -> None:
        """"""

        if dcm:
            to_wxyz = np.array([3, 0, 1, 2])
            orientation_xyzw = sixd.so3.SO3.from_matrix(
                orientation
            ).as_quaternion_xyzw()
            orientation = orientation_xyzw[to_wxyz]

        unit_quaternion = orientation / jnp.linalg.norm(orientation)
        self.data.model_state.base_quaternion = jnp.array(unit_quaternion, dtype=float)

    @functools.partial(oop.jax_tf.method_rw)
    def reset_base_transform(self, transform: jtp.Matrix) -> None:
        """"""

        if transform.shape != (4, 4):
            raise ValueError(transform.shape)

        self.reset_base_position(position=transform[0:3, 3])
        self.reset_base_orientation(orientation=transform[0:3, 0:3], dcm=True)

    @functools.partial(oop.jax_tf.method_rw)
    def reset_base_velocity(self, base_velocity: jtp.VectorJax) -> None:
        """"""

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

        self.data.model_state.base_linear_velocity = jnp.array(
            base_velocity_inertial[0:3], dtype=float
        )

        self.data.model_state.base_angular_velocity = jnp.array(
            base_velocity_inertial[3:6], dtype=float
        )

    # ===========
    # Integration
    # ===========

    @functools.partial(
        oop.jax_tf.method_rw,
        static_argnames=["sub_steps", "integrator_type", "terrain"],
        vmap_in_axes=(0, 0, 0, None, None, None, 0, None),
    )
    def integrate(
        self,
        t0: jtp.Float,
        tf: jtp.Float,
        sub_steps: int = 1,
        integrator_type: Optional[
            "jaxsim.simulation.ode_integration.IntegratorType"
        ] = None,
        terrain: soft_contacts.Terrain = soft_contacts.FlatTerrain(),
        contact_parameters: soft_contacts.SoftContactsParams = soft_contacts.SoftContactsParams(),
        clear_inputs: bool = False,
    ) -> StepData:
        """"""

        from jaxsim.simulation import ode_data, ode_integration
        from jaxsim.simulation.ode_integration import IntegratorType

        if integrator_type is None:
            integrator_type = IntegratorType.EulerForward

        x0 = ode_integration.ode.ode_data.ODEState(
            physics_model=self.data.model_state,
            soft_contacts=self.data.contact_state,
        )

        ode_input = ode_integration.ode.ode_data.ODEInput(
            physics_model=self.data.model_input
        )

        assert isinstance(integrator_type, IntegratorType)

        # Integrate the model dynamics
        ode_states, aux = ode_integration.ode_integration_fixed_step(
            x0=x0,
            t=jnp.array([t0, tf], dtype=float),
            ode_input=ode_input,
            physics_model=self.physics_model,
            soft_contacts_params=contact_parameters,
            num_sub_steps=sub_steps,
            terrain=terrain,
            integrator_type=integrator_type,
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

    # ==============
    # Motor dynamics
    # ==============

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["joint_names"])
    def set_motor_inertias(
        self, inertias: jtp.Vector, joint_names: tuple[str, ...] | None = None
    ) -> None:
        joint_names = joint_names or self.joint_names()

        if inertias.size != len(joint_names):
            raise ValueError("Wrong arguments size", inertias.size, len(joint_names))

        self.physics_model._joint_motor_inertia.update(
            dict(zip(self.physics_model._joint_motor_inertia, inertias))
        )

        logging.info("Setting attribute `motor_inertias`")

    @functools.partial(oop.jax_tf.method_rw, jit=False)
    def set_motor_gear_ratios(
        self, gear_ratios: jtp.Vector, joint_names: tuple[str, ...] | None = None
    ) -> None:
        joint_names = joint_names or self.joint_names()

        if gear_ratios.size != len(joint_names):
            raise ValueError("Wrong arguments size", gear_ratios.size, len(joint_names))

        # Check on gear ratios if motor_inertias are not zero
        for idx, gr in enumerate(gear_ratios):
            if gr != 0 and self.motor_inertias()[idx] == 0:
                raise ValueError(
                    f"Zero motor inertia with non-zero gear ratio found in position {idx}"
                )

        self.physics_model._joint_motor_gear_ratio.update(
            dict(zip(self.physics_model._joint_motor_gear_ratio, gear_ratios))
        )

        logging.info("Setting attribute `motor_gear_ratios`")

    @functools.partial(oop.jax_tf.method_rw, static_argnames=["joint_names"])
    def set_motor_viscous_frictions(
        self,
        viscous_frictions: jtp.Vector,
        joint_names: tuple[str, ...] | None = None,
    ) -> None:
        joint_names = joint_names or self.joint_names()

        if viscous_frictions.size != len(joint_names):
            raise ValueError(
                "Wrong arguments size", viscous_frictions.size, len(joint_names)
            )

        self.physics_model._joint_motor_viscous_friction.update(
            dict(
                zip(
                    self.physics_model._joint_motor_viscous_friction,
                    viscous_frictions,
                )
            )
        )

        logging.info("Setting attribute `motor_viscous_frictions`")

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def motor_inertias(self) -> jtp.Vector:
        return jnp.array(
            [*self.physics_model._joint_motor_inertia.values()], dtype=float
        )

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def motor_gear_ratios(self) -> jtp.Vector:
        return jnp.array(
            [*self.physics_model._joint_motor_gear_ratio.values()], dtype=float
        )

    @functools.partial(oop.jax_tf.method_ro, jit=False)
    def motor_viscous_frictions(self) -> jtp.Vector:
        return jnp.array(
            [*self.physics_model._joint_motor_viscous_friction.values()], dtype=float
        )

    # ===============
    # Private methods
    # ===============

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["is_force"])
    def inertial_to_active_representation(
        self, array: jtp.Array, is_force: bool = False
    ) -> jtp.Array:
        """"""

        W_array = array.squeeze()

        if W_array.size != 6:
            raise ValueError(W_array.size)

        if self.velocity_representation is VelRepr.Inertial:
            return W_array

        elif self.velocity_representation is VelRepr.Body:
            W_H_B = self.base_transform()

            if not is_force:
                B_Xv_W = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint()
                B_array = B_Xv_W @ W_array

            else:
                B_Xf_W = sixd.se3.SE3.from_matrix(W_H_B).adjoint().T
                B_array = B_Xf_W @ W_array

            return B_array

        elif self.velocity_representation is VelRepr.Mixed:
            W_H_BW = jnp.eye(4).at[0:3, 3].set(self.base_position())

            if not is_force:
                BW_Xv_W = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
                BW_array = BW_Xv_W @ W_array

            else:
                BW_Xf_W = sixd.se3.SE3.from_matrix(W_H_BW).adjoint().T
                BW_array = BW_Xf_W @ W_array

            return BW_array

        else:
            raise ValueError(self.velocity_representation)

    @functools.partial(oop.jax_tf.method_ro, static_argnames=["is_force"])
    def active_to_inertial_representation(
        self, array: jtp.Array, is_force: bool = False
    ) -> jtp.Array:
        """"""

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
                W_Xv_B: jtp.Array = sixd.se3.SE3.from_matrix(W_H_B).adjoint()
                W_array = W_Xv_B @ B_array

            else:
                W_Xf_B = sixd.se3.SE3.from_matrix(W_H_B).inverse().adjoint().T
                W_array = W_Xf_B @ B_array

            return W_array

        elif self.velocity_representation is VelRepr.Mixed:
            BW_array = array
            W_H_BW = jnp.eye(4).at[0:3, 3].set(self.base_position())

            if not is_force:
                W_Xv_BW: jtp.Array = sixd.se3.SE3.from_matrix(W_H_BW).adjoint()
                W_array = W_Xv_BW @ BW_array

            else:
                W_Xf_BW = sixd.se3.SE3.from_matrix(W_H_BW).inverse().adjoint().T
                W_array = W_Xf_BW @ BW_array

            return W_array

        else:
            raise ValueError(self.velocity_representation)

    def _joint_indices(self, joint_names: tuple[str, ...] | None = None) -> jtp.Vector:
        """"""

        if joint_names is None:
            joint_names = self.joint_names()

        if set(joint_names) - set(self.joint_names()) != set():
            raise ValueError("One or more joint names are not part of the model")

        # Note: joints share the same index as their child link, therefore the first
        #       joint has index=1. We need to subtract one to get the right entry of
        #       data stored in the PhysicsModelState arrays.
        joint_indices = [
            j.joint_description.index - 1 for j in self.joints(joint_names=joint_names)
        ]

        return np.array(joint_indices, dtype=int)
