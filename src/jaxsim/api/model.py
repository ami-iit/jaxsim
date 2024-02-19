from __future__ import annotations

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import jax_dataclasses
import rod
from jax_dataclasses import Static

import jaxsim.api as js
import jaxsim.physics.algos.forward_kinematics
import jaxsim.physics.model.physics_model
import jaxsim.typing as jtp
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.utils import JaxsimDataclass, Mutability


@jax_dataclasses.pytree_dataclass
class JaxSimModel(JaxsimDataclass):
    """
    The JaxSim model defining the kinematics and dynamics of a robot.
    """

    model_name: Static[str]

    physics_model: jaxsim.physics.model.physics_model.PhysicsModel = dataclasses.field(
        repr=False
    )

    terrain: Static[Terrain] = dataclasses.field(default=FlatTerrain(), repr=False)

    built_from: Static[str | pathlib.Path | rod.Model | None] = dataclasses.field(
        repr=False, default=None
    )

    _number_of_links: Static[int] = dataclasses.field(
        init=False, repr=False, default=None
    )

    _number_of_joints: Static[int] = dataclasses.field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):

        # These attributes are Static so that we can use `jax.vmap` and `jax.lax.scan`
        # over the all links and joints
        with self.mutable_context(
            mutability=Mutability.MUTABLE_NO_VALIDATION,
            restore_after_exception=False,
        ):
            self._number_of_links = len(self.physics_model.description.links_dict)
            self._number_of_joints = len(self.physics_model.description.joints_dict)

    # ========================
    # Initialization and state
    # ========================

    @staticmethod
    def build_from_model_description(
        model_description: str | pathlib.Path | rod.Model,
        model_name: str | None = None,
        gravity: jtp.Array = jaxsim.physics.default_gravity(),
        is_urdf: bool | None = None,
        considered_joints: list[str] | None = None,
    ) -> JaxSimModel:
        """
        Build a Model object from a model description.

        Args:
            model_description:
                A path to an SDF/URDF file, a string containing
                its content, or a pre-parsed/pre-built rod model.
            model_name:
                The optional name of the model that overrides the one in
                the description.
            gravity: The 3D gravity vector.
            is_urdf:
                Whether the model description is a URDF or an SDF. This is
                automatically inferred if the model description is a path to a file.
            considered_joints:
                The list of joints to consider. If None, all joints are considered.

        Returns:
            The built Model object.
        """

        import jaxsim.parsers.rod

        # Parse the input resource (either a path to file or a string with the URDF/SDF)
        # and build the -intermediate- model description
        intermediate_description = jaxsim.parsers.rod.build_model_description(
            model_description=model_description, is_urdf=is_urdf
        )

        # Lump links together if not all joints are considered.
        # Note: this procedure assigns a zero position to all joints not considered.
        if considered_joints is not None:
            intermediate_description = intermediate_description.reduce(
                considered_joints=considered_joints
            )

        # Create the physics model from the model description
        physics_model = jaxsim.physics.model.physics_model.PhysicsModel.build_from(
            model_description=intermediate_description, gravity=gravity
        )

        # Build the model
        model = JaxSimModel.build(physics_model=physics_model, model_name=model_name)

        with model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            model.built_from = model_description

        return model

    @staticmethod
    def build(
        physics_model: jaxsim.physics.model.physics_model.PhysicsModel,
        model_name: str | None = None,
    ) -> JaxSimModel:
        """
        Build a Model object from a physics model.

        Args:
            physics_model: The physics model.
            model_name:
                The optional name of the model overriding the physics model name.

        Returns:
            The built Model object.
        """

        # Set the model name (if not provided, use the one from the model description)
        model_name = (
            model_name if model_name is not None else physics_model.description.name
        )

        # Build the model
        model = JaxSimModel(physics_model=physics_model, model_name=model_name)  # noqa

        return model

    # ==========
    # Properties
    # ==========

    def name(self) -> str:
        """
        Return the name of the model.

        Returns:
            The name of the model.
        """

        return self.model_name

    def number_of_links(self) -> jtp.Int:
        """
        Return the number of links in the model.

        Returns:
            The number of links in the model.

        Note:
            The base link is included in the count and its index is always 0.
        """

        return self._number_of_links

    def number_of_joints(self) -> jtp.Int:
        """
        Return the number of joints in the model.

        Returns:
            The number of joints in the model.
        """

        return self._number_of_joints

    # =================
    # Base link methods
    # =================

    def floating_base(self) -> bool:
        """
        Return whether the model has a floating base.

        Returns:
            True if the model is floating-base, False otherwise.
        """

        return self.physics_model.is_floating_base

    def base_link(self) -> str:
        """
        Return the name of the base link.

        Returns:
            The name of the base link.
        """

        return self.physics_model.description.root.name

    # =====================
    # Joint-related methods
    # =====================

    def dofs(self) -> int:
        """
        Return the number of degrees of freedom of the model.

        Returns:
            The number of degrees of freedom of the model.

        Note:
            We do not yet support multi-DoF joints, therefore this is always equal to
            the number of joints. In the future, this could be different.
        """

        return len(self.physics_model.description.joints_dict)

    def joint_names(self) -> tuple[str, ...]:
        """
        Return the names of the joints in the model.

        Returns:
            The names of the joints in the model.
        """

        return tuple(self.physics_model.description.joints_dict.keys())

    # ====================
    # Link-related methods
    # ====================

    def link_names(self) -> tuple[str, ...]:
        """
        Return the names of the links in the model.

        Returns:
            The names of the links in the model.
        """

        return tuple(self.physics_model.description.links_dict.keys())


# =====================
# Model post-processing
# =====================


def reduce(model: JaxSimModel, considered_joints: tuple[str, ...]) -> JaxSimModel:
    """
    Reduce the model by lumping together the links connected by removed joints.

    Args:
        model: The model to reduce.
        considered_joints: The sequence of joints to consider.

    Note:
        If considered_joints contains joints not existing in the model, the method
        will raise an exception. If considered_joints is empty, the method will
        return a copy of the input model.
    """

    if len(considered_joints) == 0:
        return model.copy()

    # Reduce the model description.
    # If considered_joints contains joints not existing in the model, the method
    # will raise an exception.
    reduced_intermediate_description = model.physics_model.description.reduce(
        considered_joints=list(considered_joints)
    )

    # Create the physics model from the reduced model description
    physics_model = jaxsim.physics.model.physics_model.PhysicsModel.build_from(
        model_description=reduced_intermediate_description,
        gravity=model.physics_model.gravity[0:3],
    )

    # Build the reduced model
    reduced_model = JaxSimModel.build(
        physics_model=physics_model, model_name=model.name()
    )

    return reduced_model


# ===================
# Inertial properties
# ===================


@jax.jit
def total_mass(model: JaxSimModel) -> jtp.Float:
    """
    Compute the total mass of the model.

    Args:
        model: The model to consider.

    Returns:
        The total mass of the model.
    """

    return (
        jax.vmap(lambda idx: js.link.mass(model=model, link_index=idx))(
            jnp.arange(model.number_of_links())
        )
        .sum()
        .astype(float)
    )


# ==============
# Center of mass
# ==============


@jax.jit
def com_position(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Vector:
    """
    Compute the position of the center of mass of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the center of mass of the model w.r.t. the world frame.
    """

    m = total_mass(model=model)

    W_H_L = forward_kinematics(model=model, data=data)
    W_H_B = data.base_transform()
    B_H_W = jaxlie.SE3.from_matrix(W_H_B).inverse().as_matrix()

    def B_p̃_LCoM(i) -> jtp.Vector:
        m = js.link.mass(model=model, link_index=i)
        L_p_LCoM = js.link.com_position(
            model=model, data=data, link_index=i, in_link_frame=True
        )
        return m * B_H_W @ W_H_L[i] @ jnp.hstack([L_p_LCoM, 1])

    com_links = jax.vmap(B_p̃_LCoM)(jnp.arange(model.number_of_links()))

    B_p̃_CoM = (1 / m) * com_links.sum(axis=0)
    B_p̃_CoM = B_p̃_CoM.at[3].set(1)

    return (W_H_B @ B_p̃_CoM)[0:3].astype(float)


# ==============================
# Rigid Body Dynamics Algorithms
# ==============================


@jax.jit
def forward_kinematics(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Array:
    """
    Compute the SE(3) transforms from the world frame to the frames of all links.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        A (nL, 4, 4) array containing the stacked SE(3) transforms of the links.
        The first axis is the link index.
    """

    W_H_LL = jaxsim.physics.algos.forward_kinematics.forward_kinematics_model(
        model=model.physics_model,
        q=data.state.physics_model.joint_positions,
        xfb=data.state.physics_model.xfb(),
    )

    return jnp.atleast_3d(W_H_LL).astype(float)
