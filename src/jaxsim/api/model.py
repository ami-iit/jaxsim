from __future__ import annotations

import copy
import dataclasses
import functools
import pathlib
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
import rod
from jax_dataclasses import Static

import jaxsim.api as js
import jaxsim.parsers.descriptions
import jaxsim.typing as jtp
from jaxsim.utils import HashlessObject, JaxsimDataclass, Mutability

from .common import VelRepr


@jax_dataclasses.pytree_dataclass
class JaxSimModel(JaxsimDataclass):
    """
    The JaxSim model defining the kinematics and dynamics of a robot.
    """

    model_name: Static[str]

    terrain: Static[jaxsim.terrain.Terrain] = dataclasses.field(
        default=jaxsim.terrain.FlatTerrain(), repr=False, compare=False, hash=False
    )

    built_from: Static[str | pathlib.Path | rod.Model | None] = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    description: Static[
        HashlessObject[jaxsim.parsers.descriptions.ModelDescription | None]
    ] = dataclasses.field(default=None, repr=False, compare=False, hash=False)

    kin_dyn_parameters: js.kin_dyn_parameters.KynDynParameters | None = (
        dataclasses.field(default=None, repr=False, compare=False, hash=False)
    )

    # ========================
    # Initialization and state
    # ========================

    @staticmethod
    def build_from_model_description(
        model_description: str | pathlib.Path | rod.Model,
        model_name: str | None = None,
        *,
        terrain: jaxsim.terrain.Terrain | None = None,
        is_urdf: bool | None = None,
        considered_joints: Sequence[str] | None = None,
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
            terrain:
                The optional terrain to consider.
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

        # Build the model
        model = JaxSimModel.build(
            model_description=intermediate_description,
            model_name=model_name,
            terrain=terrain,
        )

        # Store the origin of the model, in case downstream logic needs it
        with model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            model.built_from = model_description

        return model

    @staticmethod
    def build(
        model_description: jaxsim.parsers.descriptions.ModelDescription,
        model_name: str | None = None,
        *,
        terrain: jaxsim.terrain.Terrain | None = None,
    ) -> JaxSimModel:
        """
        Build a Model object from an intermediate model description.

        Args:
            model_description:
                The intermediate model description defining the kinematics and dynamics
                of the model.
            model_name:
                The optional name of the model overriding the physics model name.
            terrain:
                The optional terrain to consider.

        Returns:
            The built Model object.
        """

        # Set the model name (if not provided, use the one from the model description)
        model_name = model_name if model_name is not None else model_description.name

        # Build the model
        model = JaxSimModel(
            model_name=model_name,
            description=HashlessObject(obj=model_description),
            kin_dyn_parameters=js.kin_dyn_parameters.KynDynParameters.build(
                model_description=model_description
            ),
            terrain=terrain or JaxSimModel.__dataclass_fields__["terrain"].default,
        )

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

        return self.kin_dyn_parameters.number_of_links()

    def number_of_joints(self) -> jtp.Int:
        """
        Return the number of joints in the model.

        Returns:
            The number of joints in the model.
        """

        return self.kin_dyn_parameters.number_of_joints()

    # =================
    # Base link methods
    # =================

    def floating_base(self) -> bool:
        """
        Return whether the model has a floating base.

        Returns:
            True if the model is floating-base, False otherwise.
        """

        return bool(self.kin_dyn_parameters.joint_model.joint_dofs[0] == 6)

    def base_link(self) -> str:
        """
        Return the name of the base link.

        Returns:
            The name of the base link.

        Note:
            By default, the base link is the root of the kinematic tree.
        """

        return self.link_names()[0]

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

        return int(sum(self.kin_dyn_parameters.joint_model.joint_dofs[1:]))

    def joint_names(self) -> tuple[str, ...]:
        """
        Return the names of the joints in the model.

        Returns:
            The names of the joints in the model.
        """

        return self.kin_dyn_parameters.joint_model.joint_names[1:]

    # ====================
    # Link-related methods
    # ====================

    def link_names(self) -> tuple[str, ...]:
        """
        Return the names of the links in the model.

        Returns:
            The names of the links in the model.
        """

        return self.kin_dyn_parameters.link_names

    def frame_names(self) -> tuple[str, ...]:
        """
        Return the names of the links in the model.

        Returns:
            The names of the links in the model.
        """

        return tuple([frame.name for frame in self.description.get().frames])


# =====================
# Model post-processing
# =====================


def reduce(
    model: JaxSimModel,
    considered_joints: tuple[str, ...],
    locked_joint_positions: dict[str, jtp.Float] | None = None,
) -> JaxSimModel:
    """
    Reduce the model by lumping together the links connected by removed joints.

    Args:
        model: The model to reduce.
        considered_joints: The sequence of joints to consider.
        locked_joint_positions:
            A dictionary containing the positions of the joints to be considered
            in the reduction process. The removed joints in the reduced model
            will have their position locked to their value in this dictionary.
            If a joint is not part of the dictionary, its position is set to zero.
    """

    locked_joint_positions = (
        locked_joint_positions if locked_joint_positions is not None else {}
    )

    # If locked joints are passed, make sure that they are valid.
    if not set(locked_joint_positions).issubset(model.joint_names()):
        new_joints = set(model.joint_names()) - set(locked_joint_positions)
        raise ValueError(f"Passed joints not existing in the model: {new_joints}")

    # Copy the model description with a deep copy of the joints.
    intermediate_description = dataclasses.replace(
        model.description.get(), joints=copy.deepcopy(model.description.get().joints)
    )

    # Update the initial position of the joints.
    # This is necessary to compute the correct pose of the link pairs connected
    # to removed joints.
    for joint_name in set(model.joint_names()) - set(considered_joints):
        j = intermediate_description.joints_dict[joint_name]
        with j.mutable_context():
            j.initial_position = float(locked_joint_positions.get(joint_name, 0.0))

    # Reduce the model description.
    # If `considered_joints` contains joints not existing in the model,
    # the method will raise an exception.
    reduced_intermediate_description = intermediate_description.reduce(
        considered_joints=list(considered_joints)
    )

    # Build the reduced model
    reduced_model = JaxSimModel.build(
        model_description=reduced_intermediate_description,
        model_name=model.name(),
    )

    # Store the origin of the model, in case downstream logic needs it
    with reduced_model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
        reduced_model.built_from = model.built_from

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


@jax.jit
def link_spatial_inertia_matrices(model: JaxSimModel) -> jtp.Array:
    """
    Compute the spatial 6D inertia matrices of all links of the model.

    Args:
        model: The model to consider.

    Returns:
        A 3D array containing the stacked spatial 6D inertia matrices of the links.
    """

    return jax.vmap(js.kin_dyn_parameters.LinkParameters.spatial_inertia)(
        model.kin_dyn_parameters.link_parameters
    )


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

    W_H_LL = jaxsim.rbda.forward_kinematics_model(
        model=model,
        base_position=data.base_position(),
        base_quaternion=data.base_orientation(dcm=False),
        joint_positions=data.joint_positions(model=model),
    )

    return jnp.atleast_3d(W_H_LL).astype(float)


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def generalized_free_floating_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating jacobians of all links.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobians.

    Returns:
        The `(nL, 6, 6+dofs)` array containing the stacked free-floating
        jacobians of the links. The first axis is the link index.

    Note:
        The v-stacked version of the returned Jacobian array together with the
        flattened 6D forces of the links, are useful to compute the `J.T @ f`
        product of the multi-body EoM.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Compute the doubly-left free-floating full jacobian.
    B_J_full_WX_B, _ = jaxsim.rbda.jacobian_full_doubly_left(
        model=model,
        joint_positions=data.joint_positions(),
    )

    # Update the input velocity representation such that `J_WL_I @ I_ν`.
    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_B = data.base_transform()
            B_X_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
            B_J_full_WX_I = B_J_full_WX_W = B_J_full_WX_B @ jax.scipy.linalg.block_diag(
                B_X_W, jnp.eye(model.dofs())
            )

        case VelRepr.Body:
            B_J_full_WX_I = B_J_full_WX_B

        case VelRepr.Mixed:
            W_R_B = data.base_orientation(dcm=True)
            BW_H_B = jnp.eye(4).at[0:3, 0:3].set(W_R_B)
            B_X_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            B_J_full_WX_I = B_J_full_WX_BW = (
                B_J_full_WX_B
                @ jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))
            )

        case _:
            raise ValueError(data.velocity_representation)

    # Update the output velocity representation such that `O_v_WL_I = O_J_WL_I @ I_ν`.
    match output_vel_repr:
        case VelRepr.Inertial:
            W_H_B = data.base_transform()
            W_X_B = jaxlie.SE3.from_matrix(W_H_B).adjoint()
            O_J_full_WX_I = W_J_full_WX_I = W_X_B @ B_J_full_WX_I

        case VelRepr.Body:
            O_J_full_WX_I = B_J_full_WX_I

        case VelRepr.Mixed:
            W_R_B = data.base_orientation(dcm=True)
            BW_H_B = jnp.eye(4).at[0:3, 0:3].set(W_R_B)
            BW_X_B = jaxlie.SE3.from_matrix(BW_H_B).adjoint()
            O_J_full_WX_I = BW_J_full_WX_I = BW_X_B @ B_J_full_WX_I

        case _:
            raise ValueError(output_vel_repr)

    κ_bool = model.kin_dyn_parameters.support_body_array_bool

    O_J_WL_I = jax.vmap(
        lambda κ: jnp.where(
            jnp.hstack([jnp.ones(5), κ]), O_J_full_WX_I, jnp.zeros_like(O_J_full_WX_I)
        )
    )(κ_bool)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["prefer_aba"])
def forward_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    prefer_aba: float = True,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.
        prefer_aba: Whether to prefer the ABA algorithm over the CRB one.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    forward_dynamics_fn = forward_dynamics_aba if prefer_aba else forward_dynamics_crb

    return forward_dynamics_fn(
        model=model,
        data=data,
        joint_forces=joint_forces,
        link_forces=link_forces,
    )


@jax.jit
def forward_dynamics_aba(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the ABA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    # ============
    # Prepare data
    # ============

    # Build joint forces, if not provided.
    τ = (
        jnp.atleast_1d(joint_forces.squeeze())
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build link forces, if not provided.
    f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    )

    # Create a references object that simplifies converting among representations.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=τ,
        link_forces=f_L,
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Extract the link and joint serializations.
    link_names = model.link_names()
    joint_names = model.joint_names()

    # Extract the state in inertial-fixed representation.
    with data.switch_velocity_representation(VelRepr.Inertial):
        W_p_B = data.base_position()
        W_v_WB = data.base_velocity()
        W_Q_B = data.base_orientation(dcm=False)
        s = data.joint_positions(model=model, joint_names=joint_names)
        ṡ = data.joint_velocities(model=model, joint_names=joint_names)

    # Extract the inputs in inertial-fixed representation.
    with references.switch_velocity_representation(VelRepr.Inertial):
        W_f_L = references.link_forces(model=model, data=data, link_names=link_names)
        τ = references.joint_force_references(model=model, joint_names=joint_names)

    # ========================
    # Compute forward dynamics
    # ========================

    W_v̇_WB, s̈ = jaxsim.rbda.aba(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        joint_forces=τ,
        link_forces=W_f_L,
        standard_gravity=data.standard_gravity(),
    )

    # =============
    # Adjust output
    # =============

    def to_active(
        W_v̇_WB: jtp.Vector, W_H_C: jtp.Matrix, W_v_WB: jtp.Vector, W_v_WC: jtp.Vector
    ) -> jtp.Vector:
        """
        Helper to convert the inertial-fixed apparent base acceleration W_v̇_WB to
        another representation C_v̇_WB expressed in a generic frame C.
        """

        from jaxsim.math import Cross

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        C_X_W = jaxlie.SE3.from_matrix(W_H_C).inverse().adjoint()
        return C_X_W @ (W_v̇_WB - Cross.vx(W_v_WC) @ W_v_WB)

    match data.velocity_representation:
        case VelRepr.Inertial:
            # In this case C=W
            W_H_C = W_H_W = jnp.eye(4)
            W_v_WC = W_v_WW = jnp.zeros(6)

        case VelRepr.Body:
            # In this case C=B
            W_H_C = W_H_B = data.base_transform()
            W_v_WC = W_v_WB

        case VelRepr.Mixed:
            # In this case C=B[W]
            W_H_B = data.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_ṗ_B = data.base_velocity()[0:3]
            W_v_WC = W_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)

        case _:
            raise ValueError(data.velocity_representation)

    # We need to convert the derivative of the base velocity to the active
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    C_v̇_WB = to_active(
        W_v̇_WB=W_v̇_WB,
        W_H_C=W_H_C,
        W_v_WB=jnp.hstack(
            [
                data.state.physics_model.base_linear_velocity,
                data.state.physics_model.base_angular_velocity,
            ]
        ),
        W_v_WC=W_v_WC,
    )

    # The ABA algorithm already returns a zero base 6D acceleration for
    # fixed-based models. However, the to_active function introduces an
    # additional acceleration component in Mixed representation.
    # Here below we make sure that the base acceleration is zero.
    C_v̇_WB = C_v̇_WB if model.floating_base() else jnp.zeros(6)

    return C_v̇_WB.astype(float), s̈.astype(float)


@jax.jit
def forward_dynamics_crb(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the CRB algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.

    Note:
        Compared to ABA, this method could be significantly slower, especially for
        models with a large number of degrees of freedom.
    """

    # ============
    # Prepare data
    # ============

    # Build joint torques if not provided
    τ = (
        jnp.atleast_1d(joint_forces)
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build external forces if not provided
    f = (
        jnp.atleast_2d(link_forces)
        if link_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    # Compute terms of the floating-base EoM
    M = free_floating_mass_matrix(model=model, data=data)
    h = free_floating_bias_forces(model=model, data=data)
    S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T
    J = generalized_free_floating_jacobian(model=model, data=data)

    # TODO: invert the Mss block exploiting sparsity defined by the parent array λ(i)

    # ========================
    # Compute forward dynamics
    # ========================

    if model.floating_base():
        # l: number of links.
        # g: generalized coordinates, 6 + number of joints.
        JTf = jnp.einsum("l6g,l6->g", J, f)
        ν̇ = jnp.linalg.solve(M, S @ τ - h + JTf)

    else:
        # l: number of links.
        # j: number of joints.
        JTf = jnp.einsum("l6j,l6->j", J[:, :, 6:], f)
        s̈ = jnp.linalg.solve(M[6:, 6:], τ - h[6:] + JTf)

        v̇_WB = jnp.zeros(6)
        ν̇ = jnp.hstack([v̇_WB, s̈.squeeze()])

    # =============
    # Adjust output
    # =============

    # Extract the base acceleration in the active representation.
    # Note that this is an apparent acceleration (relevant in Mixed representation),
    # therefore it cannot be always expressed in different frames with just a
    # 6D transformation X.
    v̇_WB = ν̇[0:6].squeeze().astype(float)

    # Extract the joint accelerations
    s̈ = jnp.atleast_1d(ν̇[6:].squeeze()).astype(float)

    return v̇_WB, s̈


@jax.jit
def free_floating_mass_matrix(
    model: JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the free-floating mass matrix of the model with the CRBA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The free-floating mass matrix of the model.
    """

    M_body = jaxsim.rbda.crba(
        model=model,
        joint_positions=data.state.physics_model.joint_positions,
    )

    match data.velocity_representation:
        case VelRepr.Body:
            return M_body

        case VelRepr.Inertial:

            B_X_W = jaxlie.SE3.from_matrix(data.base_transform()).inverse().adjoint()
            invT = jax.scipy.linalg.block_diag(B_X_W, jnp.eye(model.dofs()))

            return invT.T @ M_body @ invT

        case VelRepr.Mixed:

            BW_H_B = data.base_transform().at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            invT = jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))

            return invT.T @ M_body @ invT

        case _:
            raise ValueError(data.velocity_representation)


@jax.jit
def inverse_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_accelerations: jtp.VectorLike | None = None,
    base_acceleration: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute inverse dynamics with the RNEA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_accelerations:
            The joint accelerations to consider as a vector of shape `(dofs,)`.
        base_acceleration:
            The base acceleration to consider as a vector of shape `(6,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D force in the active representation applied to the
        base to obtain the considered base acceleration, and the joint forces to apply
        to obtain the considered joint accelerations.
    """

    # ============
    # Prepare data
    # ============

    # Build joint accelerations, if not provided.
    s̈ = (
        jnp.atleast_1d(jnp.array(joint_accelerations).squeeze())
        if joint_accelerations is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build base acceleration, if not provided.
    v̇_WB = (
        jnp.array(base_acceleration).squeeze()
        if base_acceleration is not None
        else jnp.zeros(6)
    )

    # Build link forces, if not provided.
    f_L = (
        jnp.atleast_2d(jnp.array(link_forces).squeeze())
        if link_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    def to_inertial(C_v̇_WB, W_H_C, C_v_WB, W_v_WC):
        """
        Helper to convert the active representation of the base acceleration C_v̇_WB
        expressed in a generic frame C to the inertial-fixed representation W_v̇_WB.
        """

        from jaxsim.math import Cross

        W_X_C = jaxlie.SE3.from_matrix(W_H_C).adjoint()
        C_X_W = jaxlie.SE3.from_matrix(W_H_C).inverse().adjoint()
        C_v_WC = C_X_W @ W_v_WC

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        return W_X_C @ (C_v̇_WB + Cross.vx(C_v_WC) @ C_v_WB)

    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_v_WC = W_v_WW = jnp.zeros(6)

        case VelRepr.Body:
            W_H_C = W_H_B = data.base_transform()
            with data.switch_velocity_representation(VelRepr.Inertial):
                W_v_WC = W_v_WB = data.base_velocity()

        case VelRepr.Mixed:
            W_H_B = data.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_ṗ_B = data.base_velocity()[0:3]
            W_v_WC = W_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)

        case _:
            raise ValueError(data.velocity_representation)

    # We need to convert the derivative of the base acceleration to the Inertial
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    W_v̇_WB = to_inertial(
        C_v̇_WB=v̇_WB,
        W_H_C=W_H_C,
        C_v_WB=data.base_velocity(),
        W_v_WC=W_v_WC,
    )

    # Create a references object that simplifies converting among representations.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        link_forces=f_L,
        velocity_representation=data.velocity_representation,
    )

    # Extract the link and joint serializations.
    link_names = model.link_names()
    joint_names = model.joint_names()

    # Extract the state in inertial-fixed representation.
    with data.switch_velocity_representation(VelRepr.Inertial):
        W_p_B = data.base_position()
        W_v_WB = data.base_velocity()
        W_Q_B = data.base_orientation(dcm=False)
        s = data.joint_positions(model=model, joint_names=joint_names)
        ṡ = data.joint_velocities(model=model, joint_names=joint_names)

    # Extract the inputs in inertial-fixed representation.
    with references.switch_velocity_representation(VelRepr.Inertial):
        W_f_L = references.link_forces(model=model, data=data, link_names=link_names)

    # ========================
    # Compute inverse dynamics
    # ========================

    W_f_B, τ = jaxsim.rbda.rnea(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        base_linear_acceleration=W_v̇_WB[0:3],
        base_angular_acceleration=W_v̇_WB[3:6],
        joint_accelerations=s̈,
        link_forces=W_f_L,
        standard_gravity=data.standard_gravity(),
    )

    # =============
    # Adjust output
    # =============

    # Express W_f_B in the active representation.
    f_B = js.data.JaxSimModelData.inertial_to_other_representation(
        array=W_f_B,
        other_representation=data.velocity_representation,
        transform=data.base_transform(),
        is_force=True,
    ).squeeze()

    return f_B.astype(float), τ.astype(float)


@jax.jit
def free_floating_gravity_forces(
    model: JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the free-floating gravity forces :math:`g(\mathbf{q})` of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The free-floating gravity forces of the model.
    """

    # Build a zeroed state
    data_rnea = js.data.JaxSimModelData.zero(
        model=model, velocity_representation=data.velocity_representation
    )

    # Set just the generalized position
    with data_rnea.mutable_context(
        mutability=Mutability.MUTABLE, restore_after_exception=False
    ):

        data_rnea.state.physics_model.base_position = (
            data.state.physics_model.base_position
        )

        data_rnea.state.physics_model.base_quaternion = (
            data.state.physics_model.base_quaternion
        )

        data_rnea.state.physics_model.joint_positions = (
            data.state.physics_model.joint_positions
        )

    return jnp.hstack(
        inverse_dynamics(
            model=model,
            data=data_rnea,
            # Set zero inputs:
            joint_accelerations=jnp.atleast_1d(jnp.zeros(model.dofs())),
            base_acceleration=jnp.zeros(6),
            link_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
        )
    ).astype(float)


@jax.jit
def free_floating_bias_forces(
    model: JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the free-floating bias forces :math:`h(\mathbf{q}, \boldsymbol{\nu})`
    of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The free-floating bias forces of the model.
    """

    # Build a zeroed state
    data_rnea = js.data.JaxSimModelData.zero(
        model=model, velocity_representation=data.velocity_representation
    )

    # Set the generalized position and generalized velocity
    with data_rnea.mutable_context(
        mutability=Mutability.MUTABLE, restore_after_exception=False
    ):

        data_rnea.state.physics_model.base_position = (
            data.state.physics_model.base_position
        )

        data_rnea.state.physics_model.base_quaternion = (
            data.state.physics_model.base_quaternion
        )

        data_rnea.state.physics_model.joint_positions = (
            data.state.physics_model.joint_positions
        )

        data_rnea.state.physics_model.joint_velocities = (
            data.state.physics_model.joint_velocities
        )

        # Make sure that base velocity is zero for fixed-base model.
        if model.floating_base():
            data_rnea.state.physics_model.base_linear_velocity = (
                data.state.physics_model.base_linear_velocity
            )

            data_rnea.state.physics_model.base_angular_velocity = (
                data.state.physics_model.base_angular_velocity
            )

    return jnp.hstack(
        inverse_dynamics(
            model=model,
            data=data_rnea,
            # Set zero inputs:
            joint_accelerations=jnp.atleast_1d(jnp.zeros(model.dofs())),
            base_acceleration=jnp.zeros(6),
            link_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
        )
    ).astype(float)


# ==========================
# Other kinematic quantities
# ==========================


@jax.jit
def locked_spatial_inertia(
    model: JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the locked 6D inertia matrix of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The locked 6D inertia matrix of the model.
    """

    return total_momentum_jacobian(model=model, data=data)[:, 0:6]


@jax.jit
def total_momentum(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Vector:
    """
    Compute the total momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The total momentum of the model in the active velocity representation.
    """

    ν = data.generalized_velocity()
    Jh = total_momentum_jacobian(model=model, data=data)

    return Jh @ ν


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def total_momentum_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the jacobian of the total momentum.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr: The output velocity representation of the jacobian.

    Returns:
        The jacobian of the total momentum of the model in the active representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    if output_vel_repr is data.velocity_representation:
        return free_floating_mass_matrix(model=model, data=data)[0:6]

    with data.switch_velocity_representation(VelRepr.Body):
        B_Jh_B = free_floating_mass_matrix(model=model, data=data)[0:6]

    match data.velocity_representation:
        case VelRepr.Body:
            B_Jh = B_Jh_B

        case VelRepr.Inertial:
            B_X_W = jaxlie.SE3.from_matrix(data.base_transform()).inverse().adjoint()
            B_Jh = B_Jh_B @ jax.scipy.linalg.block_diag(B_X_W, jnp.eye(model.dofs()))

        case VelRepr.Mixed:
            BW_H_B = data.base_transform().at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            B_Jh = B_Jh_B @ jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))

        case _:
            raise ValueError(data.velocity_representation)

    match output_vel_repr:
        case VelRepr.Body:
            return B_Jh

        case VelRepr.Inertial:
            W_H_B = data.base_transform()
            B_Xv_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
            W_Xf_B = B_Xv_W.T
            W_Jh = W_Xf_B @ B_Jh
            return W_Jh

        case VelRepr.Mixed:
            BW_H_B = data.base_transform().at[0:3, 3].set(jnp.zeros(3))
            B_Xv_BW = jaxlie.SE3.from_matrix(BW_H_B).inverse().adjoint()
            BW_Xf_B = B_Xv_BW.T
            BW_Jh = BW_Xf_B @ B_Jh
            return BW_Jh

        case _:
            raise ValueError(output_vel_repr)


@jax.jit
def average_velocity(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Vector:
    """
    Compute the average velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The average velocity of the model computed in the base frame and expressed
        in the active representation.
    """

    ν = data.generalized_velocity()
    J = average_velocity_jacobian(model=model, data=data)

    return J @ ν


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def average_velocity_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the Jacobian of the average velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr: The output velocity representation of the jacobian.

    Returns:
        The Jacobian of the average centroidal velocity of the model in the desired
        representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Depending on the velocity representation, the frame G is either G[W] or G[B].
    G_J = js.com.average_centroidal_velocity_jacobian(model=model, data=data)

    match output_vel_repr:

        case VelRepr.Inertial:

            GW_J = G_J
            W_p_CoM = js.com.com_position(model=model, data=data)

            W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            W_X_GW = jaxlie.SE3.from_matrix(W_H_GW).adjoint()

            return W_X_GW @ GW_J

        case VelRepr.Body:

            GB_J = G_J
            W_p_B = data.base_position()
            W_p_CoM = js.com.com_position(model=model, data=data)
            B_R_W = data.base_orientation(dcm=True).transpose()

            B_H_GB = jnp.eye(4).at[0:3, 3].set(B_R_W @ (W_p_CoM - W_p_B))
            B_X_GB = jaxlie.SE3.from_matrix(B_H_GB).adjoint()

            return B_X_GB @ GB_J

        case VelRepr.Mixed:

            GW_J = G_J
            W_p_B = data.base_position()
            W_p_CoM = js.com.com_position(model=model, data=data)

            BW_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM - W_p_B)
            BW_X_GW = jaxlie.SE3.from_matrix(BW_H_GW).adjoint()

            return BW_X_GW @ GW_J


# ========================
# Other dynamic quantities
# ========================


@jax.jit
def link_bias_accelerations(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
) -> jtp.Vector:
    r"""
    Compute the bias accelerations of the links of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The bias accelerations of the links of the model.

    Note:
        This function computes the component of the total 6D acceleration not due to
        the joint or base acceleration.
        It is often called :math:`\dot{J} \boldsymbol{\nu}`.
    """

    # ================================================
    # Compute the body-fixed zero base 6D acceleration
    # ================================================

    # Compute the base transform.
    W_H_B = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_quaternion_xyzw(
            xyzw=jaxsim.math.Quaternion.to_xyzw(wxyz=data.base_orientation())
        ),
        translation=data.base_position(),
    ).as_matrix()

    def other_representation_to_inertial(
        C_v̇_WB: jtp.Vector, C_v_WB: jtp.Vector, W_H_C: jtp.Matrix, W_v_WC: jtp.Vector
    ) -> jtp.Vector:
        """
        Helper to convert the active representation of the base acceleration C_v̇_WB
        expressed in a generic frame C to the inertial-fixed representation W_v̇_WB.
        """

        W_X_C = jaxlie.SE3.from_matrix(W_H_C).adjoint()
        C_X_W = jaxlie.SE3.from_matrix(W_H_C).inverse().adjoint()

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        return W_X_C @ (C_v̇_WB + jaxsim.math.Cross.vx(C_X_W @ W_v_WC) @ C_v_WB)

    # Here we initialize a zero 6D acceleration in the active representation, and
    # convert it to inertial-fixed. This is a useful intermediate representation
    # because the apparent acceleration W_v̇_WB is equal to the intrinsic acceleration
    # W_a_WB, and intrinsic accelerations can be expressed in different frames through
    # a simple C_X_W 6D transform.
    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_v_WC = W_v_WW = jnp.zeros(6)
            with data.switch_velocity_representation(VelRepr.Inertial):
                C_v_WB = W_v_WB = data.base_velocity()

        case VelRepr.Body:
            W_H_C = W_H_B
            with data.switch_velocity_representation(VelRepr.Inertial):
                W_v_WC = W_v_WB = data.base_velocity()
            with data.switch_velocity_representation(VelRepr.Body):
                C_v_WB = B_v_WB = data.base_velocity()

        case VelRepr.Mixed:
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_H_C = W_H_BW
            with data.switch_velocity_representation(VelRepr.Mixed):
                W_ṗ_B = data.base_velocity()[0:3]
                W_v_WC = W_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)
            with data.switch_velocity_representation(VelRepr.Mixed):
                C_v_WB = BW_v_WB = data.base_velocity()
        case _:
            raise ValueError(data.velocity_representation)

    # Convert a zero 6D acceleration from the active representation to inertial-fixed.
    W_v̇_WB = other_representation_to_inertial(
        C_v̇_WB=jnp.zeros(6), C_v_WB=C_v_WB, W_H_C=W_H_C, W_v_WC=W_v_WC
    )

    # ===================================
    # Initialize buffers and prepare data
    # ===================================

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute 6D transforms of the base velocity.
    B_X_W = jaxsim.math.Adjoint.from_transform(transform=W_H_B, inverse=True)

    # Compute the parent-to-child adjoints and the motion subspaces of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi, S = model.kin_dyn_parameters.joint_transforms_and_motion_subspaces(
        joint_positions=data.joint_positions(), base_transform=W_H_B
    )

    # Allocate the buffer to store the body-fixed link velocities.
    L_v_WL = jnp.zeros(shape=(model.number_of_links(), 6))

    # Store the base velocity.
    with data.switch_velocity_representation(VelRepr.Body):
        B_v_WB = data.base_velocity()
        L_v_WL = L_v_WL.at[0].set(B_v_WB)

    # Get the joint velocities.
    ṡ = data.joint_velocities(model=model, joint_names=model.joint_names())

    # Allocate the buffer to store the body-fixed link accelerations,
    # and initialize the base acceleration.
    L_v̇_WL = jnp.zeros(shape=(model.number_of_links(), 6))
    L_v̇_WL = L_v̇_WL.at[0].set(B_X_W @ W_v̇_WB)

    # ======================================
    # Propagate accelerations and velocities
    # ======================================

    # The computation of the bias forces is similar to the forward pass of RNEA,
    # this time with zero base and joint accelerations. Furthermore, here we do
    # not remove gravity during the propagation.

    # Initialize the loop.
    Carry = tuple[jtp.MatrixJax, jtp.MatrixJax]
    carry0: Carry = (L_v_WL, L_v̇_WL)

    def propagate_accelerations(carry: Carry, i: jtp.Int) -> tuple[Carry, None]:
        # Initialize index and unpack the carry.
        ii = i - 1
        v, a = carry

        # Get the motion subspace of the joint.
        Si = S[i].squeeze()

        # Project the joint velocity into its motion subspace.
        vJ = Si * ṡ[ii]

        # Propagate the link body-fixed velocity.
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        # Propagate the link body-fixed acceleration considering zero joint acceleration.
        s̈ = 0.0
        a_i = i_X_λi[i] @ a[λ[i]] + Si * s̈ + jaxsim.math.Cross.vx(v[i]) @ vJ
        a = a.at[i].set(a_i)

        return (v, a), None

    # Compute the body-fixed velocity and body-fixed apparent acceleration of the links.
    (L_v_WL, L_v̇_WL), _ = (
        jax.lax.scan(
            f=propagate_accelerations,
            init=carry0,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(L_v_WL, L_v̇_WL), None]
    )

    # ===================================================================
    # Convert the body-fixed 6D acceleration to the active representation
    # ===================================================================

    def body_to_other_representation(
        L_v̇_WL: jtp.Vector, L_v_WL: jtp.Vector, C_H_L: jtp.Matrix, L_v_CL: jtp.Vector
    ) -> jtp.Vector:
        """
        Helper to convert the body-fixed apparent acceleration L_v̇_WL to
        another representation C_v̇_WL expressed in a generic frame C.
        """

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        C_X_L = jaxsim.math.Adjoint.from_transform(transform=C_H_L)
        return C_X_L @ (L_v̇_WL + jaxsim.math.Cross.vx(L_v_CL) @ L_v_WL)

    match data.velocity_representation:
        case VelRepr.Body:
            C_H_L = L_H_L = jnp.stack([jnp.eye(4)] * model.number_of_links())
            L_v_CL = L_v_LL = jnp.zeros(shape=(model.number_of_links(), 6))

        case VelRepr.Inertial:
            C_H_L = W_H_L = js.model.forward_kinematics(model=model, data=data)
            L_v_CL = L_v_WL

        case VelRepr.Mixed:
            W_H_L = js.model.forward_kinematics(model=model, data=data)
            LW_H_L = jax.vmap(lambda W_H_L: W_H_L.at[0:3, 3].set(jnp.zeros(3)))(W_H_L)
            C_H_L = LW_H_L
            L_v_CL = L_v_LW_L = jax.vmap(lambda v: v.at[0:3].set(jnp.zeros(3)))(L_v_WL)

        case _:
            raise ValueError(data.velocity_representation)

    # Convert from body-fixed to the active representation.
    O_v̇_WL = jax.vmap(body_to_other_representation)(
        L_v̇_WL=L_v̇_WL, L_v_WL=L_v_WL, C_H_L=C_H_L, L_v_CL=L_v_CL
    )

    return O_v̇_WL


@jax.jit
def link_contact_forces(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    """
    Compute the 6D contact forces of all links of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        A (nL, 6) array containing the stacked 6D contact forces of the links,
        expressed in the frame corresponding to the active representation.
    """

    # Compute the 6D forces applied to each collidable point expressed in the
    # inertial frame.
    with data.switch_velocity_representation(VelRepr.Inertial):
        W_f_Ci = js.contact.collidable_point_forces(model=model, data=data)

    # Construct the vector defining the parent link index of each collidable point.
    # We use this vector to sum the 6D forces of all collidable points rigidly
    # attached to the same link.
    parent_link_index_of_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )

    # Sum the forces of all collidable points rigidly attached to a body.
    # Since the contact forces W_f_Ci are expressed in the world frame,
    # we don't need any coordinate transformation.
    W_f_Li = jax.vmap(
        lambda nc: (
            jnp.vstack(
                jnp.equal(parent_link_index_of_collidable_points, nc).astype(int)
            )
            * W_f_Ci
        ).sum(axis=0)
    )(jnp.arange(model.number_of_links()))

    # Convert the 6D forces to the active representation.
    f_Li = jax.vmap(
        lambda W_f_L: data.inertial_to_other_representation(
            array=W_f_L,
            other_representation=data.velocity_representation,
            transform=data.base_transform(),
            is_force=True,
        )
    )(W_f_Li)

    return f_Li


# ======
# Energy
# ======


@jax.jit
def mechanical_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the mechanical energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The mechanical energy of the model.
    """

    K = kinetic_energy(model=model, data=data)
    U = potential_energy(model=model, data=data)

    return (K + U).astype(float)


@jax.jit
def kinetic_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the kinetic energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The kinetic energy of the model.
    """

    with data.switch_velocity_representation(velocity_representation=VelRepr.Body):
        B_ν = data.generalized_velocity()
        M_B = free_floating_mass_matrix(model=model, data=data)

    K = 0.5 * B_ν.T @ M_B @ B_ν
    return K.squeeze().astype(float)


@jax.jit
def potential_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the potential energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The potential energy of the model.
    """

    m = total_mass(model=model)
    gravity = data.gravity.squeeze()
    W_p̃_CoM = jnp.hstack([js.com.com_position(model=model, data=data), 1])

    U = -jnp.hstack([gravity, 0]) @ (m * W_p̃_CoM)
    return U.squeeze().astype(float)


# ==========
# Simulation
# ==========


@jax.jit
def step(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    dt: jtp.FloatLike,
    integrator: jaxsim.integrators.Integrator,
    integrator_state: dict[str, Any] | None = None,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    **kwargs,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    """
    Perform a simulation step.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        dt: The time step to consider.
        integrator: The integrator to use.
        integrator_state: The state of the integrator.
        joint_forces: The joint forces to consider.
        link_forces:
            The link 6D forces to consider.
            The frame in which they are expressed must be `data.velocity_representation`.
        kwargs: Additional kwargs to pass to the integrator.

    Returns:
        A tuple containing the new data of the model
        and the new state of the integrator.
    """

    integrator_kwargs = kwargs if kwargs is not None else dict()
    integrator_state = integrator_state if integrator_state is not None else dict()

    # Extract the initial resources.
    t0_ns = data.time_ns
    state_x0 = data.state
    integrator_state_x0 = integrator_state

    # Step the dynamics forward.
    state_xf, integrator_state_xf = integrator.step(
        x0=state_x0,
        t0=jnp.array(t0_ns / 1e9).astype(float),
        dt=dt,
        params=integrator_state_x0,
        **(
            dict(joint_forces=joint_forces, link_forces=link_forces) | integrator_kwargs
        ),
    )

    return (
        # Store the new state of the model and the new time.
        data.replace(
            state=state_xf,
            time_ns=t0_ns + jnp.array(dt * 1e9).astype(jnp.uint64),
        ),
        integrator_state_xf,
    )
