from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
import rod
from jax_dataclasses import Static

import jaxsim.api as js
import jaxsim.physics.algos.aba
import jaxsim.physics.algos.crba
import jaxsim.physics.algos.forward_kinematics
import jaxsim.physics.algos.rnea
import jaxsim.physics.model.physics_model
import jaxsim.typing as jtp
from jaxsim.high_level.common import VelRepr
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

        # Store the origin of the model, in case downstream logic needs it
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

    # The body frame of the link.jacobian method is the link frame L.
    # In this method, we want instead to use the base link B as body frame.
    # Therefore, we always get the link jacobian having Inertial as output
    # representation, and then we convert it to the desired output representation.
    match output_vel_repr:
        case VelRepr.Inertial:
            to_output = lambda W_J_WL: W_J_WL

        case VelRepr.Body:

            def to_output(W_J_WL: jtp.Matrix) -> jtp.Matrix:
                W_H_B = data.base_transform()
                B_X_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
                return B_X_W @ W_J_WL

        case VelRepr.Mixed:

            def to_output(W_J_WL: jtp.Matrix) -> jtp.Matrix:
                W_H_B = data.base_transform()
                W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
                BW_X_W = jaxlie.SE3.from_matrix(W_H_BW).inverse().adjoint()
                return BW_X_W @ W_J_WL

        case _:
            raise ValueError(output_vel_repr)

    # Compute first the link jacobians having the active representation of `data`
    # as input representation (matching the one of ν), and inertial as output
    # representation (i.e. W_J_WL_C where C is C_ν).
    # Then, with to_output, we convert this jacobian to the desired output
    # representation, that can either be W (inertial), B (body), or B[W] (mixed).
    # This is necessary because for example the body-fixed free-floating jacobian
    # of a link is L_J_WL, but here being inside model we need B_J_WL.
    J_free_floating = jax.vmap(
        lambda i: to_output(
            W_J_WL=js.link.jacobian(
                model=model,
                data=data,
                link_index=i,
                output_vel_repr=VelRepr.Inertial,
            )
        )
    )(jnp.arange(model.number_of_links()))

    return J_free_floating


@functools.partial(jax.jit, static_argnames=["prefer_aba"])
def forward_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    external_forces: jtp.MatrixLike | None = None,
    prefer_aba: float = True,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        external_forces:
            The external forces to consider as a matrix of shape `(nL, 6)`.
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
        external_forces=external_forces,
    )


@jax.jit
def forward_dynamics_aba(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    external_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the ABA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        external_forces:
            The external forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    # Build joint torques if not provided
    τ = (
        joint_forces
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build external forces if not provided
    f_ext = (
        external_forces
        if external_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    )

    # Compute ABA
    W_v̇_WB, s̈ = jaxsim.physics.algos.aba.aba(
        model=model.physics_model,
        xfb=data.state.physics_model.xfb(),
        q=data.state.physics_model.joint_positions,
        qd=data.state.physics_model.joint_velocities,
        tau=τ,
        f_ext=f_ext,
    )

    def to_active(W_vd_WB, W_H_C, W_v_WB, W_vl_WC):
        C_X_W = jaxlie.SE3.from_matrix(W_H_C).inverse().adjoint()

        if data.velocity_representation != VelRepr.Mixed:
            return C_X_W @ W_vd_WB

        from jaxsim.math.cross import Cross

        W_v_WC = jnp.hstack([W_vl_WC, jnp.zeros(3)])
        return C_X_W @ (W_vd_WB - Cross.vx(W_v_WC) @ W_v_WB)

    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_vl_WC = W_vl_WW = jnp.zeros(3)

        case VelRepr.Body:
            W_H_C = W_H_B = data.base_transform()
            W_vl_WC = W_vl_WB = data.base_velocity()[0:3]

        case VelRepr.Mixed:
            W_H_B = data.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_vl_WC = W_vl_W_BW = data.base_velocity()[0:3]

        case _:
            raise ValueError(data.velocity_representation)

    # We need to convert the derivative of the base acceleration to the active
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    C_v̇_WB = to_active(
        W_vd_WB=W_v̇_WB.squeeze(),
        W_H_C=W_H_C,
        W_v_WB=jnp.hstack(
            [
                data.state.physics_model.base_linear_velocity,
                data.state.physics_model.base_angular_velocity,
            ]
        ),
        W_vl_WC=W_vl_WC,
    )

    # Adjust shape
    s̈ = jnp.atleast_1d(s̈.squeeze())

    return C_v̇_WB, s̈


@jax.jit
def forward_dynamics_crb(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    external_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the CRB algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        external_forces:
            The external forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.

    Note:
        Compared to ABA, this method could be significantly slower, especially for
        models with a large number of degrees of freedom.
    """

    # Build joint torques if not provided
    τ = (
        jnp.atleast_1d(joint_forces)
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build external forces if not provided
    f = (
        jnp.atleast_2d(external_forces)
        if external_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    # Compute terms of the floating-base EoM
    M = free_floating_mass_matrix(model=model, data=data)
    h = free_floating_bias_forces(model=model, data=data)
    S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T
    J = generalized_free_floating_jacobian(model=model, data=data)

    # TODO: invert the Mss block exploiting sparsity defined by the parent array λ(i)

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

    M_body = jaxsim.physics.algos.crba.crba(
        model=model.physics_model,
        q=data.state.physics_model.joint_positions,
    )

    match data.velocity_representation:
        case VelRepr.Body:
            return M_body

        case VelRepr.Inertial:
            zero_6n = jnp.zeros(shape=(6, model.dofs()))
            B_X_W = jaxlie.SE3.from_matrix(data.base_transform()).inverse().adjoint()

            invT = jnp.vstack(
                [
                    jnp.block([B_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(model.dofs())]),
                ]
            )

            return invT.T @ M_body @ invT

        case VelRepr.Mixed:
            zero_6n = jnp.zeros(shape=(6, model.dofs()))
            W_H_BW = data.base_transform().at[0:3, 3].set(jnp.zeros(3))
            BW_X_W = jaxlie.SE3.from_matrix(W_H_BW).inverse().adjoint()

            invT = jnp.vstack(
                [
                    jnp.block([BW_X_W, zero_6n]),
                    jnp.block([zero_6n.T, jnp.eye(model.dofs())]),
                ]
            )

            return invT.T @ M_body @ invT

        case _:
            raise ValueError(data.velocity_representation)


@jax.jit
def inverse_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_accelerations: jtp.Vector | None = None,
    base_acceleration: jtp.Vector | None = None,
    external_forces: jtp.Matrix | None = None,
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
        external_forces:
            The external forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D force in the active representation applied to the
        base to obtain the considered base acceleration, and the joint forces to apply
        to obtain the considered joint accelerations.
    """

    # Build joint accelerations if not provided
    joint_accelerations = (
        joint_accelerations
        if joint_accelerations is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build base acceleration if not provided
    base_acceleration = (
        base_acceleration if base_acceleration is not None else jnp.zeros(6)
    )

    external_forces = (
        external_forces
        if external_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    def to_inertial(C_v̇_WB, W_H_C, C_v_WB, W_vl_WC):
        W_X_C = jaxlie.SE3.from_matrix(W_H_C).adjoint()
        C_X_W = jaxlie.SE3.from_matrix(W_H_C).inverse().adjoint()

        if data.velocity_representation != VelRepr.Mixed:
            return W_X_C @ C_v̇_WB
        else:
            from jaxsim.math.cross import Cross

            C_v_WC = C_X_W @ jnp.hstack([W_vl_WC, jnp.zeros(3)])
            return W_X_C @ (C_v̇_WB + Cross.vx(C_v_WC) @ C_v_WB)

    match data.velocity_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)
            W_vl_WC = W_vl_WW = jnp.zeros(3)

        case VelRepr.Body:
            W_H_C = W_H_B = data.base_transform()
            W_vl_WC = W_vl_WB = data.base_velocity()[0:3]

        case VelRepr.Mixed:
            W_H_B = data.base_transform()
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_vl_WC = W_vl_W_BW = data.base_velocity()[0:3]

        case _:
            raise ValueError(data.velocity_representation)

    # We need to convert the derivative of the base acceleration to the Inertial
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    W_v̇_WB = to_inertial(
        C_v̇_WB=base_acceleration,
        W_H_C=W_H_C,
        C_v_WB=data.base_velocity(),
        W_vl_WC=W_vl_WC,
    )

    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        link_forces=external_forces,
        velocity_representation=data.velocity_representation,
    )

    # Compute RNEA
    with references.switch_velocity_representation(VelRepr.Inertial):
        W_f_B, τ = jaxsim.physics.algos.rnea.rnea(
            model=model.physics_model,
            xfb=data.state.physics_model.xfb(),
            q=data.state.physics_model.joint_positions,
            qd=data.state.physics_model.joint_velocities,
            qdd=joint_accelerations,
            a0fb=W_v̇_WB,
            f_ext=references.link_forces(model=model, data=data),
        )

    # Adjust shape
    τ = jnp.atleast_1d(τ.squeeze())

    # Express W_f_B in the active representation
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
    """
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
            external_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
        )
    ).astype(float)


@jax.jit
def free_floating_bias_forces(
    model: JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    """
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

        data_rnea.state.physics_model.base_linear_velocity = (
            data.state.physics_model.base_linear_velocity
        )

        data_rnea.state.physics_model.base_angular_velocity = (
            data.state.physics_model.base_angular_velocity
        )

        data_rnea.state.physics_model.joint_velocities = (
            data.state.physics_model.joint_velocities
        )

    return jnp.hstack(
        inverse_dynamics(
            model=model,
            data=data_rnea,
            # Set zero inputs:
            joint_accelerations=jnp.atleast_1d(jnp.zeros(model.dofs())),
            base_acceleration=jnp.zeros(6),
            external_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
        )
    ).astype(float)


# ==========================
# Other kinematic quantities
# ==========================


@jax.jit
def total_momentum(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Vector:
    """
    Compute the total momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The total momentum of the model.
    """

    # Compute the momentum in body-fixed velocity representation.
    # Note: the first 6 rows of the mass matrix define the jacobian of the
    #       floating-base momentum.
    with data.switch_velocity_representation(velocity_representation=VelRepr.Body):
        B_ν = data.generalized_velocity()
        M_B = free_floating_mass_matrix(model=model, data=data)

    # Compute the total momentum expressed in the base frame
    B_h = M_B[0:6, :] @ B_ν

    # Compute the 6D transformation matrix
    W_H_B = data.base_transform()
    B_X_W: jtp.Array = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()

    # Convert to inertial-fixed representation
    # (its coordinates transform like 6D forces)
    W_h = B_X_W.T @ B_h

    # Convert to the active representation of the model
    return js.data.JaxSimModelData.inertial_to_other_representation(
        array=W_h,
        other_representation=data.velocity_representation,
        transform=W_H_B,
        is_force=True,
    ).astype(float)


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
    W_p̃_CoM = jnp.hstack([com_position(model=model, data=data), 1])

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
    external_forces: jtp.MatrixLike | None = None,
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
        external_forces:
            The external forces to consider.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the new data of the model
        and the new state of the integrator.
    """

    integrator_state = integrator_state if integrator_state is not None else dict()

    # Extract the initial resources.
    t0_ns = data.time_ns
    state_x0 = data.state
    integrator_state_x0 = integrator_state

    # Step the dynamics forward.
    state_xf, integrator_state_xf = integrator.step(
        x0=state_x0,
        t0=jnp.array(t0_ns * 1e9).astype(float),
        dt=dt,
        params=integrator_state_x0,
        **dict(joint_forces=joint_forces, external_forces=external_forces),
    )

    return (
        # Store the new state of the model and the new time.
        data.replace(
            state=state_xf,
            time_ns=t0_ns + jnp.array(dt * 1e9).astype(jnp.uint64),
        ),
        integrator_state_xf,
    )
