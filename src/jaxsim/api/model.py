from __future__ import annotations

import dataclasses
import pathlib

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

    def __del__(self):
        print(f"deleting (id={id(self)})")

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


@jax.jit
def generalized_free_floating_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
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
        The (nL, 6, 6+dofs) array containing the stacked free-floating
        jacobians of the links. The first axis is the link index.
    """

    if output_vel_repr is None:
        output_vel_repr = data.velocity_representation

    # The body frame of the Link.jacobian method is the link frame L.
    # In this method, we want instead to use the base link B as body frame.
    # Therefore, we always get the link jacobian having Inertial as output
    # representation, and then we convert it to the desired output representation.
    match output_vel_repr:
        case VelRepr.Inertial:
            to_output = lambda J: J

        case VelRepr.Body:

            def to_output(W_J_Wi):
                W_H_B = data.base_transform()
                B_X_W = jaxlie.SE3.from_matrix(W_H_B).inverse().adjoint()
                return B_X_W @ W_J_Wi

        case VelRepr.Mixed:

            def to_output(W_J_Wi):
                W_H_B = data.base_transform()
                W_H_BW = jnp.array(W_H_B).at[0:3, 0:3].set(jnp.eye(3))
                BW_X_W = jaxlie.SE3.from_matrix(W_H_BW).inverse().adjoint()
                return BW_X_W @ W_J_Wi

        case _:
            raise ValueError(output_vel_repr)

    # Get the link jacobians in Inertial representation and convert them to the
    # target output representation in which the body frame is the base link B
    J_free_floating = jax.vmap(
        lambda i: to_output(js.link.jacobian(model=model, data=data, link_index=i))
    )(jnp.arange(model.number_of_links()))

    return J_free_floating


@functools.partial(jax.jit, static_argnames=["prefer_aba"])
def forward_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
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
        prefer_aba: Whether to prefer the ABA algorithm over the CRB one.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    forward_dynamics_fn = forward_dynamics_aba if prefer_aba else forward_dynamics_crb

    return forward_dynamics_fn(
        model=model, data=data, tau=joint_forces, external_forces=external_forces
    )


@jax.jit
def forward_dynamics_aba(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
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
        else:
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
    joint_forces: jtp.MatrixLike | None = None,
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
        joint_forces
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions())
    )

    # Build external forces if not provided
    external_forces = (
        external_forces
        if external_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    # Handle models with zero and one DoFs
    τ = jnp.atleast_1d(τ.squeeze())
    τ = jnp.vstack(τ) if τ.size > 0 else jnp.empty(shape=(0, 1))

    # Compute terms of the floating-base EoM
    M = free_floating_mass_matrix(model=model, data=data)
    h = jnp.vstack(free_floating_bias_forces(model=model, data=data))
    J = jnp.vstack(generalized_free_floating_jacobian(model=model, data=data))
    f_ext = jnp.vstack(external_forces.flatten())
    S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T

    # TODO: invert the Mss block exploiting sparsity defined by the parent array λ(i)
    if model.floating_base():
        ν̇ = jnp.linalg.inv(M) @ ((S @ τ) - h + J.T @ f_ext)
    else:
        v̇_WB = jnp.zeros(6)
        s̈ = jnp.linalg.inv(M[6:, 6:]) @ ((S @ τ)[6:] - h[6:] + J[:, 6:].T @ f_ext)
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
