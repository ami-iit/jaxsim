from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import JointGenericAxis, JointType, ModelDescription
from jaxsim.parsers.kinematic_graph import KinematicGraphTransforms

from .rotation import Rotation
from .transform import Transform


@jax_dataclasses.pytree_dataclass
class JointModel:
    """
    Class describing the joint kinematics of a robot model.

    Attributes:
        λ_H_pre:
            The homogeneous transformation between the parent link and
            the predecessor frame of each joint.
        suc_H_i:
            The homogeneous transformation between the successor frame and
            the child link of each joint.
        joint_dofs: The number of DoFs of each joint.
        joint_names: The names of each joint.
        joint_types: The types of each joint.

    Note:
        Due to the presence of the static attributes, this class needs to be created
        already in a vectorized form. In other words, it cannot be created using vmap.
    """

    λ_H_pre: jtp.Array
    suc_H_i: jtp.Array

    joint_dofs: Static[tuple[int, ...]]
    joint_names: Static[tuple[str, ...]]
    joint_types: Static[tuple[int, ...]]
    joint_axis: Static[tuple[JointGenericAxis, ...]]

    @staticmethod
    def build(description: ModelDescription) -> JointModel:
        """
        Build the joint model of a model description.

        Args:
            description: The model description to consider.

        Returns:
            The joint model of the considered model description.
        """

        # The link index is equal to its body index: [0, number_of_bodies - 1].
        ordered_links = sorted(
            list(description.links_dict.values()),
            key=lambda l: l.index,
        )

        # Note: the joint index is equal to its child link index, therefore it
        # starts from 1.
        ordered_joints = sorted(
            list(description.joints_dict.values()),
            key=lambda j: j.index,
        )

        # Allocate the parent-to-predecessor and successor-to-child transforms.
        λ_H_pre = jnp.zeros(shape=(1 + len(ordered_joints), 4, 4), dtype=float)
        suc_H_i = jnp.zeros(shape=(1 + len(ordered_joints), 4, 4), dtype=float)

        # Initialize an identical parent-to-predecessor transform for the joint
        # between the world frame W and the base link B.
        λ_H_pre = λ_H_pre.at[0].set(jnp.eye(4))

        # Initialize the successor-to-child transform of the joint between the
        # world frame W and the base link B.
        # We store here the optional transform between the root frame of the model
        # and the base link frame (this is needed only if the pose of the link frame
        # w.r.t. the implicit __model__ SDF frame is not the identity).
        suc_H_i = suc_H_i.at[0].set(ordered_links[0].pose)

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=description)

        # Compute the parent-to-predecessor and successor-to-child transforms for
        # each joint belonging to the model.
        # Note that the joint indices starts from i=1 given our joint model,
        # therefore the entries at index 0 are not updated.
        for joint in ordered_joints:
            λ_H_pre = λ_H_pre.at[joint.index].set(
                fk.relative_transform(relative_to=joint.parent.name, name=joint.name)
            )
            suc_H_i = suc_H_i.at[joint.index].set(
                fk.relative_transform(relative_to=joint.name, name=joint.child.name)
            )

        # Define the DoFs of the base link.
        base_dofs = 0 if description.fixed_base else 6

        # We always add a dummy fixed joint between world and base.
        # TODO: Port floating-base support also at this level, not only in RBDAs.
        return JointModel(
            λ_H_pre=λ_H_pre,
            suc_H_i=suc_H_i,
            # Static attributes
            joint_dofs=tuple([base_dofs] + [1 for _ in ordered_joints]),
            joint_names=tuple(["world_to_base"] + [j.name for j in ordered_joints]),
            joint_types=tuple([JointType.Fixed] + [j.jtype for j in ordered_joints]),
            joint_axis=tuple(JointGenericAxis(axis=j.axis) for j in ordered_joints),
        )

    def parent_H_child(
        self, joint_index: jtp.IntLike, joint_position: jtp.VectorLike
    ) -> tuple[jtp.Matrix, jtp.Array]:
        r"""
        Compute the homogeneous transformation between the parent link and
        the child link of a joint, and the corresponding motion subspace.

        Args:
            joint_index: The index of the joint.
            joint_position: The position of the joint.

        Returns:
            A tuple containing the homogeneous transformation
            :math:`{}^{\lambda(i)} \mathbf{H}_i(s)`
            and the motion subspace :math:`\mathbf{S}(s)`.
        """

        i = joint_index
        s = joint_position

        # Get the components of the joint model.
        λ_Hi_pre = self.parent_H_predecessor(joint_index=i)
        pre_Hi_suc, S = self.predecessor_H_successor(joint_index=i, joint_position=s)
        suc_Hi_i = self.successor_H_child(joint_index=i)

        # Compose all the transforms.
        return λ_Hi_pre @ pre_Hi_suc @ suc_Hi_i, S

    @jax.jit
    def child_H_parent(
        self, joint_index: jtp.IntLike, joint_position: jtp.VectorLike
    ) -> tuple[jtp.Matrix, jtp.Array]:
        r"""
        Compute the homogeneous transformation between the child link and
        the parent link of a joint, and the corresponding motion subspace.

        Args:
            joint_index: The index of the joint.
            joint_position: The position of the joint.

        Returns:
            A tuple containing the homogeneous transformation
            :math:`{}^{i} \mathbf{H}_{\lambda(i)}(s)`
            and the motion subspace :math:`\mathbf{S}(s)`.
        """

        λ_Hi_i, S = self.parent_H_child(
            joint_index=joint_index, joint_position=joint_position
        )

        i_Hi_λ = Transform.inverse(λ_Hi_i)

        return i_Hi_λ, S

    def parent_H_predecessor(self, joint_index: jtp.IntLike) -> jtp.Matrix:
        r"""
        Return the homogeneous transformation between the parent link and
        the predecessor frame of a joint.

        Args:
            joint_index: The index of the joint.

        Returns:
            The homogeneous transformation
            :math:`{}^{\lambda(i)} \mathbf{H}_{\text{pre}(i)}`.
        """

        return self.λ_H_pre[joint_index]

    def predecessor_H_successor(
        self, joint_index: jtp.IntLike, joint_position: jtp.VectorLike
    ) -> tuple[jtp.Matrix, jtp.Array]:
        r"""
        Compute the homogeneous transformation between the predecessor and
        the successor frame of a joint, and the corresponding motion subspace.

        Args:
            joint_index: The index of the joint.
            joint_position: The position of the joint.

        Returns:
            A tuple containing the homogeneous transformation
            :math:`{}^{\text{pre}(i)} \mathbf{H}_{\text{suc}(i)}(s)`
            and the motion subspace :math:`\mathbf{S}(s)`.
        """

        pre_H_suc, S = supported_joint_motion(
            self.joint_types[joint_index],
            joint_position,
            self.joint_axis[joint_index].axis,
        )

        return pre_H_suc, S

    def successor_H_child(self, joint_index: jtp.IntLike) -> jtp.Matrix:
        r"""
        Return the homogeneous transformation between the successor frame and
        the child link of a joint.

        Args:
            joint_index: The index of the joint.

        Returns:
            The homogeneous transformation
            :math:`{}^{\text{suc}(i)} \mathbf{H}_i`.
        """

        return self.suc_H_i[joint_index]


@jax.jit
def supported_joint_motion(
    joint_type: jtp.IntLike,
    joint_position: jtp.VectorLike,
    joint_axis: jtp.VectorLike | None = None,
    /,
) -> tuple[jtp.Matrix, jtp.Array]:
    """
    Compute the homogeneous transformation and motion subspace of a joint.

    Args:
        joint_type: The type of the joint.
        joint_position: The position of the joint.
        joint_axis: The optional 3D axis of rotation or translation of the joint.

    Returns:
        A tuple containing the homogeneous transformation and the motion subspace.
    """

    # Prepare the joint position
    s = jnp.array(joint_position).astype(float)

    def compute_F() -> tuple[jtp.Matrix, jtp.Array]:
        return jaxlie.SE3.identity(), jnp.zeros(shape=(6, 1))

    def compute_R() -> tuple[jtp.Matrix, jtp.Array]:

        # Get the additional argument specifying the joint axis.
        # This is a metadata required by only some joint types.
        axis = jnp.array(joint_axis).astype(float).squeeze()

        pre_H_suc = jaxlie.SE3.from_matrix(
            matrix=jnp.eye(4).at[:3, :3].set(Rotation.from_axis_angle(vector=s * axis))
        )

        S = jnp.vstack(jnp.hstack([jnp.zeros(3), axis]))

        return pre_H_suc, S

    def compute_P() -> tuple[jtp.Matrix, jtp.Array]:

        # Get the additional argument specifying the joint axis.
        # This is a metadata required by only some joint types.
        axis = jnp.array(joint_axis).astype(float).squeeze()

        pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.identity(),
            translation=jnp.array(s * axis),
        )

        S = jnp.vstack(jnp.hstack([axis, jnp.zeros(3)]))

        return pre_H_suc, S

    pre_H_suc, S = jax.lax.switch(
        index=joint_type,
        branches=(
            compute_F,  # JointType.Fixed
            compute_R,  # JointType.Revolute
            compute_P,  # JointType.Prismatic
        ),
    )

    return pre_H_suc.as_matrix(), S
