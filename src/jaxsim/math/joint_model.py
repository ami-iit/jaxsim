from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import (
    JointDescriptor,
    JointGenericAxis,
    JointType,
    ModelDescription,
)

from .rotation import Rotation


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

    λ_H_pre: jax.Array
    suc_H_i: jax.Array

    joint_dofs: Static[tuple[int, ...]]
    joint_names: Static[tuple[str, ...]]
    joint_types: Static[tuple[JointType | JointDescriptor, ...]]

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

        # Compute the parent-to-predecessor and successor-to-child transforms for
        # each joint belonging to the model.
        # Note that the joint indices starts from i=1 given our joint model,
        # therefore the entries at index 0 are not updated.
        for joint in ordered_joints:
            λ_H_pre = λ_H_pre.at[joint.index].set(
                description.relative_transform(
                    relative_to=joint.parent.name,
                    name=joint.name,
                )
            )
            suc_H_i = suc_H_i.at[joint.index].set(
                description.relative_transform(
                    relative_to=joint.name, name=joint.child.name
                )
            )

        # Define the DoFs of the base link.
        base_dofs = 0 if description.fixed_base else 6

        # We always add a dummy fixed joint between world and base.
        # TODO: Port floating-base support also at this level, not only in RBDAs.
        return JointModel(
            λ_H_pre=λ_H_pre,
            suc_H_i=suc_H_i,
            # Static attributes
            joint_dofs=tuple([base_dofs] + [int(1) for _ in ordered_joints]),
            joint_names=tuple(["world_to_base"] + [j.name for j in ordered_joints]),
            joint_types=tuple([JointType.F] + [j.jtype for j in ordered_joints]),
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

        i_Hi_λ = jaxlie.SE3.from_matrix(λ_Hi_i).inverse().as_matrix()

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
            joint_type=self.joint_types[joint_index],
            joint_position=joint_position,
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


@functools.partial(jax.jit, static_argnames=["joint_type"])
def supported_joint_motion(
    joint_type: JointType | JointDescriptor, joint_position: jtp.VectorLike
) -> tuple[jtp.Matrix, jtp.Array]:
    """
    Compute the homogeneous transformation and motion subspace of a joint.

    Args:
        joint_type: The type of the joint.
        joint_position: The position of the joint.

    Returns:
        A tuple containing the homogeneous transformation and the motion subspace.
    """

    if isinstance(joint_type, JointType):
        code = joint_type
    elif isinstance(joint_type, JointDescriptor):
        code = joint_type.code
    else:
        raise ValueError(joint_type)

    # Prepare the joint position
    s = jnp.array(joint_position).astype(float)

    match code:

        case JointType.R:
            joint_type: JointGenericAxis

            pre_H_suc = jaxlie.SE3.from_rotation(
                rotation=jaxlie.SO3.from_matrix(
                    Rotation.from_axis_angle(vector=s * joint_type.axis)
                )
            )

            S = jnp.vstack(jnp.hstack([jnp.zeros(3), joint_type.axis.squeeze()]))

        case JointType.P:
            joint_type: JointGenericAxis

            pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.identity(),
                translation=jnp.array(s * joint_type.axis),
            )

            S = jnp.vstack(jnp.hstack([joint_type.axis.squeeze(), jnp.zeros(3)]))

        case JointType.F:
            raise ValueError("Fixed joints shouldn't be here")

        case JointType.Rx:

            pre_H_suc = jaxlie.SE3.from_rotation(
                rotation=jaxlie.SO3.from_x_radians(theta=s)
            )

            S = jnp.vstack([0, 0, 0, 1.0, 0, 0])

        case JointType.Ry:

            pre_H_suc = jaxlie.SE3.from_rotation(
                rotation=jaxlie.SO3.from_y_radians(theta=s)
            )

            S = jnp.vstack([0, 0, 0, 0, 1.0, 0])

        case JointType.Rz:

            pre_H_suc = jaxlie.SE3.from_rotation(
                rotation=jaxlie.SO3.from_z_radians(theta=s)
            )

            S = jnp.vstack([0, 0, 0, 0, 0, 1.0])

        case JointType.Px:

            pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.identity(),
                translation=jnp.array([s, 0.0, 0.0]),
            )

            S = jnp.vstack([1.0, 0, 0, 0, 0, 0])

        case JointType.Py:

            pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.identity(),
                translation=jnp.array([0.0, s, 0.0]),
            )

            S = jnp.vstack([0, 1.0, 0, 0, 0, 0])

        case JointType.Pz:

            pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
                rotation=jaxlie.SO3.identity(),
                translation=jnp.array([0.0, 0.0, s]),
            )

            S = jnp.vstack([0, 0, 1.0, 0, 0, 0])

        case _:
            raise ValueError(joint_type)

    return pre_H_suc.as_matrix(), S
