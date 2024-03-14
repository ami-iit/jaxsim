from __future__ import annotations

import jax.lax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.math.inertia import Inertia
from jaxsim.math.joint_model import JointModel, supported_joint_motion
from jaxsim.parsers.descriptions import JointDescription, ModelDescription
from jaxsim.physics.model.ground_contact import GroundContact as ContactParameters
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class KynDynParameters(JaxsimDataclass):

    # Static
    link_names: Static[tuple[str]]
    parent_array: Static[jtp.Vector]
    support_body_array_bool: Static[jtp.Matrix]

    # Links
    link_parameters: LinkParameters

    # Contacts
    contact_parameters: ContactParameters

    # Joints
    joint_model: JointModel
    joint_parameters: JointParameters | None

    @staticmethod
    def build(model_description: ModelDescription) -> KynDynParameters:
        """
        Construct the kinematic and dynamic parameters of the model.

        Args:
            model_description: The parsed model description to consider.

        Returns:
            The kinematic and dynamic parameters of the model.

        Note:
            This class is meant to ease the management of parametric models in
            an automatic differentiation context.
        """

        # Extract the links ordered by their index.
        # The link index corresponds to the body index ∈ [0, num_bodies - 1].
        ordered_links = sorted(
            list(model_description.links_dict.values()),
            key=lambda l: l.index,
        )

        # Extract the joints ordered by their index.
        # The joint index matches the index of its child link, therefore it starts
        # from 1. Keep this in mind since this 1-indexing might introduce bugs.
        ordered_joints = sorted(
            list(model_description.joints_dict.values()),
            key=lambda j: j.index,
        )

        # ================
        # Links properties
        # ================

        # Create a list of link parameters objects.
        link_parameters_list = [
            LinkParameters.build_from_spatial_inertia(M=link.inertia)
            for link in ordered_links
        ]

        # Create a vectorized object of link parameters.
        link_parameters = jax.tree_util.tree_map(
            lambda *l: jnp.stack(l), *link_parameters_list
        )

        # =================
        # Joints properties
        # =================

        # Create a list of joint parameters objects.
        joint_parameters_list = [
            JointParameters.build_from_joint_description(joint_description=joint)
            for joint in ordered_joints
        ]

        # Create a vectorized object of joint parameters.
        joint_parameters = (
            jax.tree_util.tree_map(lambda *l: jnp.stack(l), *joint_parameters_list)
            if len(ordered_joints) > 0
            else None
        )

        # Create an object that defines the joint model (parent-to-child transforms).
        joint_model = JointModel.build(description=model_description)

        # ===============
        # Tree properties
        # ===============

        # Build the parent array λ(i) of the model.
        # Note: the parent of the base link is not set since it's not defined.
        parent_array_dict = {
            link.index: link.parent.index
            for link in ordered_links
            if link.parent is not None
        }
        parent_array = jnp.array([-1] + list(parent_array_dict.values()), dtype=int)

        # Instead of building the support parent array κ(i) of the model, having a
        # variable length that depends on the number of links connecting the root to
        # the i-th link, we build the corresponding boolean version.
        # Given a link index i, the boolean support parent array κb(i) is an array
        # with the same number of elements of λ(i) having the i-th element set to True
        # if the i-th link is in the support parent array κ(i), False otherwise.
        def κb(link_index: jtp.IntLike) -> jtp.Vector:
            κb = jnp.zeros(len(ordered_links), dtype=bool)

            carry0 = κb, link_index

            def scan_body(carry: tuple, i: jtp.Int) -> tuple[tuple, None]:

                κb, active_link_index = carry

                κb, active_link_index = jax.lax.cond(
                    pred=(i == active_link_index),
                    false_fun=lambda: (κb, active_link_index),
                    true_fun=lambda: (
                        κb.at[active_link_index].set(True),
                        parent_array[active_link_index],
                    ),
                )

                return (κb, active_link_index), None

            (κb, _), _ = jax.lax.scan(
                f=scan_body,
                init=carry0,
                xs=jnp.flip(jnp.arange(start=0, stop=len(ordered_links))),
            )

            return κb

        support_body_array_bool = jax.vmap(κb)(
            jnp.arange(start=0, stop=len(ordered_links))
        )

        return KynDynParameters(
            link_names=tuple(l.name for l in ordered_links),
            parent_array=parent_array,
            support_body_array_bool=support_body_array_bool,
            link_parameters=link_parameters,
            joint_model=joint_model,
            joint_parameters=joint_parameters,
            contact_parameters=ContactParameters.build_from(
                model_description=model_description
            ),
        )

    def __eq__(self, other: KynDynParameters) -> bool:

        if not isinstance(other, KynDynParameters):
            return False

        equal = True
        equal = equal and self.number_of_links() == other.number_of_links()
        equal = equal and self.number_of_joints() == other.number_of_joints()
        equal = equal and jnp.allclose(self.parent_array, other.parent_array)

        return equal

    def __hash__(self) -> int:

        h = 0
        h += hash(self.number_of_links())
        h += hash(self.number_of_joints())
        h += hash(tuple(self.parent_array.tolist()))

        return h

    # =============================
    # Helpers to extract parameters
    # =============================

    def number_of_links(self) -> int:
        """
        Return the number of links of the model.

        Returns:
            The number of links of the model.
        """

        return len(self.link_names)

    def number_of_joints(self) -> int:
        """
        Return the number of joints of the model.

        Returns:
            The number of joints of the model.
        """

        return len(self.joint_model.joint_names) - 1

    def support_body_array(self, link_index: jtp.IntLike) -> jtp.Vector:
        """
        Return the support parent array κ(i) of a link belonging to the model.

        Args:
            link_index: The index of the link.

        Returns:
            The support parent array κ(i) of the link.

        Note:
            This method returns a variable-length vector. In jit-compiled functions,
            it's better to use the (static) boolean version `support_body_array_bool`.
        """

        return jnp.array(
            jnp.where(self.support_body_array_bool[link_index])[0], dtype=int
        )

    # ========================
    # Quantities used by RBDAs
    # ========================

    @jax.jit
    def links_spatial_inertia(self) -> jtp.Array:
        """
        Return the spatial inertia of all links of the model.

        Returns:
            The spatial inertia of all links of the model.
        """

        return jax.vmap(LinkParameters.spatial_inertia)(self.link_parameters)

    @jax.jit
    def tree_transforms(self) -> jtp.Array:
        """
        Return the tree transforms of the model.

        Returns:
            The transforms
            :math:`{}^{\text{pre}(\text{i})} H_{\lambda(\text{i})}`
            of all joints of the model.
        """

        pre_Xi_λ = jax.vmap(
            lambda i: self.joint_model.parent_H_predecessor(joint_index=i)
            .inverse()
            .adjoint()
        )(jnp.arange(1, self.number_of_joints() + 1))

        return jnp.vstack(
            [
                jnp.zeros(shape=(1, 6, 6), dtype=float),
                pre_Xi_λ,
            ]
        )

    @jax.jit
    def joint_transforms(self, joint_positions: jtp.VectorLike) -> jtp.Array:
        """
        Return the transforms of the joints.

        Args:
            joint_positions: The joint positions.

        Returns:
            The stacked transforms
            :math:`{}^{\text{i}} \mathbf{H}_{\lambda(\text{i})}(s)`
            of each joint.
        """

        return self.joint_transforms_and_motion_subspaces(joint_positions)[0]

    @jax.jit
    def joint_motion_subspaces(self, joint_positions: jtp.VectorLike) -> jtp.Array:
        """
        Return the motion subspaces of the joints.

        Args:
            joint_positions: The joint positions.

        Returns:
            The stacked motion subspaces :math:`\mathbf{S}(s)` of each joint.
        """

        return self.joint_transforms_and_motion_subspaces(joint_positions)[1]

    @jax.jit
    def joint_transforms_and_motion_subspaces(
        self, joint_positions: jtp.VectorLike
    ) -> tuple[jtp.Array, jtp.Array]:
        """
        Return the transforms and the motion subspaces of the joints.

        Args:
            joint_positions: The joint positions.

        Returns:
            A tuple containing the stacked transforms
            :math:`{}^{\text{i}} \mathbf{H}_{\lambda(\text{i})}(s)`
            and the stacked motion subspaces :math:`\mathbf{S}(s)` of each joint.
        """

        λ_H_pre = jax.vmap(
            lambda i: self.joint_model.parent_H_predecessor(joint_index=i)
        )(jnp.arange(1, 1 + self.number_of_joints()))

        pre_H_suc_and_S = [
            supported_joint_motion(
                joint_type=self.joint_model.joint_types[index + 1],
                joint_position=s,
            )
            for index, s in enumerate(joint_positions)
        ]

        pre_H_suc = jnp.stack([jnp.eye(4)] + [H for H, _ in pre_H_suc_and_S])
        S = jnp.stack([jnp.vstack(jnp.zeros(6))] + [S for _, S in pre_H_suc_and_S])

        suc_H_i = jax.vmap(lambda i: self.joint_model.successor_H_child(joint_index=i))(
            jnp.arange(1, 1 + self.number_of_joints())
        )

        i_X_λ = jax.vmap(
            lambda λ_Hi_pre, pre_Hi_suc, suc_Hi_i: jaxlie.SE3.from_matrix(
                λ_Hi_pre @ pre_Hi_suc @ suc_Hi_i
            )
            .inverse()
            .adjoint()
        )(λ_H_pre, pre_H_suc, suc_H_i)

        return i_X_λ, S

    # ============================
    # Helpers to update parameters
    # ============================

    def set_link_mass(self, link_index: int, mass: jtp.FloatLike) -> KynDynParameters:
        """
        Set the mass of a link.

        Args:
            link_index: The index of the link.
            mass: The mass of the link.

        Returns:
            The updated kinematic and dynamic parameters of the model.
        """

        link_parameters = self.link_parameters.replace(
            mass=self.link_parameters.mass.at[link_index].set(mass)
        )

        return self.replace(link_parameters=link_parameters)

    def set_link_inertia(
        self, link_index: int, inertia: jtp.MatrixLike
    ) -> KynDynParameters:
        """
        Set the inertia tensor of a link.

        Args:
            link_index: The index of the link.
            inertia: The 3×3 inertia tensor of the link.

        Returns:
            The updated kinematic and dynamic parameters of the model.
        """

        inertia_elements = LinkParameters.flatten_inertia_tensor(I=inertia)

        link_parameters = self.link_parameters.replace(
            mass=self.link_parameters.inertia_elements.at[link_index].set(
                inertia_elements
            )
        )

        return self.replace(link_parameters=link_parameters)


@jax_dataclasses.pytree_dataclass
class JointParameters(JaxsimDataclass):

    friction_static: jtp.Float
    friction_viscous: jtp.Float

    position_limits_min: jtp.Float
    position_limits_max: jtp.Float

    position_limit_spring: jtp.Float
    position_limit_damper: jtp.Float

    @staticmethod
    def build_from_joint_description(
        joint_description: JointDescription,
    ) -> JointParameters:
        """"""

        s_min = joint_description.position_limit[0]
        s_max = joint_description.position_limit[1]

        position_limits_min = jnp.minimum(s_min, s_max)
        position_limits_max = jnp.maximum(s_min, s_max)

        friction_static = jnp.array(joint_description.friction_static).squeeze()
        friction_viscous = jnp.array(joint_description.friction_viscous).squeeze()

        position_limit_spring = jnp.array(
            joint_description.position_limit_spring
        ).squeeze()

        position_limit_damper = jnp.array(
            joint_description.position_limit_damper
        ).squeeze()

        return JointParameters(
            friction_static=friction_static.astype(float),
            friction_viscous=friction_viscous.astype(float),
            position_limits_min=position_limits_min.astype(float),
            position_limits_max=position_limits_max.astype(float),
            position_limit_spring=position_limit_spring.astype(float),
            position_limit_damper=position_limit_damper.astype(float),
        )


@jax_dataclasses.pytree_dataclass
class LinkParameters(JaxsimDataclass):

    mass: jtp.Float
    inertia_elements: jtp.Vector

    # The following is L_p_CoM, that is the translation between the link frame and
    # the link's center of mass, expressed in the coordinates of the link frame L.
    center_of_mass: jtp.Vector

    @staticmethod
    def build_from_spatial_inertia(M: jtp.Matrix) -> LinkParameters:
        """"""

        m, L_p_CoM, I = Inertia.to_params(M=M)

        return LinkParameters(
            mass=jnp.array(m).squeeze().astype(float),
            center_of_mass=jnp.atleast_1d(jnp.array(L_p_CoM).squeeze()).astype(float),
            inertia_elements=jnp.atleast_1d(I[jnp.triu_indices(3)].squeeze()).astype(
                float
            ),
        )

    @staticmethod
    def build_from_inertial_parameters(
        m: jtp.FloatLike, I: jtp.MatrixLike, c: jtp.VectorLike
    ) -> LinkParameters:

        return LinkParameters(
            mass=jnp.array(m).squeeze().astype(float),
            inertia_elements=jnp.atleast_1d(I[jnp.triu_indices(3)].squeeze()).astype(
                float
            ),
            center_of_mass=jnp.atleast_1d(c.squeeze()).astype(float),
        )

    @staticmethod
    def build_from_flat_parameters(parameters: jtp.VectorLike) -> LinkParameters:

        m = jnp.array(parameters[0]).squeeze().astype(float)
        c = jnp.atleast_1d(parameters[1:4].squeeze()).astype(float)
        I = jnp.atleast_1d(parameters[4:].squeeze()).astype(float)

        return LinkParameters(
            mass=m, inertia_elements=I[jnp.triu_indices(3)], center_of_mass=c
        )

    @staticmethod
    def parameters(params: LinkParameters) -> jtp.Vector:

        return jnp.hstack(
            [params.mass, params.center_of_mass.squeeze(), params.inertia_elements]
        )

    @staticmethod
    def inertia_tensor(params: LinkParameters) -> jtp.Matrix:

        return LinkParameters.unflatten_inertia_tensor(
            inertia_elements=params.inertia_elements
        )

    @staticmethod
    def spatial_inertia(params: LinkParameters) -> jtp.Matrix:

        return Inertia.to_sixd(
            mass=params.mass,
            I=LinkParameters.inertia_tensor(params),
            com=params.center_of_mass,
        )

    @staticmethod
    def flatten_inertia_tensor(I: jtp.Matrix) -> jtp.Vector:
        return jnp.atleast_1d(I[jnp.triu_indices(3)].squeeze())

    @staticmethod
    def unflatten_inertia_tensor(inertia_elements: jtp.Vector) -> jtp.Matrix:
        I = jnp.zeros([3, 3]).at[jnp.triu_indices(3)].set(inertia_elements.squeeze())
        return jnp.atleast_2d(jnp.where(I, I, I.T)).astype(float)
