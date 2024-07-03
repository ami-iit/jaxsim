from __future__ import annotations

import dataclasses

import jax.lax
import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Inertia, JointModel, supported_joint_motion
from jaxsim.parsers.descriptions import JointDescription, ModelDescription
from jaxsim.utils import HashedNumpyArray, JaxsimDataclass


@jax_dataclasses.pytree_dataclass(eq=False, unsafe_hash=False)
class KynDynParameters(JaxsimDataclass):
    r"""
    Class storing the kinematic and dynamic parameters of a model.

    Attributes:
        link_names: The names of the links.
        parent_array: The parent array :math:`\lambda(i)` of the model.
        support_body_array_bool:
            The boolean support parent array :math:`\kappa_{b}(i)` of the model.
        link_parameters: The parameters of the links.
        frame_parameters: The parameters of the frames.
        contact_parameters: The parameters of the collidable points.
        joint_model: The joint model of the model.
        joint_parameters: The parameters of the joints.
    """

    # Static
    link_names: Static[tuple[str]]
    _parent_array: Static[HashedNumpyArray]
    _support_body_array_bool: Static[HashedNumpyArray]

    # Links
    link_parameters: LinkParameters

    # Contacts
    contact_parameters: ContactParameters

    # Frames
    frame_parameters: FrameParameters

    # Joints
    joint_model: JointModel
    joint_parameters: JointParameters | None

    @property
    def parent_array(self) -> jtp.Vector:
        return self._parent_array.get()

    @property
    def support_body_array_bool(self) -> jtp.Matrix:
        return self._support_body_array_bool.get()

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
            LinkParameters.build_from_spatial_inertia(index=link.index, M=link.inertia)
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
            else JointParameters(
                index=jnp.array([], dtype=int),
                friction_static=jnp.array([], dtype=float),
                friction_viscous=jnp.array([], dtype=float),
                position_limits_min=jnp.array([], dtype=float),
                position_limits_max=jnp.array([], dtype=float),
                position_limit_spring=jnp.array([], dtype=float),
                position_limit_damper=jnp.array([], dtype=float),
            )
        )

        # Create an object that defines the joint model (parent-to-child transforms).
        joint_model = JointModel.build(description=model_description)

        # ===================
        # Contacts properties
        # ===================

        # Create the object storing the parameters of collidable points.
        # Note that, contrarily to LinkParameters and JointsParameters, this object
        # is not created with vmap. This is because the "body" attribute of the object
        # must be Static for JIT-related reasons, and tree_map would not consider it
        # as a leaf.
        contact_parameters = ContactParameters.build_from(
            model_description=model_description
        )

        # =================
        # Frames properties
        # =================

        # Create the object storing the parameters of frames.
        # Note that, contrarily to LinkParameters and JointsParameters, this object
        # is not created with vmap. This is because the "name" attribute of the object
        # must be Static for JIT-related reasons, and tree_map would not consider it
        # as a leaf.
        frame_parameters = FrameParameters.build_from(
            model_description=model_description
        )

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
        parent_array = jnp.array([-1, *list(parent_array_dict.values())], dtype=int)

        # Instead of building the support parent array κ(i) for each link of the model,
        # that has a variable length depending on the number of links connecting the
        # root to the i-th link, we build the corresponding boolean version.
        # Given a link index i, the boolean support parent array κb(i) is an array
        # with the same number of elements of λ(i) having the i-th element set to True
        # if the i-th link is in the support parent array κ(i), False otherwise.
        # We store the boolean κb(i) as static attribute of the PyTree so that
        # algorithms that need to access it can be jit-compiled.
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

        # =================================
        # Build and return KynDynParameters
        # =================================

        return KynDynParameters(
            link_names=tuple(l.name for l in ordered_links),
            _parent_array=HashedNumpyArray(array=parent_array),
            _support_body_array_bool=HashedNumpyArray(array=support_body_array_bool),
            link_parameters=link_parameters,
            joint_model=joint_model,
            joint_parameters=joint_parameters,
            contact_parameters=contact_parameters,
            frame_parameters=frame_parameters,
        )

    def __eq__(self, other: KynDynParameters) -> bool:

        if not isinstance(other, KynDynParameters):
            return False

        return hash(self) == hash(other)

    def __hash__(self) -> int:

        return hash(
            (
                hash(self.number_of_links()),
                hash(self.number_of_joints()),
                hash(self.frame_parameters.name),
                hash(tuple(self.frame_parameters.body.tolist())),
                hash(self._parent_array),
                hash(self._support_body_array_bool),
            )
        )

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
        r"""
        Return the support parent array :math:`\kappa(i)` of a link.

        Args:
            link_index: The index of the link.

        Returns:
            The support parent array :math:`\kappa(i)` of the link.

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
        r"""
        Return the tree transforms of the model.

        Returns:
            The transforms
            :math:`{}^{\text{pre}(i)} H_{\lambda(i)}`
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
    def joint_transforms(
        self, joint_positions: jtp.VectorLike, base_transform: jtp.MatrixLike
    ) -> jtp.Array:
        r"""
        Return the transforms of the joints.

        Args:
            joint_positions: The joint positions.
            base_transform: The homogeneous matrix defining the base pose.

        Returns:
            The stacked transforms
            :math:`{}^{i} \mathbf{H}_{\lambda(i)}(s)`
            of each joint.
        """

        return self.joint_transforms_and_motion_subspaces(
            joint_positions=joint_positions,
            base_transform=base_transform,
        )[0]

    @jax.jit
    def joint_motion_subspaces(
        self, joint_positions: jtp.VectorLike, base_transform: jtp.MatrixLike
    ) -> jtp.Array:
        r"""
        Return the motion subspaces of the joints.

        Args:
            joint_positions: The joint positions.
            base_transform: The homogeneous matrix defining the base pose.

        Returns:
            The stacked motion subspaces :math:`\mathbf{S}(s)` of each joint.
        """

        return self.joint_transforms_and_motion_subspaces(
            joint_positions=joint_positions,
            base_transform=base_transform,
        )[1]

    @jax.jit
    def joint_transforms_and_motion_subspaces(
        self, joint_positions: jtp.VectorLike, base_transform: jtp.MatrixLike
    ) -> tuple[jtp.Array, jtp.Array]:
        r"""
        Return the transforms and the motion subspaces of the joints.

        Args:
            joint_positions: The joint positions.
            base_transform: The homogeneous matrix defining the base pose.

        Returns:
            A tuple containing the stacked transforms
            :math:`{}^{i} \mathbf{H}_{\lambda(i)}(s)`
            and the stacked motion subspaces :math:`\mathbf{S}(s)` of each joint.

        Note:
            The first transform, at index 0, provides the pose of the base link
            w.r.t. the world frame. For both floating-base and fixed-base systems,
            it takes into account the base pose and the optional transform
            between the root frame of the model and the base link.
        """

        # Rename the base transform.
        W_H_B = base_transform

        # Extract the parent-to-predecessor fixed transforms of the joints.
        λ_H_pre = jnp.vstack(
            [
                jnp.eye(4)[jnp.newaxis],
                jax.vmap(
                    lambda i: self.joint_model.parent_H_predecessor(joint_index=i)
                )(jnp.arange(1, 1 + self.number_of_joints())),
            ]
        )

        # Compute the transforms and motion subspaces of the joints.
        if self.number_of_joints() == 0:
            pre_H_suc_J, S_J = jnp.empty((0, 4, 4)), jnp.empty((0, 6, 1))
        else:
            pre_H_suc_J, S_J = jax.vmap(supported_joint_motion)(
                jnp.array(self.joint_model.joint_types[1:]).astype(int),
                jnp.array(joint_positions),
                jnp.array([j.axis for j in self.joint_model.joint_axis]),
            )

        # Extract the transforms and motion subspaces of the joints.
        # We stack the base transform W_H_B at index 0, and a dummy motion subspace
        # for either the fixed or free-floating joint connecting the world to the base.
        pre_H_suc = jnp.vstack([W_H_B[jnp.newaxis, ...], pre_H_suc_J])
        S = jnp.vstack([jnp.zeros((6, 1))[jnp.newaxis, ...], S_J])

        # Extract the successor-to-child fixed transforms.
        # Note that here we include also the index 0 since suc_H_child[0] stores the
        # optional pose of the base link w.r.t. the root frame of the model.
        # This is supported by SDF when the base link <pose> element is defined.
        suc_H_i = jax.vmap(lambda i: self.joint_model.successor_H_child(joint_index=i))(
            jnp.arange(0, 1 + self.number_of_joints())
        )

        # Compute the overall transforms from the parent to the child of each joint by
        # composing all the components of our joint model.
        i_X_λ = jax.vmap(
            lambda λ_Hi_pre, pre_Hi_suc, suc_Hi_i: Adjoint.from_transform(
                transform=λ_Hi_pre @ pre_Hi_suc @ suc_Hi_i, inverse=True
            )
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
        r"""
        Set the inertia tensor of a link.

        Args:
            link_index: The index of the link.
            inertia: The :math:`3 \times 3` inertia tensor of the link.

        Returns:
            The updated kinematic and dynamic parameters of the model.
        """

        inertia_elements = LinkParameters.flatten_inertia_tensor(I=inertia)

        link_parameters = self.link_parameters.replace(
            inertia_elements=self.link_parameters.inertia_elements.at[link_index].set(
                inertia_elements
            )
        )

        return self.replace(link_parameters=link_parameters)


@jax_dataclasses.pytree_dataclass
class JointParameters(JaxsimDataclass):
    """
    Class storing the parameters of a joint.

    Attributes:
        index: The index of the joint.
        friction_static: The static friction of the joint.
        friction_viscous: The viscous friction of the joint.
        position_limits_min: The lower position limit of the joint.
        position_limits_max: The upper position limit of the joint.
        position_limit_spring: The spring constant of the position limit.
        position_limit_damper: The damper constant of the position limit.

    Note:
        This class is used inside KinDynParameters to store the vectorized set
        of joint parameters.
    """

    index: jtp.Int

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
        """
        Build a JointParameters object from a joint description.

        Args:
            joint_description: The joint description to consider.

        Returns:
            The JointParameters object.
        """

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
            index=jnp.array(joint_description.index).squeeze().astype(int),
            friction_static=friction_static.astype(float),
            friction_viscous=friction_viscous.astype(float),
            position_limits_min=position_limits_min.astype(float),
            position_limits_max=position_limits_max.astype(float),
            position_limit_spring=position_limit_spring.astype(float),
            position_limit_damper=position_limit_damper.astype(float),
        )


@jax_dataclasses.pytree_dataclass
class LinkParameters(JaxsimDataclass):
    r"""
    Class storing the parameters of a link.

    Attributes:
        index: The index of the link.
        mass: The mass of the link.
        inertia_elements:
            The unique elements of the :math:`3 \times 3` inertia tensor of the link.
        center_of_mass:
            The translation :math:`{}^L \mathbf{p}_{\text{CoM}}` between the origin
            of the link frame and the link's center of mass, expressed in the
            coordinates of the link frame.

    Note:
        This class is used inside KinDynParameters to store the vectorized set
        of link parameters.
    """

    index: jtp.Int

    mass: jtp.Float
    center_of_mass: jtp.Vector
    inertia_elements: jtp.Vector

    @staticmethod
    def build_from_spatial_inertia(index: jtp.IntLike, M: jtp.Matrix) -> LinkParameters:
        r"""
        Build a LinkParameters object from a :math:`6 \times 6` spatial inertia matrix.

        Args:
            index: The index of the link.
            M: The :math:`6 \times 6` spatial inertia matrix of the link.

        Returns:
            The LinkParameters object.
        """

        # Extract the link parameters from the 6D spatial inertia.
        m, L_p_CoM, I = Inertia.to_params(M=M)

        # Extract only the necessary elements of the inertia tensor.
        inertia_elements = I[jnp.triu_indices(3)]

        return LinkParameters(
            index=jnp.array(index).squeeze().astype(int),
            mass=jnp.array(m).squeeze().astype(float),
            center_of_mass=jnp.atleast_1d(jnp.array(L_p_CoM).squeeze()).astype(float),
            inertia_elements=jnp.atleast_1d(inertia_elements.squeeze()).astype(float),
        )

    @staticmethod
    def build_from_inertial_parameters(
        index: jtp.IntLike, m: jtp.FloatLike, I: jtp.MatrixLike, c: jtp.VectorLike
    ) -> LinkParameters:
        r"""
        Build a LinkParameters object from the inertial parameters of a link.

        Args:
            index: The index of the link.
            m: The mass of the link.
            I: The :math:`3 \times 3` inertia tensor of the link.
            c: The translation between the link frame and the link's center of mass.

        Returns:
            The LinkParameters object.
        """

        # Extract only the necessary elements of the inertia tensor.
        inertia_elements = I[jnp.triu_indices(3)]

        return LinkParameters(
            index=jnp.array(index).squeeze().astype(int),
            mass=jnp.array(m).squeeze().astype(float),
            center_of_mass=jnp.atleast_1d(c.squeeze()).astype(float),
            inertia_elements=jnp.atleast_1d(inertia_elements.squeeze()).astype(float),
        )

    @staticmethod
    def build_from_flat_parameters(
        index: jtp.IntLike, parameters: jtp.VectorLike
    ) -> LinkParameters:

        index = jnp.array(index).squeeze().astype(int)

        m = jnp.array(parameters[0]).squeeze().astype(float)
        c = jnp.atleast_1d(parameters[1:4].squeeze()).astype(float)
        inertia_elements = jnp.atleast_1d(parameters[4:].squeeze()).astype(float)

        return LinkParameters(
            index=index, mass=m, inertia_elements=inertia_elements, center_of_mass=c
        )

    @staticmethod
    def flat_parameters(params: LinkParameters) -> jtp.Vector:
        """
        Return the parameters of a link as a flat vector.

        Args:
            params: The link parameters.

        Returns:
            The parameters of the link as a flat vector.
        """

        return (
            jnp.hstack(
                [params.mass, params.center_of_mass.squeeze(), params.inertia_elements]
            )
            .squeeze()
            .astype(float)
        )

    @staticmethod
    def inertia_tensor(params: LinkParameters) -> jtp.Matrix:
        r"""
        Return the :math:`3 \times 3` inertia tensor of a link.

        Args:
            params: The link parameters.

        Returns:
            The :math:`3 \times 3` inertia tensor of the link.
        """

        return LinkParameters.unflatten_inertia_tensor(
            inertia_elements=params.inertia_elements
        )

    @staticmethod
    def spatial_inertia(params: LinkParameters) -> jtp.Matrix:
        r"""
        Return the :math:`6 \times 6` spatial inertia matrix of a link.

        Args:
            params: The link parameters.

        Returns:
            The :math:`6 \times 6` spatial inertia matrix of the link.
        """

        return Inertia.to_sixd(
            mass=params.mass,
            I=LinkParameters.inertia_tensor(params),
            com=params.center_of_mass,
        )

    @staticmethod
    def flatten_inertia_tensor(I: jtp.Matrix) -> jtp.Vector:
        r"""
        Flatten a :math:`3 \times 3` inertia tensor into a vector of unique elements.

        Args:
            I: The :math:`3 \times 3` inertia tensor.

        Returns:
            The vector of unique elements of the inertia tensor.
        """

        return jnp.atleast_1d(I[jnp.triu_indices(3)].squeeze())

    @staticmethod
    def unflatten_inertia_tensor(inertia_elements: jtp.Vector) -> jtp.Matrix:
        r"""
        Unflatten a vector of unique elements into a :math:`3 \times 3` inertia tensor.

        Args:
            inertia_elements: The vector of unique elements of the inertia tensor.

        Returns:
            The :math:`3 \times 3` inertia tensor.
        """

        I = jnp.zeros([3, 3]).at[jnp.triu_indices(3)].set(inertia_elements.squeeze())
        return jnp.atleast_2d(jnp.where(I, I, I.T)).astype(float)


@jax_dataclasses.pytree_dataclass
class ContactParameters(JaxsimDataclass):
    """
    Class storing the contact parameters of a model.

    Attributes:
        body:
            A tuple of integers representing, for each collidable point, the index of
            the body (link) to which it is rigidly attached to.
        point:
            The translations between the link frame and the collidable point, expressed
            in the coordinates of the parent link frame.

    Note:
        Contrarily to LinkParameters and JointParameters, this class is not meant
        to be created with vmap. This is because the `body` attribute must be `Static`.
    """

    body: Static[tuple[int, ...]] = dataclasses.field(default_factory=tuple)

    point: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.array([]))

    @staticmethod
    def build_from(model_description: ModelDescription) -> ContactParameters:
        """
        Build a ContactParameters object from a model description.

        Args:
            model_description: The model description to consider.

        Returns:
            The ContactParameters object.
        """

        if len(model_description.collision_shapes) == 0:
            return ContactParameters()

        # Get all the links so that we can take their updated index.
        links_dict = {link.name: link for link in model_description}

        # Get all the enabled collidable points of the model.
        collidable_points = model_description.all_enabled_collidable_points()

        # Extract the positions L_p_C of the collidable points w.r.t. the link frames
        # they are rigidly attached to.
        points = jnp.vstack([cp.position for cp in collidable_points])

        # Extract the indices of the links to which the collidable points are rigidly
        # attached to.
        link_index_of_points = tuple(
            links_dict[cp.parent_link.name].index for cp in collidable_points
        )

        # Build the ContactParameters object.
        cp = ContactParameters(point=points, body=link_index_of_points)

        assert cp.point.shape[1] == 3, cp.point.shape[1]
        assert cp.point.shape[0] == len(cp.body), cp.point.shape[0]

        return cp


@jax_dataclasses.pytree_dataclass
class FrameParameters(JaxsimDataclass):
    """
    Class storing the frame parameters of a model.

    Attributes:
        name: A tuple of strings defining the frame names.
        body:
            A vector of integers representing, for each frame, the index of
            the body (link) to which it is rigidly attached to.
        transform: The transforms of the frames w.r.t. their parent link.

    Note:
        Contrarily to LinkParameters and JointParameters, this class is not meant
        to be created with vmap. This is because the `name` attribute must be `Static`.
    """

    name: Static[tuple[str, ...]] = dataclasses.field(default_factory=tuple)

    body: jtp.Vector = dataclasses.field(default_factory=lambda: jnp.array([]))

    transform: jtp.Array = dataclasses.field(default_factory=lambda: jnp.array([]))

    @staticmethod
    def build_from(model_description: ModelDescription) -> FrameParameters:
        """
        Build a FrameParameters object from a model description.

        Args:
            model_description: The model description to consider.

        Returns:
            The FrameParameters object.
        """

        if len(model_description.frames) == 0:
            return FrameParameters()

        # Extract the frame names.
        names = tuple(frame.name for frame in model_description.frames)

        # For each frame, extract the index of the link to which it is attached to.
        parent_link_index_of_frames = tuple(
            model_description.links_dict[frame.parent.name].index
            for frame in model_description.frames
        )

        # For each frame, extract the transform w.r.t. its parent link.
        transforms = jnp.atleast_3d(
            jnp.stack([frame.pose for frame in model_description.frames])
        )

        # Build the FrameParameters object.
        fp = FrameParameters(
            name=names,
            transform=transforms.astype(float),
            body=jnp.array(parent_link_index_of_frames).astype(int),
        )

        assert fp.transform.shape[1:] == (4, 4), fp.transform.shape[1:]
        assert fp.transform.shape[0] == len(fp.body), fp.transform.shape[0]

        return fp
