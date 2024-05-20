from __future__ import annotations

import copy
import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from jaxsim import logging
from jaxsim.utils import Mutability

from . import descriptions


class RootPose(NamedTuple):
    """
    Represents the root pose in a kinematic graph.

    Attributes:
        root_position (npt.NDArray): A NumPy array of shape (3,) representing the root's position.
        root_quaternion (npt.NDArray): A NumPy array of shape (4,) representing the root's quaternion.
    """

    root_position: npt.NDArray = np.zeros(3)
    root_quaternion: npt.NDArray = np.array([1.0, 0, 0, 0])

    def __eq__(self, other):
        return (self.root_position == other.root_position).all() and (
            self.root_quaternion == other.root_quaternion
        ).all()


@dataclasses.dataclass(frozen=True)
class KinematicGraph(Sequence[descriptions.LinkDescription]):
    """
    Represents a kinematic graph of links and joints.

    Args:
        root (descriptions.LinkDescription): The root link of the kinematic graph.
        frames (List[descriptions.LinkDescription]): A list of frame links in the graph.
        joints (List[descriptions.JointDescription]): A list of joint descriptions in the graph.
        root_pose (RootPose): The root pose of the graph.
        transform_cache (Dict[str, npt.NDArray]): A dictionary to cache transformation matrices.
        extra_info (Dict[str, Any]): Additional information associated with the graph.

    Attributes:
        links_dict (Dict[str, descriptions.LinkDescription]): A dictionary mapping link names to link descriptions.
        frames_dict (Dict[str, descriptions.LinkDescription]): A dictionary mapping frame names to frame link descriptions.
        joints_dict (Dict[str, descriptions.JointDescription]): A dictionary mapping joint names to joint descriptions.
        joints_connection_dict (Dict[Tuple[str, str], descriptions.JointDescription]): A dictionary mapping pairs of parent and child link names to joint descriptions.
    """

    root: descriptions.LinkDescription
    frames: List[descriptions.LinkDescription] = dataclasses.field(default_factory=list)
    joints: List[descriptions.JointDescription] = dataclasses.field(
        default_factory=list
    )

    root_pose: RootPose = dataclasses.field(default_factory=RootPose)

    transform_cache: Dict[str, npt.NDArray] = dataclasses.field(
        repr=False, init=False, compare=False, default_factory=dict
    )

    extra_info: Dict[str, Any] = dataclasses.field(
        repr=False, compare=False, default_factory=dict
    )

    # Private attribute storing the unconnected joints from the parsed model and
    # the joints removed after model reduction.
    _joints_removed: list[descriptions.JointDescription] = dataclasses.field(
        default_factory=list, repr=False, compare=False
    )

    @functools.cached_property
    def links_dict(self) -> Dict[str, descriptions.LinkDescription]:
        return {l.name: l for l in iter(self)}

    @functools.cached_property
    def frames_dict(self) -> Dict[str, descriptions.LinkDescription]:
        return {f.name: f for f in self.frames}

    @functools.cached_property
    def joints_dict(self) -> Dict[str, descriptions.JointDescription]:
        return {j.name: j for j in self.joints}

    @functools.cached_property
    def joints_connection_dict(
        self,
    ) -> Dict[Tuple[str, str], descriptions.JointDescription]:
        return {(j.parent.name, j.child.name): j for j in self.joints}

    def __post_init__(self):
        """
        Post-initialization method to set various properties and validate the kinematic graph.
        """
        # Assign the link index traversing the graph with BFS.
        # Here we assume the model is fixed-base, therefore the base link will
        # have index 0. We will deal with the floating base in a later stage,
        # when this Model object is converted to the physics model.
        for index, link in enumerate(self):
            link.mutable(validate=False).index = index

        # Order frames with their name
        super().__setattr__("frames", sorted(self.frames, key=lambda f: f.name))

        # Assign the frame index following the name-based indexing.
        # Also here, we assume the model is fixed-base, therefore the first frame will
        # have last_link_idx + 1. These frames are not part of the physics model.
        for index, frame in enumerate(self.frames):
            with frame.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                frame.index = int(index + len(self.link_names()))

        # Number joints so that their index matches their child link index
        links_dict = {l.name: l for l in iter(self)}
        for joint in self.joints:
            with joint.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                joint.index = links_dict[joint.child.name].index

        # Check that joint indices are unique
        assert len([j.index for j in self.joints]) == len(
            {j.index for j in self.joints}
        )

        # Order joints with their indices
        super().__setattr__("joints", sorted(self.joints, key=lambda j: j.index))

    @staticmethod
    def build_from(
        links: List[descriptions.LinkDescription],
        joints: List[descriptions.JointDescription],
        root_link_name: str | None = None,
        root_pose: RootPose = RootPose(),
    ) -> "KinematicGraph":
        """
        Build a KinematicGraph from a list of links and joints.

        Args:
            links (List[descriptions.LinkDescription]): A list of link descriptions.
            joints (List[descriptions.JointDescription]): A list of joint descriptions.
            root_link_name (str, optional): The name of the root link. If not provided, it's assumed to be the first link's name.
            root_pose (RootPose, optional): The root pose of the kinematic graph.

        Returns:
            KinematicGraph: The constructed kinematic graph.
        """
        if root_link_name is None:
            root_link_name = links[0].name

        # Couple links and joints and create the graph of links.
        # Note that the pose of the frames is not updated; it's the caller's
        # responsibility to update their pose if they want to use them.
        graph_root_node, graph_joints, graph_frames, unconnected_joints = (
            KinematicGraph.create_graph(
                links=links, joints=joints, root_link_name=root_link_name
            )
        )

        for frame in graph_frames:
            logging.warning(msg=f"Ignoring unconnected link / frame: '{frame.name}'")

        for joint in unconnected_joints:
            logging.warning(msg=f"Ignoring unconnected joint: '{joint.name}'")

        return KinematicGraph(
            root=graph_root_node,
            joints=graph_joints,
            frames=[],
            root_pose=root_pose,
            _joints_removed=unconnected_joints,
        )

    @staticmethod
    def create_graph(
        links: List[descriptions.LinkDescription],
        joints: List[descriptions.JointDescription],
        root_link_name: str,
    ) -> Tuple[
        descriptions.LinkDescription,
        List[descriptions.JointDescription],
        List[descriptions.LinkDescription],
        list[descriptions.JointDescription],
    ]:
        """
        Create a kinematic graph from the lists of parsed links and joints.

        Args:
            links (List[descriptions.LinkDescription]): A list of link descriptions.
            joints (List[descriptions.JointDescription]): A list of joint descriptions.
            root_link_name (str): The name of the root link.

        Returns:
            A tuple containing the root node with the full kinematic graph as child nodes,
            the list of joints associated to graph nodes, the list of frames rigidly
            attached to graph nodes, and the list of joints not part of the graph.
        """

        # Create a dict that maps link name to the link, for easy retrieval
        links_dict: Dict[str, descriptions.LinkDescription] = {
            l.name: l.mutable(validate=False) for l in links
        }

        if root_link_name not in links_dict:
            raise ValueError(root_link_name)

        # Reset the connections of the root link
        for link in links_dict.values():
            link.children = []

        # Couple links and joints creating the final kinematic graph
        for joint in joints:
            # Get the parent and child links of the joint
            parent_link = links_dict[joint.parent.name]
            child_link = links_dict[joint.child.name]

            assert child_link.name == joint.child.name
            assert parent_link.name == joint.parent.name

            # Assign link parent
            child_link.parent = parent_link

            # Assign link children and make sure they are unique
            if child_link.name not in {l.name for l in parent_link.children}:
                parent_link.children.append(child_link)

        # Collect all the links of the kinematic graph
        all_links_in_graph = list(
            KinematicGraph.breadth_first_search(root=links_dict[root_link_name])
        )
        all_link_names_in_graph = [l.name for l in all_links_in_graph]

        # Collect all the joints not part of the kinematic graph
        removed_joints = [
            j
            for j in joints
            if not {j.parent.name, j.child.name}.issubset(all_link_names_in_graph)
        ]

        for removed_joint in removed_joints:
            msg = "Joint '{}' has been removed for the graph because unconnected"
            logging.info(msg=msg.format(removed_joint.name))

        # Store as frames all the links that are not part of the kinematic graph
        frames = list(set(links) - set(all_links_in_graph))

        # Update the frames. In particular, reset their children. The other properties
        # are kept as they are, and it's caller responsibility to update them if needed.
        for frame in frames:
            frame.children = []
            msg = f"Link '{frame.name}' became a frame"
            logging.info(msg=msg)

        return (
            links_dict[root_link_name].mutable(mutable=False),
            list(set(joints) - set(removed_joints)),
            frames,
            list(set(removed_joints)),
        )

    def reduce(self, considered_joints: List[str]) -> KinematicGraph:
        """
        Reduce the kinematic graph by removing unspecified joints.

        When a joint is removed, the mass and inertia of its child link are lumped
        with those of its parent link, obtaining a new link that combines the two.
        The description of the removed joint specifies the default angle (usually 0)
        that is considered when the joint is removed.

        Args:
            considered_joints: A list of joint names to consider.

        Returns:
            The reduced kinematic graph.
        """

        # The current object represents the complete kinematic graph
        full_graph = self

        # Get the names of the joints to remove
        joint_names_to_remove = list(
            set(full_graph.joint_names()) - set(considered_joints)
        )

        # Return early if there is no action to take
        if len(joint_names_to_remove) == 0:
            logging.info("The kinematic graph doesn't need to be reduced")
            return copy.deepcopy(self)

        # Check if all considered joints are part of the full kinematic graph
        if len(set(considered_joints) - set(j.name for j in full_graph.joints)) != 0:
            extra_j = set(considered_joints) - {j.name for j in full_graph.joints}
            msg = f"Not all joints to consider are part of the graph ({{{extra_j}}})"
            raise ValueError(msg)

        # Extract data we need to modify from the full graph
        links_dict = copy.deepcopy(full_graph.links_dict)
        joints_dict = copy.deepcopy(full_graph.joints_dict)

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=full_graph)

        # The following steps are implemented below in order to create the reduced graph:
        #
        # 1. Lump the mass of the removed links into their parent
        # 2. Update the pose and parent link of joints having the removed link as parent
        # 3. Create the reduced graph considering the removed links as frames
        # 4. Resolve the pose of the frames wrt their reduced graph parent
        #
        # We name "removed link" the link to remove, and "lumped link" the new link that
        # combines the removed link and its parent. The lumped link will share the frame
        # of the removed link's parent and the inertial properties of the two links that
        # have been combined.

        # =======================================================
        # 1. Lump the mass of the removed links into their parent
        # =======================================================

        # Get all the links to remove. They will be lumped with their parent.
        links_to_remove = [
            joint.child.name
            for joint_name, joint in joints_dict.items()
            if joint_name in joint_names_to_remove
        ]

        # Lump the mass and the inertia traversing the tree from the leaf to the root,
        # this way we propagate these properties back even in the case when also the
        # parent link of a removed joint has to be lumped with its parent.
        for link in reversed(full_graph):
            if link.name not in links_to_remove:
                continue

            # Get the link to remove and its parent, i.e. the lumped link
            link_to_remove = links_dict[link.name]
            parent_of_link_to_remove = links_dict[link.parent.name]

            msg = "Lumping chain: {}->({})->{}"
            logging.info(
                msg.format(
                    link_to_remove.name,
                    self.joints_connection_dict[
                        (parent_of_link_to_remove.name, link_to_remove.name)
                    ].name,
                    parent_of_link_to_remove.name,
                )
            )

            # Lump the link
            lumped_link = parent_of_link_to_remove.lump_with(
                link=link_to_remove,
                lumped_H_removed=fk.relative_transform(
                    relative_to=parent_of_link_to_remove.name, name=link_to_remove.name
                ),
            )

            # Pop the original two links from the dictionary...
            links_dict.pop(link_to_remove.name)
            links_dict.pop(parent_of_link_to_remove.name)

            # ... and insert the lumped link (having the same name of the parent)
            links_dict[lumped_link.name] = lumped_link

            # Insert back in the dict an entry from the removed link name to the new
            # lumped link. We need this info later, when we process the remaining joints.
            links_dict[link_to_remove.name] = lumped_link

            # As a consequence of the back-insertion, we need to adjust the resulting
            # lumped link of links that have been removed previously
            for previously_removed_link_name in {
                k
                for k, v in links_dict.items()
                if k != v.name and v.name == link_to_remove.name
            }:
                links_dict[previously_removed_link_name] = lumped_link

        # ==============================================================================
        # 2. Update the pose and parent link of joints having the removed link as parent
        # ==============================================================================

        # Find the joints having the removed links as parent
        joints_with_removed_parent_link = [
            joints_dict[joint_name]
            for joint_name in considered_joints
            if joints_dict[joint_name].parent.name in links_to_remove
        ]

        # Update the pose of all joints having as parent link a removed link
        for joint in joints_with_removed_parent_link:
            # Update the pose. Note that after the lumping process, the dict entry
            # links_dict[joint.parent.name] contains the final lumped link
            with joint.mutable_context(mutability=Mutability.MUTABLE):
                joint.pose = fk.relative_transform(
                    relative_to=links_dict[joint.parent.name].name, name=joint.name
                )
            with joint.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                # Update the parent link
                joint.parent = links_dict[joint.parent.name]

        # ===================================================================
        # 3. Create the reduced graph considering the removed links as frames
        # ===================================================================

        # Get all the original links from the full graph
        full_graph_links_dict = copy.deepcopy(full_graph.links_dict)

        # Get all the final links from the reduced graph
        links_to_keep = [
            l for link_name, l in links_dict.items() if link_name not in links_to_remove
        ]

        # Override the entries of the full graph with those of the reduced graph.
        # Those that are not overridden will become frames.
        for link in links_to_keep:
            full_graph_links_dict[link.name] = link

        # Create the reduced graph data. We pass the full list of links so that those
        # that are not part of the graph will be returned as frames.
        reduced_root_node, reduced_joints, reduced_frames, unconnected_joints = (
            KinematicGraph.create_graph(
                links=list(full_graph_links_dict.values()),
                joints=[joints_dict[joint_name] for joint_name in considered_joints],
                root_link_name=full_graph.root.name,
            )
        )

        # Create the reduced graph
        reduced_graph = KinematicGraph(
            root=reduced_root_node,
            joints=reduced_joints,
            frames=self.frames + reduced_frames,
            root_pose=full_graph.root_pose,
            _joints_removed=(
                self._joints_removed
                + unconnected_joints
                + [joints_dict[name] for name in joint_names_to_remove]
            ),
        )

        # ================================================================
        # 4. Resolve the pose of the frames wrt their reduced graph parent
        # ================================================================

        # Update frames properties using the transforms from the full graph
        for frame in reduced_graph.frames:
            # Get the link in which the removed link was lumped into
            new_parent_link = links_dict[frame.name]

            msg = f"New parent of frame '{frame.name}' is '{new_parent_link.name}'"
            logging.info(msg)

            # Update the connection of the frame
            frame.parent = new_parent_link
            frame.pose = fk.relative_transform(
                relative_to=new_parent_link.name, name=frame.name
            )

            # Update frame data
            frame.mass = 0.0
            frame.inertia = np.zeros_like(frame.inertia)

        # Return the reduced graph
        return reduced_graph

    def link_names(self) -> List[str]:
        """
        Get the names of all links in the kinematic graph.

        Returns:
            List[str]: A list of link names.
        """
        return list(self.links_dict.keys())

    def joint_names(self) -> List[str]:
        """
        Get the names of all joints in the kinematic graph.

        Returns:
            List[str]: A list of joint names.
        """
        return list(self.joints_dict.keys())

    def frame_names(self) -> List[str]:
        """
        Get the names of all frames in the kinematic graph.

        Returns:
            List[str]: A list of frame names.
        """
        return list(self.frames_dict.keys())

    def print_tree(self) -> None:
        """
        Print the tree structure of the kinematic graph.
        """
        import pptree

        root_node = self.root

        pptree.print_tree(
            root_node,
            childattr="children",
            nameattr="name_and_index",
            horizontal=True,
        )

    @staticmethod
    def breadth_first_search(
        root: descriptions.LinkDescription,
        sort_children: Optional[Callable[[Any], Any]] = lambda link: link.name,
    ) -> Iterable[descriptions.LinkDescription]:
        """
        Perform a breadth-first search (BFS) traversal of the kinematic graph.

        Args:
            root (descriptions.LinkDescription): The root link for BFS.
            sort_children (Optional[Callable[[Any], Any]]): A function to sort children of a node.

        Yields:
            Iterable[descriptions.LinkDescription]: An iterable of link descriptions.
        """
        queue = [root]

        # We assume that nodes have unique names, and mark a link as visited using
        # its name. This speeds up considerably object comparison.
        visited = []
        visited.append(root.name)

        yield root

        while len(queue) > 0:
            l = queue.pop(0)

            # Note: sorting the links with their name so that the order of children
            # insertion does not matter when assigning the link index
            for child in sorted(l.children, key=sort_children):
                if child.name in visited:
                    continue

                visited.append(child.name)
                queue.append(child)

                yield child

    # =================
    # Sequence protocol
    # =================

    def __iter__(self) -> Iterable[descriptions.LinkDescription]:
        yield from KinematicGraph.breadth_first_search(root=self.root)

    def __reversed__(self) -> Iterable[descriptions.LinkDescription]:
        yield from reversed(list(iter(self)))

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __contains__(self, item: Union[str, descriptions.LinkDescription]) -> bool:
        if isinstance(item, str):
            return item in self.link_names()

        if isinstance(item, descriptions.LinkDescription):
            return item in set(iter(self))

        raise TypeError(type(item).__name__)

    def __getitem__(self, key: Union[int, str]) -> descriptions.LinkDescription:
        if isinstance(key, str):
            if key not in self.link_names():
                raise KeyError(key)

            return self.links_dict[key]

        if isinstance(key, int):
            if key > len(self):
                raise KeyError(key)

            return list(iter(self))[key]

        raise TypeError(type(key).__name__)

    def count(self, value: descriptions.LinkDescription) -> int:
        return list(iter(self)).count(value)

    def index(
        self, value: descriptions.LinkDescription, start: int = 0, stop: int = -1
    ) -> int:
        return list(iter(self)).index(value, start, stop)


# ====================
# Other useful classes
# ====================


@dataclasses.dataclass(frozen=True)
class KinematicGraphTransforms:

    graph: KinematicGraph

    _transform_cache: dict[str, npt.NDArray] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    _initial_joint_positions: dict[str, float] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:

        super().__setattr__(
            "_initial_joint_positions",
            {joint.name: joint.initial_position for joint in self.graph.joints},
        )

    @property
    def initial_joint_positions(self) -> npt.NDArray:

        return np.atleast_1d(
            np.array(list(self._initial_joint_positions.values()))
        ).astype(float)

    @initial_joint_positions.setter
    def initial_joint_positions(
        self,
        positions: npt.NDArray | Sequence,
        joint_names: Sequence[str] | None = None,
    ) -> None:

        joint_names = (
            joint_names
            if joint_names is not None
            else list(self._initial_joint_positions.keys())
        )

        s = np.atleast_1d(np.array(positions).squeeze())

        if s.size != len(joint_names):
            raise ValueError(s.size, len(joint_names))

        for joint_name in joint_names:
            if joint_name not in self._initial_joint_positions:
                raise ValueError(joint_name)

        # Clear transform cache.
        self._transform_cache.clear()

        # Update initial joint positions.
        for joint_name, position in zip(joint_names, s):
            self._initial_joint_positions[joint_name] = position

    def transform(self, name: str) -> npt.NDArray:
        """
        Compute the SE(3) transform of elements belonging to the kinematic graph.

        Args:
            name: The name of a link, a joint, or a frame.

        Returns:
            The 4x4 transform matrix of the element w.r.t. the model frame.
        """

        # If the transform was already computed, return it.
        if name in self._transform_cache:
            return self._transform_cache[name]

        # If the name is a joint, compute M_H_J transform.
        if name in self.graph.joint_names():

            # Get the joint.
            joint = self.graph.joints_dict[name]

            # Get the transform of the parent link.
            M_H_L = self.transform(name=joint.parent.name)

            # Rename the pose of the predecessor joint frame w.r.t. its parent link.
            L_H_pre = joint.pose

            # Compute the joint transform from the predecessor to the successor frame.
            pre_H_J = self.pre_H_suc(
                joint_type=joint.jtype,
                joint_axis=joint.axis,
                joint_position=self._initial_joint_positions[joint.name],
            )

            # Compute the M_H_J transform.
            self._transform_cache[name] = M_H_L @ L_H_pre @ pre_H_J
            return self._transform_cache[name]

        # If the name is a link, compute M_H_L transform.
        if name in self.graph.link_names():

            # Get the link.
            link = self.graph.links_dict[name]

            # Handle the pose between the __model__ frame and the root link.
            if link.name == self.graph.root.name:
                M_H_B = link.pose
                return M_H_B

            # Get the joint between the link and its parent.
            parent_joint = self.graph.joints_connection_dict[
                (link.parent.name, link.name)
            ]

            # Get the transform of the parent joint.
            M_H_J = self.transform(name=parent_joint.name)

            # Rename the pose of the link w.r.t. its parent joint.
            J_H_L = link.pose

            # Compute the M_H_L transform.
            self._transform_cache[name] = M_H_J @ J_H_L
            return self._transform_cache[name]

        # It can only be a plain frame.
        if name not in self.graph.frame_names():
            raise ValueError(name)

        # Get the frame.
        frame = self.graph.frames_dict[name]

        # Get the transform of the parent link.
        M_H_L = self.transform(name=frame.parent.name)

        # Rename the pose of the frame w.r.t. its parent link.
        L_H_F = frame.pose

        # Compute the M_H_F transform.
        self._transform_cache[name] = M_H_L @ L_H_F
        return self._transform_cache[name]

    def relative_transform(self, relative_to: str, name: str) -> npt.NDArray:
        """
        Compute the SE(3) relative transform of elements belonging to the kinematic graph.

        Args:
            relative_to: The name of the reference element.
            name: The name of a link, a joint, or a frame.

        Returns:
            The 4x4 transform matrix of the element w.r.t. the desired frame.
        """

        import jaxsim.math

        M_H_target = self.transform(name=name)
        M_H_R = self.transform(name=relative_to)

        # Compute the relative transform R_H_target, where R is the reference frame,
        # and i the frame of the desired link|joint|frame.
        return np.array(jaxsim.math.Transform.inverse(M_H_R)) @ M_H_target

    @staticmethod
    def pre_H_suc(
        joint_type: descriptions.JointType,
        joint_axis: descriptions.JointGenericAxis,
        joint_position: float | None = None,
    ) -> npt.NDArray:

        import jaxsim.math

        return np.array(
            jaxsim.math.supported_joint_motion(joint_type, joint_position, joint_axis)[
                0
            ]
        )
