from __future__ import annotations

import dataclasses
import pathlib

import idyntree.bindings as idt
import numpy as np
import numpy.typing as npt

import jaxsim.api as js
from jaxsim import VelRepr


def build_kindyncomputations_from_jaxsim_model(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> KinDynComputations:
    """
    Build a `KinDynComputations` from `JaxSimModel` and `JaxSimModelData`.

    Args:
        model: The `JaxSimModel` from which to build the `KinDynComputations`.
        data: The `JaxSimModelData` from which to build the `KinDynComputations`.

    Returns:
        The `KinDynComputations` built from the `JaxSimModel` and `JaxSimModelData`.

    Note:
        Only `JaxSimModel` built from URDF files are supported.
    """

    if (
        isinstance(model.built_from, pathlib.Path)
        and model.built_from.suffix != ".urdf"
    ) or (isinstance(model.built_from, str) and "<robot" not in model.built_from):
        raise ValueError("iDynTree only supports URDF models")

    # Create the KinDynComputations from the same URDF model.
    kin_dyn = KinDynComputations.build(
        urdf=model.built_from,
        considered_joints=list(model.joint_names()),
        vel_repr=data.velocity_representation,
        gravity=np.array(data.gravity),
    )

    # Copy the state of the JaxSim model.
    kin_dyn = store_jaxsim_data_in_kindyncomputations(data=data, kin_dyn=kin_dyn)

    return kin_dyn


def store_jaxsim_data_in_kindyncomputations(
    data: js.data.JaxSimModelData, kin_dyn: KinDynComputations
) -> KinDynComputations:
    """
    Store the state of a `JaxSimModelData` in `KinDynComputations`.

    Args:
        data:
            The `JaxSimModelData` providing the desired state to copy.
        kin_dyn:
            The `KinDynComputations` in which to store the state of `JaxSimModelData`.

    Returns:
        The updated `KinDynComputations` with the state of `JaxSimModelData`.
    """

    with data.switch_velocity_representation(kin_dyn.vel_repr):
        kin_dyn.set_robot_state(
            joint_positions=np.array(data.joint_positions()),
            joint_velocities=np.array(data.joint_velocities()),
            base_transform=np.array(data.base_transform()),
            base_velocity=np.array(data.base_velocity()),
        )

    return kin_dyn


@dataclasses.dataclass
class KinDynComputations:
    """High-level wrapper of the iDynTree KinDynComputations class."""

    vel_repr: VelRepr
    gravity: npt.NDArray
    kin_dyn: idt.KinDynComputations

    @staticmethod
    def build(
        urdf: pathlib.Path | str,
        considered_joints: list[str] = None,
        vel_repr: VelRepr = VelRepr.Inertial,
        gravity: npt.NDArray = dataclasses.field(
            default_factory=lambda: np.array([0, 0, -10.0])
        ),
    ) -> "KinDynComputations":

        # Read the URDF description
        urdf_string = urdf.read_text() if isinstance(urdf, pathlib.Path) else urdf

        # Create the model loader
        mdl_loader = idt.ModelLoader()

        # Load the URDF description
        if not (
            mdl_loader.loadModelFromString(urdf_string)
            if considered_joints is None
            else mdl_loader.loadReducedModelFromString(urdf_string, considered_joints)
        ):
            raise RuntimeError("Failed to load URDF description")

        # Create KinDynComputations and insert the model
        kindyn = idt.KinDynComputations()

        if not kindyn.loadRobotModel(mdl_loader.model()):
            raise RuntimeError("Failed to load model")

        vel_repr_to_idyntree = {
            VelRepr.Inertial: idt.INERTIAL_FIXED_REPRESENTATION,
            VelRepr.Body: idt.BODY_FIXED_REPRESENTATION,
            VelRepr.Mixed: idt.MIXED_REPRESENTATION,
        }

        # Configure the frame representation
        if not kindyn.setFrameVelocityRepresentation(vel_repr_to_idyntree[vel_repr]):
            raise RuntimeError("Failed to set the frame representation")

        return KinDynComputations(
            kin_dyn=kindyn,
            vel_repr=vel_repr,
            gravity=np.array(gravity).squeeze(),
        )

    def set_robot_state(
        self,
        joint_positions: npt.NDArray | None = None,
        joint_velocities: npt.NDArray | None = None,
        base_transform: npt.NDArray = np.eye(4),
        base_velocity: npt.NDArray = np.zeros(6),
        world_gravity: npt.NDArray | None = None,
    ) -> None:

        joint_positions = (
            joint_positions if joint_positions is not None else np.zeros(self.dofs())
        )

        joint_velocities = (
            joint_velocities if joint_velocities is not None else np.zeros(self.dofs())
        )

        gravity = world_gravity if world_gravity is not None else self.gravity

        if joint_positions.size != self.dofs():
            raise ValueError(joint_positions.size, self.dofs())

        if joint_velocities.size != self.dofs():
            raise ValueError(joint_velocities.size, self.dofs())

        if gravity.size != 3:
            raise ValueError(gravity.size, 3)

        if base_transform.shape != (4, 4):
            raise ValueError(base_transform.shape, (4, 4))

        if base_velocity.size != 6:
            raise ValueError(base_velocity.size)

        g = idt.Vector3().FromPython(np.array(gravity))
        s = idt.VectorDynSize().FromPython(np.array(joint_positions))
        s_dot = idt.VectorDynSize().FromPython(np.array(joint_velocities))

        p = idt.Position(*[float(i) for i in np.array(base_transform[0:3, 3])])
        R = idt.Rotation()
        R = R.FromPython(np.array(base_transform[0:3, 0:3]))
        world_H_base = idt.Transform()
        world_H_base.setPosition(p)
        world_H_base.setRotation(R)

        v_WB = idt.Twist().FromPython(base_velocity)

        if not self.kin_dyn.setRobotState(world_H_base, s, v_WB, s_dot, g):
            raise RuntimeError("Failed to set the robot state")

        # Update stored gravity
        self.gravity = gravity

    def dofs(self) -> int:

        return self.kin_dyn.getNrOfDegreesOfFreedom()

    def joint_names(self) -> list[str]:

        model: idt.Model = self.kin_dyn.model()
        return [model.getJointName(i) for i in range(model.getNrOfJoints())]

    def link_names(self) -> list[str]:

        return [
            self.kin_dyn.getFrameName(i) for i in range(self.kin_dyn.getNrOfLinks())
        ]

    def joint_positions(self) -> npt.NDArray:

        vector = idt.VectorDynSize()

        if not self.kin_dyn.getJointPos(vector):
            raise RuntimeError("Failed to extract joint positions")

        return vector.toNumPy()

    def joint_velocities(self) -> npt.NDArray:

        vector = idt.VectorDynSize()

        if not self.kin_dyn.getJointVel(vector):
            raise RuntimeError("Failed to extract joint velocities")

        return vector.toNumPy()

    def jacobian_frame(self, frame_name: str) -> npt.NDArray:

        if self.kin_dyn.getFrameIndex(frame_name) < 0:
            raise ValueError(f"Frame '{frame_name}' does not exist")

        J = idt.MatrixDynSize(6, 6 + self.dofs())

        if not self.kin_dyn.getFrameFreeFloatingJacobian(frame_name, J):
            raise RuntimeError("Failed to get the frame free-floating jacobian")

        return J.toNumPy()

    def total_mass(self) -> float:

        model: idt.Model = self.kin_dyn.model()
        return model.getTotalMass()

    def link_spatial_inertia(self, link_name: str) -> npt.NDArray:

        if link_name not in self.link_names():
            raise ValueError(link_name)

        model = self.kin_dyn.model()
        link: idt.Link = model.getLink(model.getLinkIndex(link_name))

        return link.inertia().asMatrix().toNumPy()

    def link_mass(self, link_name: str) -> float:

        if link_name not in self.link_names():
            raise ValueError(link_name)

        model = self.kin_dyn.model()
        link: idt.Link = model.getLink(model.getLinkIndex(link_name))

        return link.getInertia().asVector().toNumPy()[0]

    def floating_base_frame(self) -> str:

        return self.kin_dyn.getFloatingBase()

    def frame_transform(self, frame_name: str) -> npt.NDArray:

        if self.kin_dyn.getFrameIndex(frame_name) < 0:
            raise ValueError(f"Frame '{frame_name}' does not exist")

        if frame_name == self.floating_base_frame():
            H_idt = self.kin_dyn.getWorldBaseTransform()
        else:
            H_idt = self.kin_dyn.getWorldTransform(frame_name)

        H = np.eye(4)
        H[0:3, 3] = H_idt.getPosition().toNumPy()
        H[0:3, 0:3] = H_idt.getRotation().toNumPy()

        return H

    def base_velocity(self) -> npt.NDArray:

        nu = idt.VectorDynSize()

        if not self.kin_dyn.getModelVel(nu):
            raise RuntimeError("Failed to get the model velocity")

        return nu.toNumPy()[0:6]

    def frame_velocity(self, frame_name: str) -> npt.NDArray:

        if self.kin_dyn.getFrameIndex(frame_name) < 0:
            raise ValueError(f"Frame '{frame_name}' does not exist")

        v_WF = self.kin_dyn.getFrameVel(frame_name)

        return v_WF.toNumPy()

    def frame_bias_acc(self, frame_name: str) -> npt.NDArray:

        if self.kin_dyn.getFrameIndex(frame_name) < 0:
            raise ValueError(f"Frame '{frame_name}' does not exist")

        J̇ν = self.kin_dyn.getFrameBiasAcc(frame_name)

        return J̇ν.toNumPy()

    def com_position(self) -> npt.NDArray:

        W_p_G = self.kin_dyn.getCenterOfMassPosition()
        return W_p_G.toNumPy()

    def com_velocity(self) -> npt.NDArray:

        W_ṗ_G = self.kin_dyn.getCenterOfMassVelocity()
        return W_ṗ_G.toNumPy()

    def mass_matrix(self) -> npt.NDArray:

        M = idt.MatrixDynSize()

        if not self.kin_dyn.getFreeFloatingMassMatrix(M):
            raise RuntimeError("Failed to get the free floating mass matrix")

        return M.toNumPy()

    def bias_forces(self) -> npt.NDArray:

        h = idt.FreeFloatingGeneralizedTorques(self.kin_dyn.model())

        if not self.kin_dyn.generalizedBiasForces(h):
            raise RuntimeError("Failed to get the generalized bias forces")

        base_wrench: idt.Wrench = h.baseWrench()
        joint_torques: idt.JointDOFsDoubleArray = h.jointTorques()

        return np.hstack(
            [base_wrench.toNumPy().flatten(), joint_torques.toNumPy().flatten()]
        )

    def gravity_forces(self) -> npt.NDArray:

        g = idt.FreeFloatingGeneralizedTorques(self.kin_dyn.model())

        if not self.kin_dyn.generalizedGravityForces(g):
            raise RuntimeError("Failed to get the generalized gravity forces")

        base_wrench: idt.Wrench = g.baseWrench()
        joint_torques: idt.JointDOFsDoubleArray = g.jointTorques()

        return np.hstack(
            [base_wrench.toNumPy().flatten(), joint_torques.toNumPy().flatten()]
        )

    def total_momentum(self) -> npt.NDArray:

        return self.kin_dyn.getLinearAngularMomentum().toNumPy().flatten()

    def centroidal_momentum(self) -> npt.NDArray:

        return self.kin_dyn.getCentroidalTotalMomentum().toNumPy().flatten()

    def total_momentum_jacobian(self) -> npt.NDArray:

        Jh = idt.MatrixDynSize()

        if not self.kin_dyn.getLinearAngularMomentumJacobian(Jh):
            raise RuntimeError("Failed to get the total momentum jacobian")

        return Jh.toNumPy()

    def centroidal_momentum_jacobian(self) -> npt.NDArray:

        Jh = idt.MatrixDynSize()

        if not self.kin_dyn.getCentroidalTotalMomentumJacobian(Jh):
            raise RuntimeError("Failed to get the centroidal momentum jacobian")

        return Jh.toNumPy()

    def locked_spatial_inertia(self) -> npt.NDArray:

        return self.kin_dyn.getRobotLockedInertia().asMatrix().toNumPy()

    def locked_centroidal_spatial_inertia(self) -> npt.NDArray:

        return self.kin_dyn.getCentroidalRobotLockedInertia().asMatrix().toNumPy()

    def average_velocity(self) -> npt.NDArray:

        return self.kin_dyn.getAverageVelocity().toNumPy()

    def average_velocity_jacobian(self) -> npt.NDArray:

        Jh = idt.MatrixDynSize()

        if not self.kin_dyn.getAverageVelocityJacobian(Jh):
            raise RuntimeError("Failed to get the average velocity jacobian")

        return Jh.toNumPy()

    def average_centroidal_velocity(self) -> npt.NDArray:

        return self.kin_dyn.getCentroidalAverageVelocity().toNumPy()

    def average_centroidal_velocity_jacobian(self) -> npt.NDArray:

        Jh = idt.MatrixDynSize()

        if not self.kin_dyn.getCentroidalAverageVelocityJacobian(Jh):
            raise RuntimeError("Failed to get the average centroidal velocity jacobian")

        return Jh.toNumPy()
