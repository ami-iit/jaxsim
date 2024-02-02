import dataclasses
import pathlib
from typing import List, Union

import idyntree.bindings as idt
import numpy as np
import numpy.typing as npt

from jaxsim.high_level.common import VelRepr


@dataclasses.dataclass
class KinDynComputations:
    """High-level wrapper of the iDynTree KinDynComputations class."""

    vel_repr: VelRepr
    gravity: npt.NDArray
    kin_dyn: idt.KinDynComputations

    @staticmethod
    def build(
        urdf: Union[pathlib.Path, str],
        considered_joints: List[str] = None,
        vel_repr: VelRepr = VelRepr.Inertial,
        gravity: npt.NDArray = dataclasses.field(
            default_factory=lambda: np.array([0, 0, -10.0])
        ),
    ) -> "KinDynComputations":
        """"""

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
            raise RuntimeError(f"Failed to load URDF description")

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
        self.world_gravity = gravity

    def dofs(self) -> int:
        return self.kin_dyn.getNrOfDegreesOfFreedom()

    def joint_names(self) -> List[str]:
        model: idt.Model = self.kin_dyn.model()
        return [model.getJointName(i) for i in range(model.getNrOfJoints())]

    def link_names(self) -> List[str]:
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

    def spatial_inertia(self, link_name: str) -> npt.NDArray:
        if link_name not in self.link_names():
            raise ValueError(link_name)

        model = self.kin_dyn.model()

        return (
            model.getLink(model.getLinkIndex(link_name)).inertia().asMatrix().toNumPy()
        )

    def floating_base_frame(self) -> str:
        return self.kin_dyn.getFloatingBase()

    def frame_transform(self, frame_name: str) -> npt.NDArray:
        if self.kin_dyn.getFrameIndex(frame_name) < 0:
            raise ValueError(f"Frame '{frame_name}' does not exist")

        if frame_name == self.floating_base_frame():
            H_idt = self.kin_dyn.getWorldBaseTransform()
        else:
            H_idt = self.kin_dyn.getWorldTransform(frame_name)

        # return H_idt.asHomogeneousTransform().toNumPy()

        H = np.eye(4)
        H[0:3, 3] = H_idt.getPosition().toNumPy()
        H[0:3, 0:3] = H_idt.getRotation().toNumPy()

        return H

    def base_velocity(self) -> npt.NDArray:
        nu = idt.VectorDynSize()

        if not self.kin_dyn.getModelVel(nu):
            raise RuntimeError("Failed to get the model velocity")

        return nu.toNumPy()[0:6]

    def com_position(self) -> npt.NDArray:
        W_p_G = self.kin_dyn.getCenterOfMassPosition()
        return W_p_G.toNumPy()

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
