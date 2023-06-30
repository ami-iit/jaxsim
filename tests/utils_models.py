import enum
import pathlib

import robot_descriptions.anymal_c_description
import robot_descriptions.cassie_description
import robot_descriptions.double_pendulum_description
import robot_descriptions.icub_description
import robot_descriptions.laikago_description
import robot_descriptions.panda_description
import robot_descriptions.ur10_description


class Robot(enum.IntEnum):
    iCub = enum.auto()
    Ur10 = enum.auto()
    Panda = enum.auto()
    Cassie = enum.auto()
    AnymalC = enum.auto()
    Laikago = enum.auto()
    DoublePendulum = enum.auto()


class ModelFactory:
    """Factory class providing URDF files used by the tests."""

    @staticmethod
    def get_model_description(robot: Robot) -> pathlib.Path:
        """
        Get the URDF file of different robots.

        Args:
            robot: Robot name of the desired URDF file.

        Returns:
            Path to the URDF file of the robot.
        """

        match robot:
            case Robot.iCub:
                return pathlib.Path(robot_descriptions.icub_description.URDF_PATH)
            case Robot.Ur10:
                return pathlib.Path(robot_descriptions.ur10_description.URDF_PATH)
            case Robot.Panda:
                return pathlib.Path(robot_descriptions.panda_description.URDF_PATH)
            case Robot.Cassie:
                return pathlib.Path(robot_descriptions.cassie_description.URDF_PATH)
            case Robot.AnymalC:
                return pathlib.Path(robot_descriptions.anymal_c_description.URDF_PATH)
            case Robot.Laikago:
                return pathlib.Path(robot_descriptions.laikago_description.URDF_PATH)
            case Robot.DoublePendulum:
                return pathlib.Path(
                    robot_descriptions.double_pendulum_description.URDF_PATH
                )
            case _:
                raise ValueError(f"Unknown robot '{robot}'")
