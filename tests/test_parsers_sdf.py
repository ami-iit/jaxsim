import gym_ignition_models
import jax.numpy as jnp
import pytest

from jaxsim.parsers.sdf import build_model_from_sdf


def test_pendulum():

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name="pendulum", resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )

    model_description = build_model_from_sdf(sdf=sdf_file_path)

    assert model_description.name == "pendulum"
    assert model_description.fixed_base is False

    assert model_description.root_pose.root_position == pytest.approx(jnp.zeros(3))
    assert model_description.root_pose.root_quaternion == pytest.approx(
        jnp.array([1, 0, 0, 0])
    )

    assert model_description.root.name == "support"
    assert set(model_description.link_names()) == {"support", "pendulum"}
    assert set(model_description.frame_names()) == set()
    assert set(model_description.joint_names()) == {"pivot"}


def test_panda():

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name="panda", resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )

    model_description = build_model_from_sdf(sdf=sdf_file_path)

    assert model_description.name == "panda"
    assert model_description.fixed_base is True

    assert model_description.root_pose.root_position == pytest.approx(jnp.zeros(3))
    assert model_description.root_pose.root_quaternion == pytest.approx(
        jnp.array([1, 0, 0, 0])
    )

    assert model_description.root.name == "panda_link0"
    assert set(model_description.link_names()) == {
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_leftfinger",
        "panda_rightfinger",
    }
    assert set(model_description.frame_names()) == {"end_effector_frame"}
    assert set(model_description.joint_names()) == {
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    }


def test_icub():

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name="iCubGazeboV2_5",
        resource_type=gym_ignition_models.ResourceType.SDF_PATH,
    )

    model_description = build_model_from_sdf(sdf=sdf_file_path)

    assert model_description.name == "iCubGazeboV2_5"
    assert model_description.fixed_base is False

    assert model_description.root_pose.root_position == pytest.approx(
        jnp.array([0, 0, 0.63])
    )
    assert model_description.root_pose.root_quaternion == pytest.approx(
        jnp.array([0, 0, 0, 1]), abs=1e-3
    )

    assert model_description.root.name == "base_link"
    assert set(model_description.link_names()) == {
        "base_link",
        "l_hip_1",
        "r_hip_1",
        "torso_1",
        "l_hip_2",
        "r_hip_2",
        "torso_2",
        "l_upper_leg",
        "r_upper_leg",
        "chest",
        "l_lower_leg",
        "r_lower_leg",
        "l_shoulder_1",
        "neck_1",
        "r_shoulder_1",
        "l_ankle_1",
        "r_ankle_1",
        "l_shoulder_2",
        "neck_2",
        "r_shoulder_2",
        "l_ankle_2",
        "r_ankle_2",
        "l_shoulder_3",
        "head",
        "r_shoulder_3",
        "l_elbow_1",
        "r_elbow_1",
        "l_forearm",
        "r_forearm",
        "l_wrist_1",
        "r_wrist_1",
        "l_hand",
        "r_hand",
    }
    assert set(model_description.frame_names()) == {
        "l_foot",
        "l_hip_3",
        "l_upper_arm",
        "r_foot",
        "r_hip_3",
        "r_upper_arm",
    }
    assert set(model_description.joint_names()) == {
        "l_hip_pitch",
        "r_hip_pitch",
        "torso_pitch",
        "l_hip_roll",
        "r_hip_roll",
        "torso_roll",
        "l_hip_yaw",
        "r_hip_yaw",
        "torso_yaw",
        "l_knee",
        "r_knee",
        "l_shoulder_pitch",
        "neck_pitch",
        "r_shoulder_pitch",
        "l_ankle_pitch",
        "r_ankle_pitch",
        "l_shoulder_roll",
        "neck_roll",
        "r_shoulder_roll",
        "l_ankle_roll",
        "r_ankle_roll",
        "l_shoulder_yaw",
        "neck_yaw",
        "r_shoulder_yaw",
        "l_elbow",
        "r_elbow",
        "l_wrist_prosup",
        "r_wrist_prosup",
        "l_wrist_pitch",
        "r_wrist_pitch",
        "l_wrist_yaw",
        "r_wrist_yaw",
    }


def test_icub_reduced():

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name="iCubGazeboV2_5",
        resource_type=gym_ignition_models.ResourceType.SDF_PATH,
    )

    considered_joints = [
        "l_knee",
        "r_knee",
        "l_elbow",
        "r_elbow",
        "neck_pitch",
        "neck_roll",
        "neck_yaw",
    ]

    model_description = build_model_from_sdf(sdf=sdf_file_path).reduce(
        considered_joints=considered_joints
    )

    assert model_description.name == "iCubGazeboV2_5"
    assert model_description.fixed_base is False

    assert model_description.root_pose.root_position == pytest.approx(
        jnp.array([0, 0, 0.63])
    )
    assert model_description.root_pose.root_quaternion == pytest.approx(
        jnp.array([0, 0, 0, 1]), abs=1e-3
    )

    assert model_description.root.name == "base_link"
    assert set(model_description.link_names()) == {
        "base_link",
        "l_elbow_1",
        "l_lower_leg",
        "neck_1",
        "r_elbow_1",
        "r_lower_leg",
        "neck_2",
        "head",
    }
    assert set(model_description.frame_names()) == {
        "chest",
        "l_ankle_1",
        "l_ankle_2",
        "l_forearm",
        "l_hand",
        "l_hip_1",
        "l_hip_2",
        "l_shoulder_1",
        "l_shoulder_2",
        "l_shoulder_3",
        "l_upper_leg",
        "l_wrist_1",
        "r_ankle_1",
        "r_ankle_2",
        "r_forearm",
        "r_hand",
        "r_hip_1",
        "r_hip_2",
        "r_shoulder_1",
        "r_shoulder_2",
        "r_shoulder_3",
        "r_upper_leg",
        "r_wrist_1",
        "torso_1",
        "torso_2",
    }
    assert set(model_description.joint_names()) == set(considered_joints)
