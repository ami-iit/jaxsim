import pathlib

import jax_dataclasses
import numpy as np
from gym_ignition.rbd.idyntree import kindyncomputations

import jaxsim.high_level.model
from jaxsim import logging, sixd
from jaxsim.high_level.common import VelRepr
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.physics.model.physics_model_state import (
    PhysicsModelInput,
    PhysicsModelState,
)

# Initialize a global RNG used for all the tests
test_rng = None


def get_rng(seed: int = None) -> np.random.Generator:

    global test_rng

    if test_rng is not None and seed is not None:
        logging.warning(msg="Seed was already configured globally")

    seed = seed if seed is not None else 42
    test_rng = test_rng if test_rng is not None else np.random.default_rng(seed=seed)

    return test_rng


def random_physics_model_state(physics_model: PhysicsModel) -> PhysicsModelState:

    import jax_dataclasses

    rng = get_rng()
    zero_state = PhysicsModelState.zero(physics_model=physics_model)

    with jax_dataclasses.copy_and_mutate(zero_state) as state:

        state.joint_positions = rng.uniform(size=physics_model.dofs(), low=-1)
        state.joint_velocities = rng.uniform(size=physics_model.dofs(), low=-1)

        state.base_position = rng.uniform(size=3)
        state.base_quaternion = sixd.so3.SO3.from_rpy_radians(
            *rng.uniform(low=0, high=2 * np.pi, size=3)
        ).as_quaternion_xyzw()[np.array([3, 0, 1, 2])]

        if physics_model.is_floating_base:
            state.base_linear_velocity = rng.uniform(size=3, low=-1)
            state.base_angular_velocity = rng.uniform(size=3, low=-1)

    return state


def random_physics_model_input(physics_model: PhysicsModel) -> PhysicsModelInput:

    rng = get_rng()
    zero_input = PhysicsModelInput.zero(physics_model=physics_model)

    with jax_dataclasses.copy_and_mutate(zero_input) as model_input:

        model_input.tau = rng.uniform(size=physics_model.dofs(), low=-1)
        model_input.f_ext = rng.uniform(size=[physics_model.NB, 6], low=-1)

        if not physics_model.is_floating_base and physics_model.NB > 0:
            model_input.f_ext[0] = np.zeros(6)

    return model_input


# ========
# iDynTree
# ========


def get_kindyncomputations(
    model: jaxsim.high_level.model.Model,
    urdf_path: pathlib.Path,
    vel_repr: VelRepr = None,
) -> kindyncomputations.KinDynComputations:

    if not urdf_path.exists():
        raise ValueError(urdf_path)

    vel_repr = vel_repr if vel_repr is not None else model.velocity_representation

    map_vel_repr = {
        VelRepr.Body: kindyncomputations.FrameVelocityRepresentation.BODY_FIXED_REPRESENTATION,
        VelRepr.Mixed: kindyncomputations.FrameVelocityRepresentation.MIXED_REPRESENTATION,
        VelRepr.Inertial: kindyncomputations.FrameVelocityRepresentation.INERTIAL_FIXED_REPRESENTATION,
    }

    if vel_repr not in map_vel_repr.keys():
        raise ValueError(f"Velocity representation '{vel_repr}' not supported")

    # Create KinDynComputations and get the iDynTree model.
    # Note: pay attention to the serialization, here we take all joints and it's ok,
    #       however if any joint is left out, iDynTree considers it as a fixed joint and
    #       lumps parent/child links together applying the default joint position
    #       specified in the URDF.
    #       This might produce unexpected results if we forget that the removed joint
    #       angle is not zero.
    kin_dyn = kindyncomputations.KinDynComputations(
        world_gravity=np.array(model.physics_model.gravity[3:6]).astype(float),
        model_file=str(urdf_path),
        considered_joints=model.joint_names(),
        velocity_representation=map_vel_repr[vel_repr],
    )

    # Initialize the robot state of KinDynComputations
    kin_dyn.set_robot_state(
        s=np.array(model.joint_positions()).astype(float),
        ds=np.array(model.joint_velocities()).astype(float),
        world_H_base=np.array(model.base_transform()).astype(float),
        base_velocity=np.array(model.base_velocity()).astype(float),
    )

    return kin_dyn
