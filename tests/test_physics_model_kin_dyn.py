import pathlib

import gym_ignition.rbd.idyntree.kindyncomputations as kindyncomputations
import gym_ignition_models
import numpy as np
import pytest
import utils_testing

from jaxsim.high_level.model import Model, VelRepr


@pytest.mark.parametrize(
    "model_name, vel_repr",
    [
        ("pendulum", VelRepr.Inertial),
        ("pendulum", VelRepr.Body),
        # ("pendulum", VelRepr.Mixed),
        ("panda", VelRepr.Inertial),
        ("panda", VelRepr.Body),
        # ("panda", VelRepr.Mixed),
        ("iCubGazeboV2_5", VelRepr.Inertial),
        ("iCubGazeboV2_5", VelRepr.Body),
        # ("iCubGazeboV2_5", VelRepr.Mixed),
    ],
)
def test_physics_model_kin_dyn(model_name: str, vel_repr: VelRepr):

    gravity = np.array([0, 0, -10.0])

    sdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.SDF_PATH
    )

    urdf_file_path = gym_ignition_models.get_model_resource(
        robot_name=model_name, resource_type=gym_ignition_models.ResourceType.URDF_PATH
    )

    # Build the JAXsim model
    model = Model.build_from_sdf(sdf=sdf_file_path, vel_repr=vel_repr, gravity=gravity)

    # Random model data
    model = model.update_data(
        model_state=utils_testing.random_physics_model_state(model.physics_model),
        model_input=utils_testing.random_physics_model_input(model.physics_model),
    )

    # ==========================
    # Ground truth with iDynTree
    # ==========================

    kin_dyn = utils_testing.get_kindyncomputations(
        model=model, urdf_path=pathlib.Path(urdf_file_path)
    )

    # Get the iDynTree model
    idt_model: kindyncomputations.idt.Model = kin_dyn.kindyn.model()

    # Ground truth with iDynTree
    M_idyntree = kin_dyn.get_mass_matrix()
    h_idyntree = kin_dyn.get_bias_forces()
    g_idyntree = kin_dyn.get_generalized_gravity_forces()
    J_idyntree = np.vstack(
        [
            kin_dyn.get_frame_jacobian(frame_name=link_name)
            for link_name in model.physics_model.description.link_names()
        ]
    )

    # ================
    # Test with JAXsim
    # ================

    assert model.free_floating_mass_matrix() == pytest.approx(M_idyntree, abs=0.050)

    if model.floating_base():
        assert model.free_floating_gravity_forces() == pytest.approx(
            g_idyntree, abs=0.100
        )
        assert model.free_floating_generalized_forces() == pytest.approx(
            h_idyntree, abs=0.100
        )
    else:
        assert model.free_floating_gravity_forces()[6:] == pytest.approx(
            g_idyntree[6:], abs=0.100
        )
        assert model.free_floating_generalized_forces()[6:] == pytest.approx(
            h_idyntree[6:], abs=0.100
        )

    assert model.generalized_jacobian() == pytest.approx(J_idyntree, abs=0.001)

    for link in model.links():

        assert link.transform() == pytest.approx(
            kin_dyn.get_world_transform(frame_name=link.name()), abs=0.001
        ), link.name()

        assert link.jacobian() == pytest.approx(
            kin_dyn.get_frame_jacobian(frame_name=link.name()), abs=0.001
        ), link.name()

    for link in model.links():

        if link.name() == model.base_frame():
            continue

        link_idt: kindyncomputations.idt.Link = idt_model.getLink(
            kin_dyn.kindyn.getFrameIndex(link.name())
        )

        assert link.mass() == pytest.approx(
            link_idt.getInertia().asVector().toNumPy()[0], abs=0.001
        ), link.name()

        assert link.spatial_inertia() == pytest.approx(
            link_idt.getInertia().asMatrix().toNumPy(), abs=0.001
        ), link.name()
