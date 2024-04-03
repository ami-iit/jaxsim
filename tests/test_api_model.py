import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rod

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils_idyntree


def test_model_creation_and_reduction(
    jaxsim_model_ergocub: js.model.JaxSimModel,
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel,
):

    model_full = jaxsim_model_ergocub
    model_reduced = jaxsim_model_ergocub_reduced

    # Build the data of the full model.
    data = js.data.JaxSimModelData.build(
        model=model_full,
        base_position=jnp.array([0, 0, 0.8]),
        velocity_representation=VelRepr.Inertial,
    )

    # =====
    # Tests
    # =====

    # Check that the data of the full model is valid.
    assert data.valid(model=model_full)

    # Build the ROD model from the original description.
    assert isinstance(model_full.built_from, (str, pathlib.Path))
    rod_sdf = rod.Sdf.load(sdf=model_full.built_from)
    assert len(rod_sdf.models()) == 1

    # Get all non-fixed joint names from the description.
    joint_names_in_description = [
        j.name for j in rod_sdf.models()[0].joints() if j.type != "fixed"
    ]

    # Check that all non-fixed joints are in the full model.
    assert set(joint_names_in_description) == set(model_full.joint_names())

    # Build the data of the reduced model.
    data_reduced = js.data.JaxSimModelData.build(
        model=model_reduced,
        base_position=jnp.array([0, 0, 0.8]),
        velocity_representation=VelRepr.Inertial,
    )

    # Check that the reduced model data is valid.
    assert not data_reduced.valid(model=model_full)
    assert data_reduced.valid(model=model_reduced)

    # Check that the total mass is preserved.
    assert js.model.total_mass(model=model_full) == pytest.approx(
        js.model.total_mass(model=model_reduced)
    )

    # Check that the CoM position is preserved.
    assert js.com.com_position(model=model_full, data=data) == pytest.approx(
        js.com.com_position(model=model_reduced, data=data_reduced), abs=1e-6
    )


def test_model_properties(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    m_idt = kin_dyn.total_mass()
    m_js = js.model.total_mass(model=model)
    assert pytest.approx(m_idt) == m_js

    J_Bh_idt = kin_dyn.total_momentum_jacobian()
    J_Bh_js = js.model.total_momentum_jacobian(model=model, data=data)
    assert pytest.approx(J_Bh_idt) == J_Bh_js

    h_tot_idt = kin_dyn.total_momentum()
    h_tot_js = js.model.total_momentum(model=model, data=data)
    assert pytest.approx(h_tot_idt) == h_tot_js

    M_locked_idt = kin_dyn.locked_spatial_inertia()
    M_locked_js = js.model.locked_spatial_inertia(model=model, data=data)
    assert pytest.approx(M_locked_idt) == M_locked_js

    J_avg_idt = kin_dyn.average_velocity_jacobian()
    J_avg_js = js.model.average_velocity_jacobian(model=model, data=data)
    assert pytest.approx(J_avg_idt) == J_avg_js

    v_avg_idt = kin_dyn.average_velocity()
    v_avg_js = js.model.average_velocity(model=model, data=data)
    assert pytest.approx(v_avg_idt) == v_avg_js


def test_model_rbda(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
    velocity_representation: VelRepr,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils_idyntree.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    # =====
    # Tests
    # =====

    # Support both fixed-base and floating-base models by slicing the first six rows
    sl = np.s_[0:] if model.floating_base() else np.s_[6:]

    # Mass matrix
    M_idt = kin_dyn.mass_matrix()
    M_js = js.model.free_floating_mass_matrix(model=model, data=data)
    assert pytest.approx(M_idt[sl, sl]) == M_js[sl, sl]

    # Gravity forces
    g_idt = kin_dyn.gravity_forces()
    g_js = js.model.free_floating_gravity_forces(model=model, data=data)
    assert pytest.approx(g_idt[sl]) == g_js[sl]

    # Bias forces
    h_idt = kin_dyn.bias_forces()
    h_js = js.model.free_floating_bias_forces(model=model, data=data)
    assert pytest.approx(h_idt[sl]) == h_js[sl]

    # Forward kinematics
    HH_js = js.model.forward_kinematics(model=model, data=data)
    HH_idt = jnp.stack(
        [kin_dyn.frame_transform(frame_name=name) for name in model.link_names()]
    )
    assert pytest.approx(HH_idt) == HH_js

    # Bias accelerations
    Jν_js = js.model.link_bias_accelerations(model=model, data=data)
    Jν_idt = jnp.stack(
        [kin_dyn.frame_bias_acc(frame_name=name) for name in model.link_names()]
    )
    assert pytest.approx(Jν_idt) == Jν_js


def test_model_jacobian(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    # =====
    # Tests
    # =====

    # Create random references (joint torques and link forces)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    # Get the J.T @ f product in inertial-fixed input/output representation.
    # We use doubly right-trivialized jacobian with inertial-fixed 6D forces.
    with references.switch_velocity_representation(VelRepr.Inertial):
        with data.switch_velocity_representation(VelRepr.Inertial):

            f = references.link_forces(model=model, data=data)
            assert f == pytest.approx(references.input.physics_model.f_ext)

            J = js.model.generalized_free_floating_jacobian(model=model, data=data)
            JTf_inertial = jnp.einsum("l6g,l6->g", J, f)

    for vel_repr in [VelRepr.Body, VelRepr.Mixed]:
        with references.switch_velocity_representation(vel_repr):

            # Get the jacobian having an inertial-fixed input representation (so that
            # it computes the same quantity computed above) and an output representation
            # compatible with the frame in which the external forces are expressed.
            with data.switch_velocity_representation(VelRepr.Inertial):

                J = js.model.generalized_free_floating_jacobian(
                    model=model, data=data, output_vel_repr=vel_repr
                )

            # Get the forces in the tested representation and compute the product
            # O_J_WL_W.T @ O_f, producing a generalized acceleration in W.
            # The resulting acceleration can be tested again the one computed before.
            with data.switch_velocity_representation(vel_repr):

                f = references.link_forces(model=model, data=data)
                JTf_other = jnp.einsum("l6g,l6->g", J, f)
                assert pytest.approx(JTf_inertial) == JTf_other, vel_repr.name


def test_model_fd_id_consistency(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    # =====
    # Tests
    # =====

    # Create random references (joint torques and link forces)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    # Compute forward dynamics with ABA
    v̇_WB_aba, s̈_aba = js.model.forward_dynamics_aba(
        model=model,
        data=data,
        joint_forces=references.joint_force_references(),
        link_forces=references.link_forces(model=model, data=data),
    )

    # Compute forward dynamics with CRB
    v̇_WB_crb, s̈_crb = js.model.forward_dynamics_crb(
        model=model,
        data=data,
        joint_forces=references.joint_force_references(),
        link_forces=references.link_forces(model=model, data=data),
    )

    assert pytest.approx(s̈_aba) == s̈_crb
    assert pytest.approx(v̇_WB_aba) == v̇_WB_crb

    # Compute inverse dynamics with the quantities computed by forward dynamics
    fB_id, τ_id = js.model.inverse_dynamics(
        model=model,
        data=data,
        joint_accelerations=s̈_aba,
        base_acceleration=v̇_WB_aba,
        link_forces=references.link_forces(model=model, data=data),
    )

    # Check consistency between FD and ID
    assert pytest.approx(τ_id) == references.joint_force_references(model=model)
    assert pytest.approx(fB_id, abs=1e-9) == jnp.zeros(6)

    if model.floating_base():
        # If we remove the base 6D force from the inputs, we should find it as output.
        fB_id, τ_id = js.model.inverse_dynamics(
            model=model,
            data=data,
            joint_accelerations=s̈_aba,
            base_acceleration=v̇_WB_aba,
            link_forces=references.link_forces(model=model, data=data)
            .at[0]
            .set(jnp.zeros(6)),
        )

        assert pytest.approx(τ_id) == references.joint_force_references(model=model)
        assert (
            pytest.approx(fB_id, abs=1e-9)
            == references.link_forces(model=model, data=data)[0]
        )
