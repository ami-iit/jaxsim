import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import VelRepr


def test_box_with_external_forces(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
):
    """
    Simulate a box falling due to gravity.

    We apply to its CoM a 6D force that balances exactly the gravitational force.
    The box should not fall.
    """

    model = jaxsim_model_box

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.5]),
        velocity_representation=velocity_representation,
    )

    # Compute the force due to gravity at the CoM.
    mg = -model.gravity * js.model.total_mass(model=model)
    G_f = jnp.array([0.0, 0.0, mg, 0, 0, 0])

    # Compute the position of the CoM expressed in the coordinates of the link frame L.
    L_p_CoM = js.link.com_position(
        model=model, data=data0, link_index=0, in_link_frame=True
    )

    # Compute the transform of 6D forces from the CoM to the link frame.
    L_H_G = jaxsim.math.Transform.from_quaternion_and_translation(translation=L_p_CoM)
    G_Xv_L = jaxsim.math.Adjoint.from_transform(transform=L_H_G, inverse=True)
    L_Xf_G = G_Xv_L.T
    L_f = L_Xf_G @ G_f

    # Initialize a references object that simplifies handling external forces.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data0,
        velocity_representation=velocity_representation,
    )

    # Apply a link forces to the base link.
    with references.switch_velocity_representation(VelRepr.Body):
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(L_f),
            link_names=model.link_names()[0:1],
            model=model,
            data=data0,
            additive=False,
        )

    # Initialize the integrator.
    tf = 0.5
    T_ns = jnp.arange(start=0, stop=tf * 1e9, step=model.time_step * 1e9, dtype=int)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T_ns:

        data = js.model.step(
            model=model,
            data=data,
            link_forces=references.link_forces(model, data),
        )

    # Check that the box didn't move.
    assert data.base_position == pytest.approx(data0.base_position)
    assert data.base_orientation == pytest.approx(data0.base_orientation)


def test_box_with_zero_gravity(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jnp.ndarray,
):

    model = jaxsim_model_box

    # Move the terrain (almost) infinitely far away from the box.
    with model.editable(validate=False) as model:
        model.terrain = jaxsim.terrain.FlatTerrain.build(height=-1e9)
        model.gravity = 0.0

    # Split the PRNG key.
    _, subkey = jax.random.split(prng_key, num=2)

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jax.random.uniform(subkey, shape=(3,)),
        velocity_representation=velocity_representation,
    )

    # Initialize a references object that simplifies handling external forces.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data0,
        velocity_representation=velocity_representation,
    )

    # Apply a link forces to the base link.
    with references.switch_velocity_representation(jaxsim.VelRepr.Mixed):

        # Generate a random linear force.
        # We enforce them to be the same for all velocity representations so that
        # we can compare their outcomes.
        LW_f = 10.0 * (
            jax.random.uniform(jax.random.key(0), shape=(model.number_of_links(), 6))
            .at[:, 3:]
            .set(jnp.zeros(3))
        )

        # Note that the context manager does not switch back the newly created
        # `references` (that is not the yielded object) to the original representation.
        # In the simulation loop below, we need to make sure that we switch both `data`
        # and `references` to the same representation before extracting the information
        # passed to the step function.
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(LW_f),
            link_names=model.link_names(),
            model=model,
            data=data0,
            additive=False,
        )

    tf = 0.01
    T = jnp.arange(start=0, stop=tf * 1e9, step=model.time_step * 1e9, dtype=int)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T:
        with (
            data.switch_velocity_representation(velocity_representation),
            references.switch_velocity_representation(velocity_representation),
        ):
            data = js.model.step(
                model=model,
                data=data,
                link_forces=references.link_forces(model=model, data=data),
            )

    # Check that the box moved as expected.
    assert data.base_position == pytest.approx(
        data0.base_position
        + 0.5 * LW_f[:, :3].squeeze() / js.model.total_mass(model=model) * tf**2,
        abs=1e-3,
    )


def run_simulation(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    tf: jtp.FloatLike,
) -> js.data.JaxSimModelData:

    # Initialize the integration horizon.
    T_ns = jnp.arange(
        start=0.0, stop=int(tf * 1e9), step=int(model.time_step * 1e9)
    ).astype(int)

    # Initialize the simulation data.
    data = data_t0.copy()

    for _ in T_ns:

        data = js.model.step(
            model=model,
            data=data,
        )

    return data


def test_simulation_with_relaxed_rigid_contacts(
    jaxsim_model_box: js.model.JaxSimModel, integrator
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts.build(
            solver_options={"tol": 1e-3},
        )
        # Enable a subset of the collidable points.
        enabled_collidable_points_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_collidable_points_mask[[0, 1, 2, 3]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(
            enabled_collidable_points_mask.tolist()
        )
        model.integrator = integrator

    assert np.sum(model.kin_dyn_parameters.contact_parameters.enabled) == 4

    # Initialize the maximum penetration of each collidable point at steady state.
    # This model is quasi-rigid, so we expect (almost) no penetration.
    max_penetration = 0.000

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, tf=1.0)

    # With this contact model, we need to slightly increase the tolerances.
    assert data_tf.base_position[0:2] == pytest.approx(
        data_t0.base_position[0:2], abs=0.000_010
    )
    assert data_tf.base_position[2] + max_penetration == pytest.approx(
        box_height / 2, abs=0.000_100
    )


def test_joint_limits(
    jaxsim_model_single_pendulum: js.model.JaxSimModel,
):

    model = jaxsim_model_single_pendulum

    with model.editable(validate=False) as model:
        model.kin_dyn_parameters.joint_parameters.position_limits_max = jnp.atleast_1d(
            jnp.array(1.5708)
        )
        model.kin_dyn_parameters.joint_parameters.position_limits_min = jnp.atleast_1d(
            jnp.array(-1.5708)
        )
        model.kin_dyn_parameters.joint_parameters.position_limit_spring = (
            jnp.atleast_1d(jnp.array(75.0))
        )
        model.kin_dyn_parameters.joint_parameters.position_limit_damper = (
            jnp.atleast_1d(jnp.array(0.1))
        )

    position_limits_min, position_limits_max = js.joint.position_limits(model=model)

    data = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
    )

    theta = 10 * np.pi / 180

    # Define a tolerance since the spring-damper model does
    # not guarantee that the joint position will be exactly
    # below the limit.
    tolerance = theta * 0.10

    # Test minimum joint position limits.
    data_t0 = data.replace(model=model, joint_positions=position_limits_min - theta)

    model = model.replace(time_step=0.005, validate=False)
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=3.0)

    assert (
        np.min(np.array(data_tf.joint_positions), axis=0) + tolerance
        >= position_limits_min
    )

    # Test maximum joint position limits.
    data_t0 = data.replace(model=model, joint_positions=position_limits_max - theta)

    model = model.replace(time_step=0.001)
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=3.0)

    assert (
        np.max(np.array(data_tf.joint_positions), axis=0) - tolerance
        <= position_limits_max
    )
