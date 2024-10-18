import functools

import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js
import jaxsim.integrators
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import VelRepr


def test_box_with_external_forces(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
):
    """
    This test simulates a box falling due to gravity.
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
    mg = data0.standard_gravity() * js.model.total_mass(model=model)
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

    # Create the integrator.
    integrator = jaxsim.integrators.fixed_step.RungeKutta4SO3.build(
        dynamics=js.ode.wrap_system_dynamics_for_integration(
            model=model, data=data0, system_dynamics=js.ode.system_dynamics
        )
    )

    # Initialize the integrator.
    tf = 0.5
    T_ns = jnp.arange(start=0, stop=tf * 1e9, step=model.time_step * 1e9, dtype=int)
    integrator_state = integrator.init(x0=data0.state, t0=0.0, dt=model.time_step)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T_ns:

        data, integrator_state = js.model.step(
            model=model,
            data=data,
            integrator=integrator,
            integrator_state=integrator_state,
            link_forces=references.link_forces(model=model, data=data),
        )

    # Check that the box didn't move.
    assert data.base_position() == pytest.approx(data0.base_position())
    assert data.base_orientation() == pytest.approx(data0.base_orientation())


def test_box_with_zero_gravity(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jnp.ndarray,
):

    model = jaxsim_model_box

    # Move the terrain (almost) infinitely far away from the box.
    with model.editable(validate=False) as model:
        model.terrain = jaxsim.terrain.FlatTerrain.build(height=-1e9)

    # Split the PRNG key.
    _, subkey = jax.random.split(prng_key, num=2)

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jax.random.uniform(subkey, shape=(3,)),
        velocity_representation=velocity_representation,
        standard_gravity=0.0,
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

    # Create the integrator.
    integrator = jaxsim.integrators.fixed_step.RungeKutta4SO3.build(
        dynamics=js.ode.wrap_system_dynamics_for_integration(
            model=model, data=data0, system_dynamics=js.ode.system_dynamics
        )
    )

    # Initialize the integrator.
    tf, dt = 1.0, 0.010
    T_ns = jnp.arange(start=0, stop=tf * 1e9, step=dt * 1e9, dtype=int)
    integrator_state = integrator.init(x0=data0.state, t0=0.0, dt=dt)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T_ns:

        with (
            data.switch_velocity_representation(velocity_representation),
            references.switch_velocity_representation(velocity_representation),
        ):

            data, integrator_state = js.model.step(
                model=model,
                data=data,
                dt=dt,
                integrator=integrator,
                integrator_state=integrator_state,
                link_forces=references.link_forces(model=model, data=data),
            )

    # Check that the box moved as expected.
    assert data.base_position() == pytest.approx(
        data0.base_position()
        + 0.5 * LW_f[:, :3].squeeze() / js.model.total_mass(model=model) * tf**2,
        abs=1e-3,
    )


def run_simulation(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    dt: jtp.FloatLike,
    tf: jtp.FloatLike,
) -> js.data.JaxSimModelData:

    @functools.cache
    def get_integrator() -> tuple[jaxsim.integrators.Integrator, dict[str, jtp.PyTree]]:

        # Create the integrator.
        integrator = jaxsim.integrators.fixed_step.Heun2.build(
            fsal_enabled_if_supported=False,
            dynamics=js.ode.wrap_system_dynamics_for_integration(
                model=model,
                data=data_t0,
                system_dynamics=js.ode.system_dynamics,
            ),
        )

        # Initialize the integrator state.
        integrator_state_t0 = integrator.init(x0=data_t0.state, t0=0.0, dt=dt)

        return integrator, integrator_state_t0

    # Initialize the integration horizon.
    T_ns = jnp.arange(start=0.0, stop=int(tf * 1e9), step=int(dt * 1e9)).astype(int)

    # Initialize the simulation data.
    integrator = None
    integrator_state = None
    data = data_t0.copy()

    for t_ns in T_ns:

        match model.contact_model:

            case jaxsim.rbda.contacts.ViscoElasticContacts():

                data, _ = jaxsim.rbda.contacts.visco_elastic.step(
                    model=model,
                    data=data,
                    dt=dt,
                )

            case _:

                integrator, integrator_state = (
                    get_integrator() if t_ns == 0 else (integrator, integrator_state)
                )

                data, integrator_state = js.model.step(
                    model=model,
                    data=data,
                    dt=dt,
                    integrator=integrator,
                    integrator_state=integrator_state,
                )

    return data


def test_simulation_with_soft_contacts(
    jaxsim_model_box: js.model.JaxSimModel,
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.SoftContacts.build(
            terrain=model.terrain,
        )

    # Initialize the maximum penetration of each collidable point at steady state.
    max_penetration = 0.001

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
        contacts_params=js.contact.estimate_good_contact_parameters(
            model=model,
            number_of_active_collidable_points_steady_state=4,
            static_friction_coefficient=1.0,
            damping_ratio=1.0,
            max_penetration=0.001,
        ),
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, dt=0.001, tf=1.0)

    assert data_tf.base_position()[0:2] == pytest.approx(data_t0.base_position()[0:2])
    assert data_tf.base_position()[2] + max_penetration == pytest.approx(box_height / 2)


def test_simulation_with_visco_elastic_contacts(
    jaxsim_model_box: js.model.JaxSimModel,
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.ViscoElasticContacts.build(
            terrain=model.terrain,
        )

    # Initialize the maximum penetration of each collidable point at steady state.
    max_penetration = 0.001

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
        contacts_params=js.contact.estimate_good_contact_parameters(
            model=model,
            number_of_active_collidable_points_steady_state=4,
            static_friction_coefficient=1.0,
            damping_ratio=1.0,
            max_penetration=0.001,
        ),
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, dt=0.001, tf=1.0)

    assert data_tf.base_position()[0:2] == pytest.approx(data_t0.base_position()[0:2])
    assert data_tf.base_position()[2] + max_penetration == pytest.approx(box_height / 2)


def test_simulation_with_rigid_contacts(
    jaxsim_model_box: js.model.JaxSimModel,
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.RigidContacts.build(
            terrain=model.terrain,
        )

    # Initialize the maximum penetration of each collidable point at steady state.
    # This model is rigid, so we expect (almost) no penetration.
    max_penetration = 0.000

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
        # In order to achieve almost no penetration, we need to use a fairly large
        # Baumgarte stabilization term.
        contacts_params=js.contact.estimate_good_contact_parameters(
            model=model,
            K=100_000,
        ),
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, dt=0.001, tf=1.0)

    assert data_tf.base_position()[0:2] == pytest.approx(data_t0.base_position()[0:2])
    assert data_tf.base_position()[2] + max_penetration == pytest.approx(box_height / 2)


def test_simulation_with_relaxed_rigid_contacts(
    jaxsim_model_box: js.model.JaxSimModel,
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts.build(
            terrain=model.terrain,
        )

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
        # For this contact model, the following method is practically no-op.
        # Let's leave it there for consistency and to make sure that nothing
        # gets broken if it is updated in the future.
        contacts_params=js.contact.estimate_good_contact_parameters(
            model=model,
        ),
    )
    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, dt=0.001, tf=1.0)

    # With this contact model, we need to slightly increase the tolerances.
    assert data_tf.base_position()[0:2] == pytest.approx(
        data_t0.base_position()[0:2], abs=0.000_010
    )
    assert data_tf.base_position()[2] + max_penetration == pytest.approx(
        box_height / 2, abs=0.000_100
    )
