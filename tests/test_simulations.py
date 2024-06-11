import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js
import jaxsim.integrators
import jaxsim.rbda
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
    dt = 0.001
    T = jnp.arange(start=0, stop=tf * 1e9, step=dt * 1e9, dtype=int)
    integrator_state = integrator.init(x0=data0.state, t0=0.0, dt=dt)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for t_ns in T:

        data, integrator_state = js.model.step(
            model=model,
            data=data,
            dt=dt,
            integrator=integrator,
            integrator_state=integrator_state,
            link_forces=references.link_forces(model=model, data=data),
        )

    # Check that the box didn't move.
    assert data.time() == t_ns / 1e9 + dt
    assert data.base_position() == pytest.approx(data0.base_position())
    assert data.base_orientation() == pytest.approx(data0.base_orientation())


def test_box_with_zero_gravity(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jnp.ndarray,
):

    model = jaxsim_model_box

    # Split the PRNG key.
    key, subkey, subkey2 = jax.random.split(prng_key, num=3)

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jax.random.uniform(subkey2, shape=(3,)),
        velocity_representation=velocity_representation,
        standard_gravity=0.0,
        soft_contacts_params=jaxsim.rbda.SoftContactsParams.build(K=0.0, D=0.0, mu=0.0),
    )

    # Generate a random linear force.
    L_f = (
        jax.random.uniform(subkey, shape=(model.number_of_links(), 6))
        .at[:, 3:]
        .set(jnp.zeros(3))
    )

    # Initialize a references object that simplifies handling external forces.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data0,
        velocity_representation=velocity_representation,
    )

    # Apply a link forces to the base link.
    references = references.apply_link_forces(
        forces=jnp.atleast_2d(L_f),
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
    tf = 1.0
    dt = 0.010
    T = jnp.arange(start=0, stop=tf * 1e9, step=dt * 1e9, dtype=int)
    integrator_state = integrator.init(x0=data0.state, t0=0.0, dt=dt)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for t_ns in T:

        data, integrator_state = js.model.step(
            model=model,
            data=data,
            dt=dt,
            integrator=integrator,
            integrator_state=integrator_state,
            link_forces=references.link_forces(model=model, data=data),
        )

    # Check that the box moved as expected.
    assert data.time() == t_ns / 1e9 + dt
    assert data.base_position() == pytest.approx(
        data0.base_position()
        + 0.5 * L_f[:, :3].squeeze() / js.model.total_mass(model=model) * tf**2,
        abs=1e-3,
    )
