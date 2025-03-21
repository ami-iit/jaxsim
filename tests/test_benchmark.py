from collections.abc import Callable

import jax
import pytest

import jaxsim
import jaxsim.api as js


def vectorize_data(model: js.model.JaxSimModel, batch_size: int):
    key = jax.random.PRNGKey(seed=0)
    keys = jax.random.split(key, num=batch_size)

    return jax.vmap(
        lambda key: js.data.random_model_data(
            model=model,
            key=key,
        )
    )(keys)


def benchmark_test_function(
    func: Callable, model: js.model.JaxSimModel, benchmark, batch_size
):
    """Reusability wrapper for benchmark tests."""
    data = vectorize_data(model=model, batch_size=batch_size)

    # Warm-up call to avoid including compilation time
    jax.vmap(func, in_axes=(None, 0))(model, data)

    # Benchmark the function call
    # Note: jax.block_until_ready is used to ensure that the benchmark is not measuring only the asynchronous dispatch
    benchmark(jax.block_until_ready(jax.vmap(func, in_axes=(None, 0))), model, data)


@pytest.mark.benchmark
def test_forward_dynamics_aba(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(js.model.forward_dynamics_aba, model, benchmark, batch_size)


@pytest.mark.benchmark
def test_free_floating_bias_forces(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(
        js.model.free_floating_bias_forces, model, benchmark, batch_size
    )


@pytest.mark.benchmark
def test_forward_kinematics(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(js.model.forward_kinematics, model, benchmark, batch_size)


@pytest.mark.benchmark
def test_free_floating_mass_matrix(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(
        js.model.free_floating_mass_matrix, model, benchmark, batch_size
    )


@pytest.mark.benchmark
def test_free_floating_jacobian(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(
        js.model.generalized_free_floating_jacobian, model, benchmark, batch_size
    )


@pytest.mark.benchmark
def test_free_floating_jacobian_derivative(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    benchmark_test_function(
        js.model.generalized_free_floating_jacobian_derivative,
        model,
        benchmark,
        batch_size,
    )


@pytest.mark.benchmark
def test_soft_contact_model(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    with model.editable(validate=False) as model:
        model.contact_model = jaxsim.rbda.contacts.SoftContacts()
        model.contact_params = js.contact.estimate_good_contact_parameters(model=model)

    benchmark_test_function(js.ode.system_dynamics, model, benchmark, batch_size)


@pytest.mark.benchmark
def test_rigid_contact_model(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    with model.editable(validate=False) as model:
        model.contact_model = jaxsim.rbda.contacts.RigidContacts()
        model.contact_params = js.contact.estimate_good_contact_parameters(model=model)

    benchmark_test_function(js.ode.system_dynamics, model, benchmark, batch_size)


@pytest.mark.benchmark
def test_relaxed_rigid_contact_model(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    with model.editable(validate=False) as model:
        model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts()
        model.contact_params = js.contact.estimate_good_contact_parameters(model=model)

    benchmark_test_function(js.ode.system_dynamics, model, benchmark, batch_size)


@pytest.mark.benchmark
def test_simulation_step(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark, batch_size
):
    model = jaxsim_model_ergocub_reduced

    with model.editable(validate=False) as model:
        model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts()
        model.contact_params = js.contact.estimate_good_contact_parameters(model=model)

    benchmark_test_function(js.model.step, model, benchmark, batch_size)
