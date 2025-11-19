from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

import jaxsim
import jaxsim.api as js
from jaxsim.api.kin_dyn_parameters import ScalingFactors


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


@pytest.mark.benchmark
def test_update_hw_parameters(
    jaxsim_model_garpez: js.model.JaxSimModel, benchmark, batch_size
):
    """Benchmark hardware parameter scaling/update operation (vmapped)."""
    model = jaxsim_model_garpez
    n_links = model.number_of_links()

    # Create a function that generates random scaling factors and updates the model
    def update_with_random_scaling(key: jax.Array) -> js.model.JaxSimModel:
        # Generate scaling factors in a reasonable range [0.8, 1.2]
        dims_scale = jax.random.uniform(key, shape=(n_links, 3), minval=0.8, maxval=1.2)
        density_scale = jax.random.uniform(
            jax.random.fold_in(key, 1), shape=(n_links,), minval=0.8, maxval=1.2
        )
        scaling_factors = ScalingFactors(dims=dims_scale, density=density_scale)
        return js.model.update_hw_parameters(model, scaling_factors)

    # Generate batch of random keys
    key = jax.random.PRNGKey(seed=42)
    keys = jax.random.split(key, num=batch_size)

    # Warm-up call to avoid including compilation time
    jax.vmap(update_with_random_scaling)(keys)

    # Benchmark the vmapped update operation
    benchmark(jax.block_until_ready(jax.vmap(update_with_random_scaling)), keys)


@pytest.mark.benchmark
def test_export_updated_model(
    jaxsim_model_garpez: js.model.JaxSimModel, benchmark, batch_size
):
    """Benchmark model export after hardware parameter update."""
    model = jaxsim_model_garpez
    n_links = model.number_of_links()

    # Create multiple scaled models for benchmarking
    # Each with slightly different scaling to simulate realistic scenarios
    key = jax.random.PRNGKey(seed=42)
    scaling_values = jax.random.uniform(
        key, shape=(batch_size,), minval=0.9, maxval=1.2
    )

    updated_models = []
    for scale_value in scaling_values:
        scaling_factors = ScalingFactors(
            dims=jnp.ones((n_links, 3)) * float(scale_value),
            density=jnp.ones(n_links),
        )
        updated_models.append(js.model.update_hw_parameters(model, scaling_factors))

    # Benchmark the export operation (sequentially for all models)
    # Note: This is not JIT-compiled since it returns a string (URDF/SDF)
    def export_all():
        return [m.export_updated_model() for m in updated_models]

    benchmark(export_all)
