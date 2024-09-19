import pytest

import jaxsim.api as js


# Define the test functions for pytest-benchmark
@pytest.mark.benchmark
def test_free_floating_jacobian(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark
):
    model = jaxsim_model_ergocub_reduced
    data = js.data.random_model_data(model=model)

    # Warm-up call to avoid including compilation time
    js.model.generalized_free_floating_jacobian(model=model, data=data)

    benchmark(js.model.generalized_free_floating_jacobian, model, data)


@pytest.mark.benchmark
def test_free_floating_bias_forces(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark
):
    model = jaxsim_model_ergocub_reduced
    data = js.data.random_model_data(model=model)

    # Warm-up call to avoid including compilation time
    js.model.free_floating_bias_forces(model=model, data=data)

    benchmark(js.model.free_floating_bias_forces, model, data)


@pytest.mark.benchmark
def test_free_floating_mass_matrix(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark
):
    model = jaxsim_model_ergocub_reduced
    data = js.data.random_model_data(model=model)

    # Warm-up call to avoid including compilation time
    js.model.free_floating_mass_matrix(model=model, data=data)

    benchmark(js.model.free_floating_mass_matrix, model, data)


@pytest.mark.benchmark
def test_free_floating_jacobian_derivative(
    jaxsim_model_ergocub_reduced: js.model.JaxSimModel, benchmark
):
    model = jaxsim_model_ergocub_reduced
    data = js.data.random_model_data(model=model)

    # Warm-up call to avoid including compilation time
    js.model.generalized_free_floating_jacobian_derivative(model=model, data=data)

    benchmark(js.model.generalized_free_floating_jacobian_derivative, model, data)
