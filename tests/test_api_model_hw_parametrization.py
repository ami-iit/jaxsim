import jax
import jax.numpy as jnp
import pytest
import rod

import jaxsim.api as js
from jaxsim.api.kin_dyn_parameters import HwLinkMetadata, ScalingFactors


def test_update_hw_link_parameters(jaxsim_model_garpez: js.model.JaxSimModel):
    """
    Test that the hardware parameters of the model are updated correctly.
    """

    model = jaxsim_model_garpez

    # Store initial hardware parameters
    initial_metadata = model.kin_dyn_parameters.hw_link_metadata

    # Create the scaling factors
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [2.0, 1.5, 1.0],  # Scale x, y, z for link1
                [1.2, 1.0, 1.0],  # Scale r for link2
                [1.5, 0.8, 1.0],  # Scale r, l for link3
                [1.5, 1.0, 0.8],  # Scale x, y, z for link4
            ]
        ),
        density=jnp.ones(4),
    )

    # Update the model using the scaling factors
    updated_model = js.model.update_hw_parameters(model, scaling_parameters)

    # Compare updated hardware parameters
    for link_idx, link_name in enumerate(model.link_names()):
        updated_metadata = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx],
            updated_model.kin_dyn_parameters.hw_link_metadata,
        )
        initial_metadata_link = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx], initial_metadata
        )

        # TODO: Compute the 3D scaling vector
        # scale_vector = HwLinkMetadata._convert_scaling_to_3d_vector(
        #     initial_metadata_link.shape, scaling_parameters.dims[link_idx]
        # )

        # Compare shape dimensions
        assert jnp.allclose(
            updated_metadata.dims,
            initial_metadata_link.dims * scaling_parameters.dims[link_idx],
            atol=1e-6,
        ), f"Mismatch in dimensions for link {link_name}: expected {initial_metadata_link.dims * scaling_parameters.dims[link_idx]}, got {updated_metadata.dims}"


@pytest.mark.parametrize(
    "jaxsim_model_garpez_scaled",
    [
        {
            "link1_scale": 4.0,
            "link2_scale": 3.0,
            "link3_scale": 2.0,
            "link4_scale": 1.5,
        }
    ],
    indirect=True,
)
def test_model_scaling_against_rod(
    jaxsim_model_garpez: js.model.JaxSimModel,
    jaxsim_model_garpez_scaled: js.model.JaxSimModel,
):
    """
    Test that scaling the HW parameters of JaxSim model matches the kin/dyn quantities of a JaxSim model obtained from a pre-scaled rod model.
    """

    # Define scaling parameters
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [4.0, 1.0, 1.0],  # Scale only x-dimension for link1
                [3.0, 1.0, 1.0],  # Scale only r-dimension for link2
                [1.0, 2.0, 1.0],  # Scale l dimension for link3
                [1.5, 1.0, 1.0],  # Scale only x-dimension for link4
            ]
        ),
        density=jnp.ones(4),
    )

    # Apply scaling to the original JaxSim model
    updated_model = js.model.update_hw_parameters(
        jaxsim_model_garpez, scaling_parameters
    )

    # Compare hardware parameters of the scaled JaxSim model with the pre-scaled JaxSim model
    for link_idx, link_name in enumerate(jaxsim_model_garpez.link_names()):
        scaled_metadata = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx],
            updated_model.kin_dyn_parameters.hw_link_metadata,
        )
        pre_scaled_metadata = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx],
            jaxsim_model_garpez_scaled.kin_dyn_parameters.hw_link_metadata,
        )

        # Compare shape dimensions
        assert jnp.allclose(scaled_metadata.dims, pre_scaled_metadata.dims, atol=1e-6)

        # Compare mass
        scaled_mass, _ = HwLinkMetadata.compute_mass_and_inertia(scaled_metadata)
        pre_scaled_mass, _ = HwLinkMetadata.compute_mass_and_inertia(
            pre_scaled_metadata
        )
        assert scaled_mass == pytest.approx(pre_scaled_mass, abs=1e-6)

        # Compare inertia tensors
        _, scaled_inertia = HwLinkMetadata.compute_mass_and_inertia(scaled_metadata)
        _, pre_scaled_inertia = HwLinkMetadata.compute_mass_and_inertia(
            pre_scaled_metadata
        )
        assert jnp.allclose(scaled_inertia, pre_scaled_inertia, atol=1e-6)

        # Compare transformations
        assert jnp.allclose(scaled_metadata.L_H_G, pre_scaled_metadata.L_H_G, atol=1e-6)
        assert jnp.allclose(
            scaled_metadata.L_H_vis, pre_scaled_metadata.L_H_vis, atol=1e-6
        )


def test_update_hw_parameters_vmap(
    jaxsim_model_garpez: js.model.JaxSimModel,
):
    """
    Test that the hardware parameters of the model are updated correctly using vmap
    to create a set of n updated models.
    """

    model_nominal = jaxsim_model_garpez
    dofs = model_nominal.dofs()

    # Define a set of scaling factors for n models
    n = 10  # Number of updated models to create
    scaling_factors = [
        ScalingFactors(
            dims=(scale * jnp.ones((model_nominal.number_of_links(), 3))),
            density=(scale * jnp.ones(model_nominal.number_of_links())),
        )
        for scale in jnp.linspace(2.0, 2.0 + n - 1, n)
    ]

    # Convert the list of ScalingFactors to a JAX array of pytrees
    scaling_factors = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scaling_factors)

    # Generate a batch of updated models using vmap
    updated_models = jax.vmap(js.model.update_hw_parameters, in_axes=(None, 0))(
        model_nominal,
        scaling_factors,
    )

    def validate_model(updated_model):
        assert updated_model is not None

        # Compute forward kinematics for the "link3" link
        H_link3 = js.link.transform(
            model=updated_model,
            data=js.data.JaxSimModelData.build(model=updated_model),
            link_index=js.link.name_to_idx(model=updated_model, link_name="link3"),
        )

        # Compute the mass matrix
        M = js.model.free_floating_mass_matrix(
            model=updated_model,
            data=js.data.JaxSimModelData.build(model=updated_model),
        )

        assert H_link3 is not None
        assert H_link3.shape == (4, 4)
        assert M is not None
        assert isinstance(M, jnp.ndarray)
        assert M.shape == (6 + dofs, 6 + dofs)

    # Use vmap to validate all updated models
    jax.vmap(validate_model)(updated_models)


@pytest.mark.parametrize(
    "jaxsim_model_garpez_scaled",
    [
        {
            "link1_scale": 4.0,
            "link2_scale": 3.0,
            "link3_scale": 2.0,
            "link4_scale": 1.5,
        }
    ],
    indirect=True,
)
def test_export_updated_model(
    jaxsim_model_garpez: js.model.JaxSimModel,
    jaxsim_model_garpez_scaled: js.model.JaxSimModel,
):
    """
    Test the export of an updated model using JaxSimModel.export_updated_model.
    """

    model: js.model.JaxSimModel = jaxsim_model_garpez

    # Define scaling parameters
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [4.0, 1.0, 1.0],  # Scale x-dimension for link1
                [3.0, 1.0, 1.0],  # Scale r-dimension for link2
                [1.0, 2.0, 1.0],  # Scale l-dimension for link3
                [1.5, 1.0, 1.0],  # Scale x-dimension for link4
            ]
        ),
        density=jnp.ones(4),
    )

    # Update the model with the scaling parameters
    updated_model: js.model.JaxSimModel = js.model.update_hw_parameters(
        model, scaling_parameters
    )

    # Export the updated model
    exported_model_urdf = updated_model.export_updated_model()
    assert isinstance(exported_model_urdf, str), "Exported model URDF is not a string."

    # Convert the URDF string to a ROD model
    exported_model_sdf = rod.Sdf.load(exported_model_urdf, is_urdf=True)
    assert isinstance(
        exported_model_sdf, rod.Sdf
    ), "Failed to load exported model as ROD Sdf."
    assert (
        len(exported_model_sdf.models()) == 1
    ), "Exported ROD model does not contain exactly one model."
    exported_model_rod = exported_model_sdf.models()[0]

    # Get the pre-scaled ROD model
    pre_scaled_model_rod = rod.Sdf.load(jaxsim_model_garpez_scaled.built_from).models()[
        0
    ]
    assert isinstance(
        pre_scaled_model_rod, rod.Model
    ), "Failed to load pre-scaled model as ROD Model."

    # Validate that the exported model matches the pre-scaled model
    for link_idx, link_name in enumerate(model.link_names()):
        try:
            exported_link = next(
                link for link in exported_model_rod.links() if link.name == link_name
            )
        except StopIteration:
            raise ValueError(
                f"Link '{link_name}' not found in exported model. "
                f"Available links: {[link.name for link in exported_model_rod.links()]}"
            ) from None

        pre_scaled_link = next(
            link for link in pre_scaled_model_rod.links() if link.name == link_name
        )

        # Compare shape dimensions
        exported_geometry = exported_link.visual.geometry.geometry()
        pre_scaled_geometry = pre_scaled_link.visual.geometry.geometry()

        # Ensure both geometries have the same attributes for comparison
        exported_values = jnp.array(
            [
                getattr(exported_geometry, attr, 0)
                for attr in vars(exported_geometry)
                if hasattr(pre_scaled_geometry, attr)
            ]
        )
        pre_scaled_values = jnp.array(
            [
                getattr(pre_scaled_geometry, attr, 0)
                for attr in vars(pre_scaled_geometry)
                if hasattr(exported_geometry, attr)
            ]
        )

        assert jnp.allclose(exported_values, pre_scaled_values, atol=1e-6), (
            f"Mismatch in geometry dimensions for link {link_name}: "
            f"expected {pre_scaled_values}, got {exported_values}"
        )

        # Compare mass
        assert exported_link.inertial.mass == pytest.approx(
            pre_scaled_link.inertial.mass, abs=1e-4
        ), (
            f"Mismatch in mass for link {link_name}: "
            f"expected {pre_scaled_link.inertial.mass}, got {exported_link.inertial.mass}"
        )

        # Compare inertia tensors
        assert jnp.allclose(
            exported_link.inertial.inertia.matrix(),
            pre_scaled_link.inertial.inertia.matrix(),
            atol=1e-4,
        ), (
            f"Mismatch in inertia tensor for link {link_name}: "
            f"expected {pre_scaled_link.inertial.inertia.matrix()}, "
            f"got {exported_link.inertial.inertia.matrix()}"
        )


def test_hw_parameters_optimization(jaxsim_model_garpez: js.model.JaxSimModel):
    """
    Test that updating hardware parameters allows optimizing the position of a link
    to match a target value along a specific world axis.
    """

    model = jaxsim_model_garpez
    data = js.data.JaxSimModelData.build(model=model)

    # Define the target height for the link.
    target_height = 3.0

    # Get the index of the link to optimize (e.g., "torso").
    link_idx = js.link.name_to_idx(model, link_name="link4")

    # Define the initial hardware parameters (scaling factors).
    initial_dims = jnp.ones(
        (
            model.number_of_links(),
            3,
        )
    )  # Initial dimensions (1.0 for all links).
    initial_density = jnp.ones(
        (model.number_of_links(),)
    )  # Initial density (1.0 for all links).
    scaling_factors = js.kin_dyn_parameters.ScalingFactors(
        dims=initial_dims, density=initial_density
    )

    # Define the loss function.
    def loss(scaling_factors):
        # Update the model with the new hardware parameters.
        updated_model = js.model.update_hw_parameters(
            model=model, scaling_factors=scaling_factors
        )

        # Compute forward kinematics for the link.
        W_H_L = js.model.forward_kinematics(model=updated_model, data=data)[link_idx]

        # Extract the height (z-axis position) of the link.
        link4_height = W_H_L[2, 3]  # Assuming z-axis is the third row.

        # Compute the loss as the squared difference from the target height.
        return (link4_height - target_height) ** 2

    # Compute the gradient of the loss function with respect to the scaling factors.
    loss_grad = jax.grad(loss)

    # Perform gradient descent.
    alpha = 0.01  # Learning rate.
    num_iterations = 1000  # Number of gradient descent steps.
    for _ in range(num_iterations):
        # Compute the gradient.
        grad_scaling_factors = loss_grad(scaling_factors)

        # Update the scaling factors.
        scaling_factors = js.kin_dyn_parameters.ScalingFactors(
            dims=scaling_factors.dims - alpha * grad_scaling_factors.dims,
            density=scaling_factors.density - alpha * grad_scaling_factors.density,
        )

        # Compute the current loss value.
        current_loss = loss(scaling_factors)

        # Optionally, print the progress.
        if _ % 100 == 0:
            print(f"Iteration {_}: Loss = {current_loss}")

    # Assert that the final loss is close to zero.
    assert current_loss < 1e-3, "Optimization did not converge to the target height."
