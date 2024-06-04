import jax
import pytest

import jaxsim.api as js
from jaxsim import VelRepr


def test_collidable_point_jacobians(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    # =====
    # Tests
    # =====

    # Compute the velocity of the collidable points with a RBDA.
    # This function always returns the linear part of the mixed velocity of the
    # implicit frame C corresponding to the collidable point.
    W_ṗ_C = js.contact.collidable_point_velocities(model=model, data=data)

    # Compute the generalized velocity and the free-floating Jacobian of the frame C.
    ν = data.generalized_velocity()
    CW_J_WC = js.contact.jacobian(model=model, data=data, output_vel_repr=VelRepr.Mixed)

    # Compute the velocity of the collidable points using the Jacobians.
    v_WC_from_jax = jax.vmap(lambda J, ν: J @ ν, in_axes=(0, None))(CW_J_WC, ν)

    assert W_ṗ_C == pytest.approx(v_WC_from_jax[:, 0:3])
