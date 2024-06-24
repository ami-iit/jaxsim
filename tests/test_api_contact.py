import jax
import jax.numpy as jnp
import pytest

import jaxsim.api as js
from jaxsim import VelRepr


def test_contact_kinematics(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    # =====
    # Tests
    # =====

    # Compute the pose of the implicit contact frame associated to the collidable points
    # and the transforms of all links.
    W_H_C = js.contact.transforms(model=model, data=data)
    W_H_L = js.model.forward_kinematics(model=model, data=data)

    # Check that the orientation of the implicit contact frame matches with the
    # orientation of the link to which the contact point is attached.
    for contact_idx, index_of_parent_link in enumerate(
        model.kin_dyn_parameters.contact_parameters.body
    ):
        assert W_H_C[contact_idx, 0:3, 0:3] == pytest.approx(
            W_H_L[index_of_parent_link][0:3, 0:3]
        )

    # Check that the origin of the implicit contact frame is located over the
    # collidable point.
    W_p_C = js.contact.collidable_point_positions(model=model, data=data)
    assert W_p_C == pytest.approx(W_H_C[:, 0:3, 3])

    # Compute the velocity of the collidable point.
    # This quantity always matches with the linear component of the mixed 6D velocity
    # of the implicit frame associated to the collidable point.
    W_ṗ_C = js.contact.collidable_point_velocities(model=model, data=data)

    # Compute the velocity of the collidable point using the contact Jacobian.
    ν = data.generalized_velocity()
    CW_J_WC = js.contact.jacobian(model=model, data=data, output_vel_repr=VelRepr.Mixed)
    CW_vl_WC = jnp.einsum("c6g,g->c6", CW_J_WC, ν)[:, 0:3]

    # Compare the two velocities.
    assert W_ṗ_C == pytest.approx(CW_vl_WC)
