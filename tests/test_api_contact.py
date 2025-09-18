import jax
import jax.numpy as jnp
<<<<<<< HEAD
=======
import numpy as np
import pytest
>>>>>>> 91f80b4 (Fix contact API test)
import rod

import jaxsim.api as js
from jaxsim import VelRepr

from .utils import assert_allclose


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

    # Compute the pose of the implicit contact frame associated to the collidable shapes
    # and the transforms of all links.
    W_H_C = js.contact.transforms(model=model, data=data)
 
    # Check that the origin of the implicit contact frame is located over the
    # collidable shape.
    W_p_C = js.contact.contact_point_positions(model=model, data=data)
    assert_allclose(W_p_C, W_H_C[:, :, 0:3, 3])

    # Compute the velocity of the collidable shape.
    # This quantity always matches with the linear component of the mixed 6D velocity
    # of the implicit frame associated to the collidable shape.
    W_ṗ_C = js.contact.contact_point_velocities(model=model, data=data)

    # Compute the velocity of the collidable shape using the contact Jacobian.
    ν = data.generalized_velocity
    CW_J_WC = js.contact.jacobian(model=model, data=data, output_vel_repr=VelRepr.Mixed)
    CW_vl_WC = jnp.einsum("c6g,g->c6", CW_J_WC, ν)[:, 0:3]

    # Compare the two velocities.
    assert_allclose(jnp.contatenate(W_ṗ_C), CW_vl_WC)


def test_contact_point_jacobians(
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

    # Compute the velocity of the collidable shapes with a RBDA.
    # This function always returns the linear part of the mixed velocity of the
    # implicit frame C corresponding to the collidable shape.
    W_ṗ_C = js.contact.contact_point_velocities(model=model, data=data)

    # Compute the generalized velocity and the free-floating Jacobian of the frame C.
    ν = data.generalized_velocity
    CW_J_WC = js.contact.jacobian(model=model, data=data, output_vel_repr=VelRepr.Mixed)

    # Compute the velocity of the collidable shapes using the Jacobians.
    v_WC_from_jax = jax.vmap(lambda J, ν: J @ ν, in_axes=(0, None))(CW_J_WC, ν)

    assert_allclose(jnp.concatenate(W_ṗ_C), v_WC_from_jax[:, 0:3])


def test_contact_jacobian_derivative(
    jaxsim_models_floating_base: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_floating_base

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    W_H_L = data._link_transforms
    W_p_C = js.contact.contact_point_positions(model=model, data=data)

    # Vectorize over the 3 points for one link
    transform_points = jax.vmap(
        lambda H, p: H @ jnp.hstack([p, 1.0]), in_axes=(None, 0)
    )

    # Vectorize over the links
    L_p_Ci = jax.vmap(transform_points, in_axes=(0, 0))(W_H_L, W_p_C)[..., :3]

    # =====
    # Tests
    # =====

    # Load the model in ROD.
    rod_model = rod.Sdf.load(sdf=model.built_from).model

    # Add dummy frames on the contact shapes.

    for idx, link_name, points in zip(
        np.arange(model.number_of_links()), model.link_names(), L_p_Ci, strict=True
    ):
        # points: shape (3, 3) for this link
        for j, p in enumerate(points):
            rod_model.add_frame(
                frame=rod.Frame(
                    name=f"contact_shape_{idx}_{j}",
                    attached_to=link_name,
                    pose=rod.Pose(
                        relative_to=link_name,
                        pose=jnp.zeros((6,)).at[0:3].set(p),
                    ),
                ),
            )

    # Rebuild the JaxSim model.
    model_with_frames = js.model.JaxSimModel.build_from_model_description(
        model_description=rod_model
    )
    model_with_frames = js.model.reduce(
        model=model_with_frames, considered_joints=model.joint_names()
    )

    # Rebuild the JaxSim data.
    data_with_frames = js.data.JaxSimModelData.build(
        model=model_with_frames,
        base_position=data.base_position,
        base_quaternion=data.base_orientation,
        joint_positions=data.joint_positions,
        base_linear_velocity=data.base_velocity[0:3],
        base_angular_velocity=data.base_velocity[3:6],
        joint_velocities=data.joint_velocities,
        velocity_representation=velocity_representation,
    )

    # Extract the indexes of the frames attached to the contact shapes.
    frame_idxs = js.frame.names_to_idxs(
        model=model_with_frames,
        frame_names=(
            f"contact_shape_{idx}_{j}"
            for idx in np.arange(model.number_of_links())
            for j in range(3)
        ),
    )

    # Compute the contact Jacobian derivative.
    J̇_WC = js.contact.jacobian_derivative(
        model=model_with_frames, data=data_with_frames
    )

    # Compute the contact Jacobian derivative using frames.
    J̇_WF = jax.vmap(
        js.frame.jacobian_derivative,
        in_axes=(None, None),
    )(model_with_frames, data, frame_index=frame_idxs)

    # Compare the two Jacobians.
    assert_allclose(J̇_WC, J̇_WF)
