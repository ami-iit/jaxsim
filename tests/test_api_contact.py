import jax
import jax.numpy as jnp
import numpy as np
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
    assert_allclose(jnp.concatenate(W_ṗ_C), CW_vl_WC)


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

    body_indices = np.array(model.kin_dyn_parameters.contact_parameters.body)

    # Get link transforms for each collision shape
    W_H_L = data._link_transforms[body_indices]

    # Get contact point positions (shape: num_collision_shapes, 3, 3)
    W_p_C = js.contact.contact_point_positions(model=model, data=data)

    # Transform contact points from world to link frame
    # For each collision shape, transform its 3 contact points
    def transform_to_link_frame(W_H_L_i, W_p_Ci):
        """Transform 3 contact points from world to link frame."""

        L_H_W = jnp.linalg.inv(W_H_L_i)
        return jax.vmap(lambda p: (L_H_W @ jnp.hstack([p, 1.0]))[:3])(W_p_Ci)

    # Apply to all collision shapes: shape (num_collision_shapes, 3, 3)
    L_p_Ci = jax.vmap(transform_to_link_frame)(W_H_L, W_p_C)

    # =====
    # Tests
    # =====

    # Load the model in ROD.
    rod_model = rod.Sdf.load(sdf=model.built_from).model

    for shape_idx, (link_idx, points) in enumerate(
        zip(body_indices, L_p_Ci, strict=True)
    ):
        link_name = model.link_names()[link_idx]

        for j, p in enumerate(points):
            rod_model.add_frame(
                frame=rod.Frame(
                    name=f"contact_shape_{shape_idx}_{j}",
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
    num_collision_shapes = len(model.kin_dyn_parameters.contact_parameters.body)
    frame_idxs = js.frame.names_to_idxs(
        model=model_with_frames,
        frame_names=(
            f"contact_shape_{shape_idx}_{j}"
            for shape_idx in range(num_collision_shapes)
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
