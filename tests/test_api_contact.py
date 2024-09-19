import jax
import jax.numpy as jnp
import pytest
import rod

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

    # =====
    # Tests
    # =====

    # Extract the parent link names and the poses of the contact points.
    parent_link_names = js.link.idxs_to_names(
        model=model, link_indices=model.kin_dyn_parameters.contact_parameters.body
    )
    W_p_Ci = model.kin_dyn_parameters.contact_parameters.point

    # Load the model in ROD.
    rod_model = rod.Sdf.load(sdf=model.built_from).model

    # Add dummy frames on the contact points.
    for idx, (link_name, W_p_C) in enumerate(
        zip(parent_link_names, W_p_Ci, strict=True)
    ):
        rod_model.add_frame(
            frame=rod.Frame(
                name=f"contact_point_{idx}",
                attached_to=link_name,
                pose=rod.Pose(
                    relative_to=link_name, pose=jnp.zeros(shape=(6,)).at[0:3].set(W_p_C)
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
        base_position=data.base_position(),
        base_quaternion=data.base_orientation(dcm=False),
        joint_positions=data.joint_positions(),
        base_linear_velocity=data.base_velocity()[0:3],
        base_angular_velocity=data.base_velocity()[3:6],
        joint_velocities=data.joint_velocities(),
        velocity_representation=velocity_representation,
    )

    # Extract the indexes of the frames attached to the contact points.
    frame_idxs = js.frame.names_to_idxs(
        model=model_with_frames,
        frame_names=(
            f"contact_point_{idx}" for idx in list(range(len(parent_link_names)))
        ),
    )

    # Check that the number of frames is correct.
    assert len(frame_idxs) == len(parent_link_names)

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
    assert J̇_WC == pytest.approx(J̇_WF)
