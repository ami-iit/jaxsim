from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr
from jaxsim.rbda import contacts


@jax.jit
@js.common.named_scope
def link_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_force_references: jtp.VectorLike | None = None,
    **kwargs,
) -> jtp.Matrix:
    """
    Compute the 6D contact forces of all links of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D external forces to apply to the links expressed in the same
            representation of data.
        joint_force_references:
            The joint force references to apply to the joints.
        kwargs: Additional keyword arguments to pass to the active contact model..

    Returns:
        A `(nL, 6)` array containing the stacked 6D contact forces of the links,
        expressed in the frame corresponding to the active representation.
    """

    # Build link forces if not provided.
    # These forces are expressed in the frame corresponding to the velocity
    # representation of data.
    O_f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # Build joint force references if not provided.
    joint_force_references = (
        jnp.atleast_1d(joint_force_references)
        if joint_force_references is not None
        else jnp.zeros(model.dofs())
    )

    # We expect that the 6D forces included in the `link_forces` argument are expressed
    # in the frame corresponding to the velocity representation of `data`.
    input_references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        velocity_representation=data.velocity_representation,
        link_forces=O_f_L,
        joint_force_references=joint_force_references,
    )

    # Compute the 6D forces applied to the links equivalent to the forces applied
    # to the frames associated to the collidable points.
    f_L, _ = compute_link_contact_forces(
        model=model,
        data=data,
        link_forces=input_references.link_forces(model=model, data=data),
        joint_force_references=input_references.joint_force_references(),
        **kwargs,
    )

    return f_L


@staticmethod
def compute_link_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    **kwargs,
) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
    """
    Compute the link contact forces.

    Args:
        model: The robot model considered by the contact model.
        data: The data of the considered model.
        **kwargs: Optional additional arguments, specific to the contact model.

    Returns:
        A tuple containing as first element the 6D contact force applied to the
        links and expressed in the frame of the velocity representation of data,
        and as second element a dictionary of optional additional information.
    """

    # Compute the contact forces expressed in the inertial frame.
    # This function, contrarily to `compute_contact_forces`, already handles how
    # the optional kwargs should be passed to the specific contact models.
    W_f_C, aux_dict = js.contact.collidable_point_dynamics(
        model=model, data=data, **kwargs
    )

    # Compute the 6D forces applied to the links equivalent to the forces applied
    # to the frames associated to the collidable points.
    with data.switch_velocity_representation(VelRepr.Inertial):

        W_f_L = link_forces_from_contact_forces(
            model=model, data=data, contact_forces=W_f_C
        )

    # Store the link forces in the references object for easy conversion.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        link_forces=W_f_L,
        velocity_representation=VelRepr.Inertial,
    )

    # Convert the link forces to the frame corresponding to the velocity
    # representation of data.
    with references.switch_velocity_representation(data.velocity_representation):
        f_L = references.link_forces(model=model, data=data)

    return f_L, aux_dict


@staticmethod
def link_forces_from_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_forces: jtp.MatrixLike,
) -> jtp.Matrix:
    """
    Compute the link forces from the contact forces.

    Args:
        model: The robot model considered by the contact model.
        data: The data of the considered model.
        contact_forces: The contact forces computed by the contact model.

    Returns:
        The 6D contact forces applied to the links and expressed in the frame of
        the velocity representation of data.
    """

    # Get the object storing the contact parameters of the model.
    contact_parameters = model.kin_dyn_parameters.contact_parameters

    # Extract the indices corresponding to the enabled collidable points.
    indices_of_enabled_collidable_points = (
        contact_parameters.indices_of_enabled_collidable_points
    )

    # Convert the contact forces to a JAX array.
    f_C = jnp.atleast_2d(jnp.array(contact_forces, dtype=float).squeeze())

    # Get the pose of the enabled collidable points.
    W_H_C = js.contact.transforms(model=model, data=data)[
        indices_of_enabled_collidable_points
    ]

    # Convert the contact forces to inertial-fixed representation.
    W_f_C = jax.vmap(
        lambda f_C, W_H_C: (
            ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=f_C,
                other_representation=data.velocity_representation,
                transform=W_H_C,
                is_force=True,
            )
        )
    )(f_C, W_H_C)

    # Construct the vector defining the parent link index of each collidable point.
    # We use this vector to sum the 6D forces of all collidable points rigidly
    # attached to the same link.
    parent_link_index_of_collidable_points = jnp.array(
        contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    # Create the mask that associate each collidable point to their parent link.
    # We use this mask to sum the collidable points to the right link.
    mask = parent_link_index_of_collidable_points[:, jnp.newaxis] == jnp.arange(
        model.number_of_links()
    )

    # Sum the forces of all collidable points rigidly attached to a body.
    # Since the contact forces W_f_C are expressed in the world frame,
    # we don't need any coordinate transformation.
    W_f_L = mask.T @ W_f_C

    # Compute the link transforms.
    W_H_L = (
        js.model.forward_kinematics(model=model, data=data)
        if data.velocity_representation is not VelRepr.Inertial
        else jnp.zeros(shape=(model.number_of_links(), 4, 4))
    )

    # Convert the inertial-fixed link forces to the velocity representation of data.
    f_L = jax.vmap(
        lambda W_f_L, W_H_L: (
            ModelDataWithVelocityRepresentation.inertial_to_other_representation(
                array=W_f_L,
                other_representation=data.velocity_representation,
                transform=W_H_L,
                is_force=True,
            )
        )
    )(W_f_L, W_H_L)

    return f_L


@jax.jit
@js.common.named_scope
def collidable_point_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    link_forces: jtp.MatrixLike | None = None,
    joint_force_references: jtp.VectorLike | None = None,
    **kwargs,
) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
    r"""
    Compute the 6D force applied to each enabled collidable point.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D external forces to apply to the links expressed in the same
            representation of data.
        joint_force_references:
            The joint force references to apply to the joints.
        kwargs: Additional keyword arguments to pass to the active contact model.

    Returns:
        The 6D force applied to each enabled collidable point and additional data based
        on the contact model configured:
        - Soft: the material deformation rate.
        - Rigid: no additional data.
        - QuasiRigid: no additional data.

    Note:
        The material deformation rate is always returned in the mixed frame`
        `C[W] = ({}^W \mathbf{p}_C, [W])`. This is convenient for integration purpose.
        Instead, the 6D forces are returned in the active representation.
    """

    # Build the common kw arguments to pass to the computation of the contact forces.
    common_kwargs = dict(
        link_forces=link_forces,
        joint_force_references=joint_force_references,
    )

    # Build the additional kwargs to pass to the computation of the contact forces.
    match model.contact_model:

        case contacts.RelaxedRigidContacts():

            kwargs_contact_model = common_kwargs | kwargs

        case _:
            raise ValueError(f"Invalid contact model: {model.contact_model}")

    # Compute the contact forces with the active contact model.
    W_f_C, aux_data = model.contact_model.compute_contact_forces(
        model=model,
        data=data,
        **kwargs_contact_model,
    )

    # Compute the transforms of the implicit frames `C[L] = (W_p_C, [L])`
    # associated to the enabled collidable point.
    # In inertial-fixed representation, the computation of these transforms
    # is not necessary and the conversion below becomes a no-op.

    # Get the indices of the enabled collidable points.
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    W_H_C = (
        js.contact.transforms(model=model, data=data)
        if data.velocity_representation is not VelRepr.Inertial
        else jnp.stack([jnp.eye(4)] * len(indices_of_enabled_collidable_points))
    )

    # Convert the 6D forces to the active representation.
    f_Ci = jax.vmap(
        lambda W_f_C, W_H_C: data.inertial_to_other_representation(
            array=W_f_C,
            other_representation=data.velocity_representation,
            transform=W_H_C,
            is_force=True,
        )
    )(W_f_C, W_H_C)

    return f_Ci, aux_data
