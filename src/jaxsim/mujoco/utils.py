import mujoco as mj
import numpy as np

from . import MujocoModelHelper


def mujoco_data_from_jaxsim(
    mujoco_model: mj.MjModel,
    jaxsim_model,
    jaxsim_data,
    mujoco_data: mj.MjData | None = None,
    update_removed_joints: bool = True,
) -> mj.MjData:
    """
    Create a Mujoco data object from a JaxSim model and data objects.

    Args:
        mujoco_model: The Mujoco model object corresponding to the JaxSim model.
        jaxsim_model: The JaxSim model object from which the Mujoco model was created.
        jaxsim_data: The JaxSim data object containing the state of the model.
        mujoco_data: An optional Mujoco data object. If None, a new one will be created.
        update_removed_joints:
            If True, the positions of the joints that have been removed during the
            model reduction process will be set to their initial values.

    Returns:
        The Mujoco data object containing the state of the JaxSim model.

    Note:
        This method is useful to initialize a Mujoco data object used for visualization
        with the state of a JaxSim model. In particular, this function takes care of
        initializing the positions of the joints that have been removed during the
        model reduction process. After the initial creation of the Mujoco data object,
        it's faster to update the state using an external MujocoModelHelper object.
    """

    # The package `jaxsim.mujoco` is supposed to be jax-independent.
    # We import all the JaxSim resources privately.
    import jaxsim.api as js

    if not isinstance(jaxsim_model, js.model.JaxSimModel):
        raise ValueError("The `jaxsim_model` argument must be a JaxSimModel object.")

    if not isinstance(jaxsim_data, js.data.JaxSimModelData):
        raise ValueError("The `jaxsim_data` argument must be a JaxSimModelData object.")

    # Create the helper to operate on the Mujoco model and data.
    model_helper = MujocoModelHelper(model=mujoco_model, data=mujoco_data)

    # If the model is fixed-base, the Mujoco model won't have the joint corresponding
    # to the floating base, and the helper would raise an exception.
    if jaxsim_model.floating_base():

        # Set the model position.
        model_helper.set_base_position(position=np.array(jaxsim_data.base_position()))

        # Set the model orientation.
        model_helper.set_base_orientation(
            orientation=np.array(jaxsim_data.base_orientation())
        )

    # Set the joint positions.
    if jaxsim_model.dofs() > 0:

        model_helper.set_joint_positions(
            joint_names=list(jaxsim_model.joint_names()),
            positions=np.array(
                jaxsim_data.joint_positions(
                    model=jaxsim_model, joint_names=jaxsim_model.joint_names()
                )
            ),
        )

    # Updating these joints is not necessary after the first time.
    # Users can disable this update after initialization.
    if update_removed_joints:

        # Create a dictionary with the joints that have been removed for various reasons
        # (like link lumping due to model reduction).
        joints_removed_dict = {
            j.name: j
            for j in jaxsim_model.description._joints_removed
            if j.name not in set(jaxsim_model.joint_names())
        }

        # Set the positions of the removed joints.
        _ = [
            model_helper.set_joint_position(
                position=joints_removed_dict[joint_name].initial_position,
                joint_name=joint_name,
            )
            # Select all original joint that have been removed from the JaxSim model
            # that are still present in the Mujoco model.
            for joint_name in joints_removed_dict
            if joint_name in model_helper.joint_names()
        ]

    # Return the mujoco data with updated kinematics.
    mj.mj_forward(mujoco_model, model_helper.data)

    return model_helper.data
