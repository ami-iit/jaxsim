import functools
from typing import Sequence

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import exceptions

# =======================
# Index-related functions
# =======================


@functools.partial(jax.jit, static_argnames="joint_name")
def name_to_idx(model: js.model.JaxSimModel, *, joint_name: str) -> jtp.Int:
    """
    Convert the name of a joint to its index.

    Args:
        model: The model to consider.
        joint_name: The name of the joint.

    Returns:
        The index of the joint.
    """

    if joint_name not in model.joint_names():
        raise ValueError(f"Joint '{joint_name}' not found in the model.")

    # Note: the index of the joint for RBDAs starts from 1, but the index for
    # accessing the right element starts from 0. Therefore, there is a -1.
    return (
        jnp.array(
            model.kin_dyn_parameters.joint_model.joint_names.index(joint_name) - 1
        )
        .astype(int)
        .squeeze()
    )


def idx_to_name(model: js.model.JaxSimModel, *, joint_index: jtp.IntLike) -> str:
    """
    Convert the index of a joint to its name.

    Args:
        model: The model to consider.
        joint_index: The index of the joint.

    Returns:
        The name of the joint.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [joint_index < 0, joint_index >= model.number_of_joints()]
        ).any(),
        msg="Invalid joint index '{idx}'",
        idx=joint_index,
    )

    return model.kin_dyn_parameters.joint_model.joint_names[joint_index + 1]


@functools.partial(jax.jit, static_argnames="joint_names")
def names_to_idxs(
    model: js.model.JaxSimModel, *, joint_names: Sequence[str]
) -> jax.Array:
    """
    Convert a sequence of joint names to their corresponding indices.

    Args:
        model: The model to consider.
        joint_names: The names of the joints.

    Returns:
        The indices of the joints.
    """

    return jnp.array(
        [name_to_idx(model=model, joint_name=name) for name in joint_names],
    ).astype(int)


def idxs_to_names(
    model: js.model.JaxSimModel,
    *,
    joint_indices: Sequence[jtp.IntLike] | jtp.VectorLike,
) -> tuple[str, ...]:
    """
    Convert a sequence of joint indices to their corresponding names.

    Args:
        model: The model to consider.
        joint_indices: The indices of the joints.

    Returns:
        The names of the joints.
    """

    return tuple(idx_to_name(model=model, joint_index=idx) for idx in joint_indices)


# ============
# Joint limits
# ============


@jax.jit
def position_limit(
    model: js.model.JaxSimModel, *, joint_index: jtp.IntLike
) -> tuple[jtp.Float, jtp.Float]:
    """
    Get the position limits of a joint.

    Args:
        model: The model to consider.
        joint_index: The index of the joint.

    Returns:
        The position limits of the joint.
    """

    if model.number_of_joints() <= 1:
        return jnp.empty(0).astype(float), jnp.empty(0).astype(float)

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [joint_index < 0, joint_index >= model.number_of_joints()]
        ).any(),
        msg="Invalid joint index '{idx}'",
        idx=joint_index,
    )

    s_min = model.kin_dyn_parameters.joint_parameters.position_limits_min[joint_index]
    s_max = model.kin_dyn_parameters.joint_parameters.position_limits_max[joint_index]

    return s_min.astype(float), s_max.astype(float)


@functools.partial(jax.jit, static_argnames=["joint_names"])
def position_limits(
    model: js.model.JaxSimModel, *, joint_names: Sequence[str] | None = None
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Get the position limits of a list of joint.

    Args:
        model: The model to consider.
        joint_names: The names of the joints.

    Returns:
        The position limits of the joints.
    """

    joint_names = joint_names if joint_names is not None else model.joint_names()

    if len(joint_names) == 0:
        return jnp.empty(0).astype(float), jnp.empty(0).astype(float)

    joint_idxs = names_to_idxs(joint_names=joint_names, model=model)
    return jax.vmap(lambda i: position_limit(model=model, joint_index=i))(joint_idxs)


# ======================
# Random data generation
# ======================


@functools.partial(jax.jit, static_argnames=["joint_names"])
def random_joint_positions(
    model: js.model.JaxSimModel,
    *,
    joint_names: Sequence[str] | None = None,
    key: jax.Array | None = None,
) -> jtp.Vector:
    """
    Generate random joint positions.

    Args:
        model: The model to consider.
        joint_names: The names of the joints.
        key: The random key.

    Returns:
        The random joint positions.
    """

    key = key if key is not None else jax.random.PRNGKey(seed=0)

    s_min, s_max = position_limits(model=model, joint_names=joint_names)

    s_random = jax.random.uniform(
        minval=s_min,
        maxval=s_max,
        key=key,
        shape=s_min.shape,
    )

    return s_random
