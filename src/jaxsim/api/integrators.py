import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp


def semi_implicit_euler_integration(model, data, link_forces, joint_force_references):
    """Integrate the system state using the semi-implicit Euler method."""
    # Step the dynamics forward.

    with data.switch_velocity_representation(jaxsim.api.common.VelRepr.Inertial):
        a_b, dds, _ = js.ode.system_velocity_dynamics(
            model=model,
            data=data,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )
        generalized_acceleration = jnp.hstack(((a_b), (dds)))
        new_velocity = (
            data.generalized_velocity() + generalized_acceleration * model.time_step
        )

        base_lin_velocity = new_velocity[:3]
        base_ang_velocity = new_velocity[3:6]
        joint_velocity = new_velocity[6:]

        quat = data.base_orientation(dcm=False)
        angular_velocity_norm = jnp.linalg.norm(base_ang_velocity)
        axis_angular_velocity = base_ang_velocity / (
            angular_velocity_norm + 1e-6 * (angular_velocity_norm == 0)
        )
        angle_rotation = model.time_step * angular_velocity_norm
        delta_quat = axis_angle_to_quat(axis_angular_velocity, angle_rotation)
        new_quaternion = quat_mul(quat, delta_quat)
        new_quaternion = new_quaternion / jnp.linalg.norm(new_quaternion)

        new_joint_position = data.joint_positions() + joint_velocity * model.time_step
        new_base_position = data.base_position() + base_lin_velocity * model.time_step
        data = data.replace(
            validate=False,
            state=data.state.replace(
                physics_model=data.state.physics_model.replace(
                    base_quaternion=new_quaternion,
                    base_position=new_base_position,
                    joint_positions=new_joint_position,
                    joint_velocities=joint_velocity,
                    base_linear_velocity=base_lin_velocity,
                    base_angular_velocity=base_ang_velocity,
                )
            ),
        )
    return data


def heun2_integration(model, data, link_forces, joint_force_references):
    """Integrate the system state using the Heun's method."""
    A: jtp.Matrix = jnp.array(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=float,
    )

    b: jtp.Matrix = jnp.array([[1 / 2, 1 / 2]], dtype=float).transpose()
    c: jtp.Vector = jnp.array([0, 1], dtype=float)

    row_index_of_solution: int = 0

    # Initialize the carry of the for loop with the stacked kᵢ vectors.
    carry0 = jax.tree.map(
        lambda l: jnp.zeros((c.size, *l.shape), dtype=l.dtype), data.state
    )

    def scan_body(carry, i):
        # Compute ∑ⱼ aᵢⱼ kⱼ.
        op_sum_ak = lambda k: jnp.einsum("s,s...->...", A[i], k)
        sum_ak = jax.tree.map(op_sum_ak, carry)

        # Compute the next state for the kᵢ evaluation.
        # Note that this is not a Δt integration since aᵢⱼ could be fractional.
        op = lambda x0_leaf, k_leaf: x0_leaf + model.time_step * k_leaf
        xi = jax.tree.map(op, data.state, sum_ak)

        # Compute the next time for the kᵢ evaluation.
        # ti = c[i] * model.time_step  # TODO: Check why it is not used (wrapper dynamics)

        # Evaluate the dynamics.
        with data.editable(validate=True) as data_rw:
            data_rw.state = xi

        ki, _ = js.ode.system_dynamics(
            model,
            data,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )

        # Store the kᵢ derivative in K.
        op = lambda l_k, l_ki: l_k.at[i].set(l_ki)
        carry = jax.tree.map(op, carry, ki)

        return carry, {}

    # Compute the state derivatives kᵢ.
    K, _ = jax.lax.scan(
        f=scan_body,
        init=carry0,
        xs=jnp.arange(c.size),
    )

    # Compute the output state.
    # Note that z contains as many new states as the rows of `b.T`.
    op = lambda x0, k: x0 + model.time_step * jnp.einsum("zs,s...->z...", b.T, k)
    z = jax.tree.map(op, data.state, K)

    # The next state is the batch element located at the configured index of solution.
    next_state = jax.tree.map(lambda l: l[row_index_of_solution], z)
    return data.replace(state=next_state)


def axis_angle_to_quat(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """
    s, c = jnp.sin(angle * 0.5), jnp.cos(angle * 0.5)
    return jnp.insert(axis * s, 0, c)


def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
    """Multiplies two quaternions.

    Args:
        u: (4,) quaternion (w,x,y,z)
        v: (4,) quaternion (w,x,y,z)

    Returns:
        A quaternion u * v.
    """
    return jnp.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )
