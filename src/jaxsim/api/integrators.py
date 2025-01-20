import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp


def semi_implicit_euler_integration(model, data, link_forces, joint_force_references):
    """Integrate the system state using the semi-implicit Euler method."""
    a_b, dds, _ = js.ode.system_velocity_dynamics(
        model=model, data=data, link_forces=link_forces, joint_force_references=joint_force_references
    )
    generalized_acceleration = jnp.hstack(((a_b), (dds)))
    new_velocity = (
        data.generalized_velocity() + generalized_acceleration * model.time_step
    )
    base_lin_velocity = new_velocity[:3]
    base_ang_velocity = new_velocity[3:6]
    joint_velocity = new_velocity[6:]
    new_joint_position = data.joint_positions() + joint_velocity * model.time_step
    new_base_position = data.base_position() + base_lin_velocity * model.time_step
    new_quaternion = jaxsim.math.Quaternion.integration(
        data.base_orientation(dcm=False), base_ang_velocity, model.time_step
    )
    new_position = jnp.hstack(
        (new_base_position, new_quaternion, new_joint_position)
    )
    return data.replace(
        validate=False,
        state=data.state.replace(
            physics_model=data.state.physics_model.replace(
                base_quaternion=new_position[3:7],
                base_position=new_position[0:3],
                joint_positions=new_position[7:],
                joint_velocities=new_velocity[6:],
                base_linear_velocity=new_velocity[0:3],
                base_angular_velocity=new_velocity[3:6],
            )
        ),
    )


def heun2_integration(model, data, link_forces, joint_force_references):
    """Integrate the system state using the Heun's method."""
    A: jtp.Matrix = jnp.array([
        [0, 0],
        [1, 0],
    ], dtype=float)

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
        ki, _ = js.ode.system_dynamics(model, data)

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
