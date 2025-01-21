import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint, Transform


def semi_implicit_euler_integration(model, data, link_forces, joint_force_references):
    """Integrate the system state using the semi-implicit Euler method."""
    # Step the dynamics forward.

    dt = model.time_step

    with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):
        W_v̇_WB, s̈ = js.ode.system_velocity_dynamics(
            model=model,
            data=data,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )

        with data.switch_velocity_representation(
            velocity_representation=jaxsim.VelRepr.Mixed
        ):
            B_H_W = Transform.inverse(data.base_transform()).at[:3, :3].set(jnp.eye(3))
            BW_X_W = Adjoint.from_transform(B_H_W)

        new_generalized_acceleration = jnp.hstack([W_v̇_WB, s̈])

        new_generalized_velocity = (
            data.generalized_velocity() + dt * new_generalized_acceleration
        )

        new_base_velocity_inertial = new_generalized_velocity[0:6]
        new_joint_velocities = new_generalized_velocity[6:]

        base_lin_velocity_inertial = new_base_velocity_inertial[0:3]

        new_base_velocity_mixed = BW_X_W @ new_generalized_velocity[0:6]
        base_lin_velocity_mixed = new_base_velocity_mixed[0:3]
        base_ang_velocity_mixed = new_base_velocity_mixed[3:6]

        base_quaternion_derivative = jaxsim.math.Quaternion.derivative(
            quaternion=data.base_orientation(),
            omega=base_ang_velocity_mixed,
            omega_in_body_fixed=False,
        ).squeeze()

        new_base_position = data.base_position() + dt * base_lin_velocity_mixed
        new_base_quaternion = data.base_orientation() + dt * base_quaternion_derivative

        base_quaternion_norm = jaxsim.math.safe_norm(new_base_quaternion)

        new_base_quaternion = new_base_quaternion / jnp.where(
            base_quaternion_norm == 0, 1.0, base_quaternion_norm
        )

        new_joint_position = data.joint_positions() + dt * new_joint_velocities

        data = data.replace(
            validate=True,
            state=data.state.replace(
                physics_model=data.state.physics_model.replace(
                    base_quaternion=new_base_quaternion,
                    base_position=new_base_position,
                    joint_positions=new_joint_position,
                    joint_velocities=new_joint_velocities,
                    base_linear_velocity=base_lin_velocity_inertial,
                    base_angular_velocity=base_ang_velocity_mixed,
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

        ki = js.ode.system_dynamics(
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
