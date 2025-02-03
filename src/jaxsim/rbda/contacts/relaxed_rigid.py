from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses
import optax

import jaxsim.api as js
import jaxsim.rbda.contacts
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr
from jaxsim.terrain.terrain import Terrain

from . import common

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class RelaxedRigidContactsParams(common.ContactsParams):
    """Parameters of the relaxed rigid contacts model."""

    # Time constant
    time_constant: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.01, dtype=float)
    )

    # Adimensional damping coefficient
    damping_coefficient: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )

    # Minimum impedance
    d_min: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.9, dtype=float)
    )

    # Maximum impedance
    d_max: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.95, dtype=float)
    )

    # Width
    width: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.0001, dtype=float)
    )

    # Midpoint
    midpoint: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )

    # Power exponent
    power: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )

    # Stiffness
    stiffness: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )

    # Damping
    damping: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )

    # Friction coefficient
    mu: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    def __hash__(self) -> int:
        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray(self.time_constant),
                HashedNumpyArray(self.damping_coefficient),
                HashedNumpyArray(self.d_min),
                HashedNumpyArray(self.d_max),
                HashedNumpyArray(self.width),
                HashedNumpyArray(self.midpoint),
                HashedNumpyArray(self.power),
                HashedNumpyArray(self.stiffness),
                HashedNumpyArray(self.damping),
                HashedNumpyArray(self.mu),
            )
        )

    def __eq__(self, other: RelaxedRigidContactsParams) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def build(
        cls: type[Self],
        *,
        time_constant: jtp.FloatLike | None = None,
        damping_coefficient: jtp.FloatLike | None = None,
        d_min: jtp.FloatLike | None = None,
        d_max: jtp.FloatLike | None = None,
        width: jtp.FloatLike | None = None,
        midpoint: jtp.FloatLike | None = None,
        power: jtp.FloatLike | None = None,
        stiffness: jtp.FloatLike | None = None,
        damping: jtp.FloatLike | None = None,
        mu: jtp.FloatLike | None = None,
    ) -> Self:
        """Create a `RelaxedRigidContactsParams` instance."""

        def default(name: str):
            return cls.__dataclass_fields__[name].default_factory()

        return cls(
            time_constant=jnp.array(
                (
                    time_constant
                    if time_constant is not None
                    else default("time_constant")
                ),
                dtype=float,
            ),
            damping_coefficient=jnp.array(
                (
                    damping_coefficient
                    if damping_coefficient is not None
                    else default("damping_coefficient")
                ),
                dtype=float,
            ),
            d_min=jnp.array(
                d_min if d_min is not None else default("d_min"), dtype=float
            ),
            d_max=jnp.array(
                d_max if d_max is not None else default("d_max"), dtype=float
            ),
            width=jnp.array(
                width if width is not None else default("width"), dtype=float
            ),
            midpoint=jnp.array(
                midpoint if midpoint is not None else default("midpoint"), dtype=float
            ),
            power=jnp.array(
                power if power is not None else default("power"), dtype=float
            ),
            stiffness=jnp.array(
                stiffness if stiffness is not None else default("stiffness"),
                dtype=float,
            ),
            damping=jnp.array(
                damping if damping is not None else default("damping"), dtype=float
            ),
            mu=jnp.array(mu if mu is not None else default("mu"), dtype=float),
        )

    def valid(self) -> jtp.BoolLike:
        """Check if the parameters are valid."""

        return bool(
            jnp.all(self.time_constant >= 0.0)
            and jnp.all(self.damping_coefficient > 0.0)
            and jnp.all(self.d_min >= 0.0)
            and jnp.all(self.d_max <= 1.0)
            and jnp.all(self.d_min <= self.d_max)
            and jnp.all(self.width >= 0.0)
            and jnp.all(self.midpoint >= 0.0)
            and jnp.all(self.power >= 0.0)
            and jnp.all(self.mu >= 0.0)
        )


@jax_dataclasses.pytree_dataclass
class RelaxedRigidContacts(common.ContactModel):
    """Relaxed rigid contacts model."""

    _solver_options_keys: jax_dataclasses.Static[tuple[str, ...]] = dataclasses.field(
        default=("tol", "maxiter", "memory_size"), kw_only=True
    )
    _solver_options_values: jax_dataclasses.Static[tuple[Any, ...]] = dataclasses.field(
        default=(1e-6, 50, 10), kw_only=True
    )

    @property
    def solver_options(self) -> dict[str, Any]:
        """Get the solver options."""

        return dict(
            zip(
                self._solver_options_keys,
                self._solver_options_values,
                strict=True,
            )
        )

    @classmethod
    def build(
        cls: type[Self],
        solver_options: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a `RelaxedRigidContacts` instance with specified parameters.

        Args:
            solver_options: The options to pass to the L-BFGS solver.
            **kwargs: The parameters of the relaxed rigid contacts model.

        Returns:
            The `RelaxedRigidContacts` instance.
        """

        # Get the default solver options.
        default_solver_options = dict(
            zip(cls._solver_options_keys, cls._solver_options_values, strict=True)
        )

        # Create the solver options to set by combining the default solver options
        # with the user-provided solver options.
        solver_options = default_solver_options | (
            solver_options if solver_options is not None else {}
        )

        # Make sure that the solver options are hashable.
        # We need to check this because the solver options are static.
        try:
            hash(tuple(solver_options.values()))
        except TypeError as exc:
            raise ValueError(
                "The values of the solver options must be hashable."
            ) from exc

        return cls(
            _solver_options_keys=tuple(solver_options.keys()),
            _solver_options_values=tuple(solver_options.values()),
            **kwargs,
        )

    @jax.jit
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the contact forces.

        Args:
            model: The model to consider.
            data: The data of the considered model.
            link_forces:
                Optional `(n_links, 6)` matrix of external forces acting on the links,
                expressed in the same representation of data.
            joint_force_references:
                Optional `(n_joints,)` vector of joint forces.

        Returns:
            A tuple containing as first element the computed contact forces in inertial representation.
        """

        link_forces = jnp.atleast_2d(
            jnp.array(link_forces, dtype=float).squeeze()
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        )

        joint_force_references = jnp.atleast_1d(
            jnp.array(joint_force_references, dtype=float).squeeze()
            if joint_force_references is not None
            else jnp.zeros(model.number_of_joints())
        )

        references = js.references.JaxSimModelReferences.build(
            model=model,
            data=data,
            velocity_representation=data.velocity_representation,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )

        # Compute the position and linear velocities (mixed representation) of
        # all collidable points belonging to the robot.
        position, velocity = js.contact.collidable_point_kinematics(
            model=model, data=data
        )

        # Compute the penetration depth and velocity of the collidable points.
        # Note that this function considers the penetration in the normal direction.
        δ, _, n̂ = jax.vmap(common.compute_penetration_data, in_axes=(0, 0, None))(
            position, velocity, model.terrain
        )

        # Compute the position in the constraint frame.
        position_constraint = jax.vmap(lambda δ, n̂: -δ * n̂)(δ, n̂)

        # Compute the transforms of the implicit frames corresponding to the
        # collidable points.
        W_H_C = js.contact.transforms(model=model, data=data)

        with (
            data.switch_velocity_representation(VelRepr.Mixed),
            references.switch_velocity_representation(VelRepr.Mixed),
        ):
            BW_ν = data.generalized_velocity

            BW_ν̇_free = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    link_forces=references.link_forces(model=model, data=data),
                    joint_torques=references.joint_force_references(model=model),
                )
            )

            M = js.model.free_floating_mass_matrix(model=model, data=data)

            Jl_WC = jnp.vstack(
                jax.vmap(lambda J, δ: J * (δ > 0))(
                    js.contact.jacobian(model=model, data=data)[:, :3, :], δ
                )
            )

            J̇_WC = jnp.vstack(
                jax.vmap(lambda J̇, δ: J̇ * (δ > 0))(
                    js.contact.jacobian_derivative(model=model, data=data)[:, :3], δ
                ),
            )

        # Compute the regularization terms.
        a_ref, R, *_ = self._regularizers(
            model=model,
            position_constraint=position_constraint,
            velocity_constraint=velocity,
            parameters=model.contacts_params,
        )

        # Compute the Delassus matrix and the free mixed linear acceleration of
        # the collidable points.
        G = Jl_WC @ jnp.linalg.pinv(M) @ Jl_WC.T
        CW_al_free_WC = Jl_WC @ BW_ν̇_free + J̇_WC @ BW_ν

        # Calculate quantities for the linear optimization problem.
        A = G + R
        b = CW_al_free_WC - a_ref

        # Create the objective function to minimize as a lambda computing the cost
        # from the optimized variables x.
        objective = lambda x, A, b: jnp.sum(jnp.square(A @ x + b))

        # ========================================
        # Helper function to run the L-BFGS solver
        # ========================================

        def run_optimization(
            init_params: jtp.Vector,
            fun: Callable,
            opt: optax.GradientTransformationExtraArgs,
            maxiter: int,
            tol: float,
        ) -> tuple[jtp.Vector, optax.OptState]:

            # Get the function to compute the loss and the gradient w.r.t. its inputs.
            value_and_grad_fn = optax.value_and_grad_from_state(fun)

            # Initialize the carry of the following loop.
            OptimizationCarry = tuple[jtp.Vector, optax.OptState]
            init_carry: OptimizationCarry = (init_params, opt.init(params=init_params))

            def step(carry: OptimizationCarry) -> OptimizationCarry:

                params, state = carry

                value, grad = value_and_grad_fn(
                    params,
                    state=state,
                    A=A,
                    b=b,
                )

                updates, state = opt.update(
                    updates=grad,
                    state=state,
                    params=params,
                    value=value,
                    grad=grad,
                    value_fn=fun,
                    A=A,
                    b=b,
                )

                params = optax.apply_updates(params, updates)

                return params, state

            # TODO: maybe fix the number of iterations and switch to scan?
            def continuing_criterion(carry: OptimizationCarry) -> jtp.Bool:

                _, state = carry

                iter_num = optax.tree_utils.tree_get(state, "count")
                grad = optax.tree_utils.tree_get(state, "grad")
                err = optax.tree_utils.tree_l2_norm(grad)

                return (iter_num == 0) | ((iter_num < maxiter) & (err >= tol))

            final_params, final_state = jax.lax.while_loop(
                continuing_criterion, step, init_carry
            )

            return final_params, final_state

        # ======================================
        # Compute the contact forces with L-BFGS
        # ======================================

        # Initialize the optimized forces with a linear Hunt/Crossley model.
        init_params = jax.vmap(
            lambda p, v: self._hunt_crossley_contact_model(
                position=p,
                velocity=v,
                terrain=model.terrain,
                K=1e6,
                D=2e3,
                p=0.5,
                q=0.5,
                # No tangential initial forces.
                mu=0.0,
                tangential_deformation=jnp.zeros(3),
            )[0]
        )(position, velocity).flatten()

        # Get the solver options.
        solver_options = self.solver_options

        # Extract the options corresponding to the convergence criteria.
        # All the remaining options are passed to the solver.
        tol = solver_options.pop("tol")
        maxiter = solver_options.pop("maxiter")

        # Compute the 3D linear force in C[W] frame.
        solution, _ = run_optimization(
            init_params=init_params,
            fun=objective,
            opt=optax.lbfgs(**solver_options),
            tol=tol,
            maxiter=maxiter,
        )

        # Reshape the optimized solution to be a matrix of 3D contact forces.
        CW_fl_C = solution.reshape(-1, 3)

        # Convert the contact forces from mixed to inertial-fixed representation.
        W_f_C = jax.vmap(
            lambda CW_fl_C, W_H_C: (
                ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                    array=jnp.zeros(6).at[0:3].set(CW_fl_C),
                    transform=W_H_C,
                    other_representation=VelRepr.Mixed,
                    is_force=True,
                )
            ),
        )(CW_fl_C, W_H_C)

        return W_f_C, {}

    @staticmethod
    def _regularizers(
        model: js.model.JaxSimModel,
        position_constraint: jtp.Vector,
        velocity_constraint: jtp.Vector,
        parameters: RelaxedRigidContactsParams,
    ) -> tuple:
        """
        Compute the contact jacobian and the reference acceleration.

        Args:
            model: The jaxsim model.
            position_constraint: The position of the collidable points in the constraint frame.
            velocity_constraint: The velocity of the collidable points in the constraint frame.
            parameters: The parameters of the relaxed rigid contacts model.

        Returns:
            A tuple containing the reference acceleration, the regularization matrix,
            the stiffness, and the damping.
        """

        # Extract the parameters of the contact model.
        Ω, ζ, ξ_min, ξ_max, width, mid, p, K, D, μ = (
            getattr(parameters, field)
            for field in (
                "time_constant",
                "damping_coefficient",
                "d_min",
                "d_max",
                "width",
                "midpoint",
                "power",
                "stiffness",
                "damping",
                "mu",
            )
        )

        # Get the indices of the enabled collidable points.
        indices_of_enabled_collidable_points = (
            model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
        )

        parent_link_idx_of_enabled_collidable_points = jnp.array(
            model.kin_dyn_parameters.contact_parameters.body, dtype=int
        )[indices_of_enabled_collidable_points]

        # Compute the 6D inertia matrices of all links.
        M_L = js.model.link_spatial_inertia_matrices(model=model)

        def imp_aref(
            pos: jtp.Vector,
            vel: jtp.Vector,
        ) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector, jtp.Vector]:
            """
            Calculate impedance and offset acceleration in constraint frame.

            Args:
                pos: position in constraint frame.
                vel: velocity in constraint frame.

            Returns:
                ξ: computed impedance
                a_ref: offset acceleration in constraint frame
                K: computed stiffness
                D: computed damping
            """

            imp_x = jnp.abs(pos) / width

            imp_a = (1.0 / jnp.power(mid, p - 1)) * jnp.power(imp_x, p)
            imp_b = 1 - (1.0 / jnp.power(1 - mid, p - 1)) * jnp.power(1 - imp_x, p)
            imp_y = jnp.where(imp_x < mid, imp_a, imp_b)

            # Compute the impedance.
            ξ = ξ_min + imp_y * (ξ_max - ξ_min)
            ξ = jnp.clip(ξ, ξ_min, ξ_max)
            ξ = jnp.where(imp_x > 1.0, ξ_max, ξ)

            # Compute the spring and damper parameters during runtime from the
            # impedance and other contact parameters.
            K = 1 / (ξ_max * Ω * ζ) ** 2
            D = 2 / (ξ_max * Ω)

            # If the user specifies K and D and they are negative, the computed `a_ref`
            # becomes something more similar to a classic Baumgarte regularization.
            K = jnp.where(K < 0, -K / ξ_max**2, K)
            D = jnp.where(D < 0, -D / ξ_max, D)

            # Compute the reference acceleration.
            a_ref = -(D * vel + K * ξ * pos)

            return ξ, a_ref, K, D

        def compute_row(
            *,
            link_idx: jtp.Int,
            pos: jtp.Vector,
            vel: jtp.Vector,
        ) -> tuple[jtp.Vector, jtp.Matrix, jtp.Vector, jtp.Vector]:

            # Compute the reference acceleration.
            ξ, a_ref, K, D = imp_aref(pos=pos, vel=vel)

            # Compute the regularization term.
            R = (
                (2 * μ**2 * (1 - ξ) / (ξ + 1e-12))
                * (1 + μ**2)
                @ jnp.linalg.inv(M_L[link_idx, :3, :3])
            )

            # Return the computed values, setting them to zero in case of no contact.
            is_active = (pos.dot(pos) > 0).astype(float)
            return jax.tree.map(
                lambda x: jnp.atleast_1d(x) * is_active, (a_ref, R, K, D)
            )

        a_ref, R, K, D = jax.tree.map(
            f=jnp.concatenate,
            tree=(
                *jax.vmap(compute_row)(
                    link_idx=parent_link_idx_of_enabled_collidable_points,
                    pos=position_constraint,
                    vel=velocity_constraint,
                ),
            ),
        )

        return a_ref, jnp.diag(R), K, D

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def _hunt_crossley_contact_model(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        terrain: Terrain,
        K: jtp.FloatLike,
        D: jtp.FloatLike,
        mu: jtp.FloatLike,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact force using the Hunt/Crossley model.

        Args:
            position: The position of the collidable point.
            velocity: The velocity of the collidable point.
            tangential_deformation: The material deformation of the collidable point.
            terrain: The terrain model.
            K: The stiffness parameter.
            D: The damping parameter of the soft contacts model.
            mu: The static friction coefficient.
            p:
                The exponent p corresponding to the damping-related non-linearity
                of the Hunt/Crossley model.
            q:
                The exponent q corresponding to the spring-related non-linearity
                of the Hunt/Crossley model

        Returns:
            A tuple containing the computed contact force and the derivative of the
            material deformation.
        """

        # Convert the input vectors to arrays.
        W_p_C = jnp.array(position, dtype=float).squeeze()
        W_ṗ_C = jnp.array(velocity, dtype=float).squeeze()
        m = jnp.array(tangential_deformation, dtype=float).squeeze()

        # Use symbol for the static friction.
        μ = mu

        # Compute the penetration depth, its rate, and the considered terrain normal.
        δ, δ̇, n̂ = common.compute_penetration_data(p=W_p_C, v=W_ṗ_C, terrain=terrain)

        # There are few operations like computing the norm of a vector with zero length
        # or computing the square root of zero that are problematic in an AD context.
        # To avoid these issues, we introduce a small tolerance ε to their arguments
        # and make sure that we do not check them against zero directly.
        ε = jnp.finfo(float).eps

        # Compute the powers of the penetration depth.
        # Inject ε to address AD issues in differentiating the square root when
        #  p and q are fractional.
        δp = jnp.power(δ + ε, p)
        δq = jnp.power(δ + ε, q)

        # ========================
        # Compute the normal force
        # ========================

        # Non-linear spring-damper model (Hunt/Crossley model).
        # This is the force magnitude along the direction normal to the terrain.
        force_normal_mag = (K * δp) * δ + (D * δq) * δ̇

        # Depending on the magnitude of δ̇, the normal force could be negative.
        force_normal_mag = jnp.maximum(0.0, force_normal_mag)

        # Compute the 3D linear force in C[W] frame.
        f_normal = force_normal_mag * n̂

        # ============================
        # Compute the tangential force
        # ============================

        # Extract the tangential component of the velocity.
        v_tangential = W_ṗ_C - jnp.dot(W_ṗ_C, n̂) * n̂

        # Extract the normal and tangential components of the material deformation.
        m_normal = jnp.dot(m, n̂) * n̂
        m_tangential = m - jnp.dot(m, n̂) * n̂

        # Compute the tangential force in the sticking case.
        # Using the tangential component of the material deformation should not be
        # necessary if the sticking-slipping transition occurs in a terrain area
        # with a locally constant normal. However, this assumption is not true in
        # general, especially for highly uneven terrains.
        f_tangential = -((K * δp) * m_tangential + (D * δq) * v_tangential)

        # Detect the contact type (sticking or slipping).
        # Note that if there is no contact, sticking is set to True, and this detail
        # is exploited in the computation of the `contact_status` variable.
        sticking = jnp.logical_or(
            δ <= 0, f_tangential.dot(f_tangential) <= (μ * force_normal_mag) ** 2
        )

        # Compute the direction of the tangential force.
        # To prevent dividing by zero, we use a switch statement.
        norm = jaxsim.math.safe_norm(f_tangential)
        f_tangential_direction = f_tangential / (
            norm + jnp.finfo(float).eps * (norm == 0)
        )

        # Project the tangential force to the friction cone if slipping.
        f_tangential = jnp.where(
            sticking,
            f_tangential,
            jnp.minimum(μ * force_normal_mag, norm) * f_tangential_direction,
        )

        # Set the tangential force to zero if there is no contact.
        f_tangential = jnp.where(δ <= 0, jnp.zeros(3), f_tangential)

        # =====================================
        # Compute the material deformation rate
        # =====================================

        # Compute the derivative of the material deformation.
        # Note that we included an additional relaxation of `m_normal` in the
        # sticking case, so that the normal deformation that could have accumulated
        # from a previous slipping phase can relax to zero.
        ṁ_no_contact = -(K / D) * m
        ṁ_sticking = v_tangential - (K / D) * m_normal
        ṁ_slipping = -(f_tangential + (K * δp) * m_tangential) / (D * δq)

        # Compute the contact status:
        # 0: slipping
        # 1: sticking
        # 2: no contact
        contact_status = sticking.astype(int)
        contact_status += (δ <= 0).astype(int)

        # Select the right material deformation rate depending on the contact status.
        ṁ = jax.lax.select_n(contact_status, ṁ_slipping, ṁ_sticking, ṁ_no_contact)

        # ==========================================
        # Compute and return the final contact force
        # ==========================================

        # Sum the normal and tangential forces.
        CW_fl = f_normal + f_tangential

        return CW_fl, ṁ
