from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses
import optax

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import VelRepr
from jaxsim.math import Adjoint
from jaxsim.terrain.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams, ContactsState

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class RelaxedRigidContactsParams(ContactsParams):
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

    # Maximum number of iterations
    max_iterations: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array(50, dtype=int)
    )

    # Solver tolerance
    tolerance: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(1e-6, dtype=float)
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
                HashedNumpyArray(self.max_iterations),
                HashedNumpyArray(self.tolerance),
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
        max_iterations: jtp.IntLike | None = None,
        tolerance: jtp.FloatLike | None = None,
    ) -> Self:
        """Create a `RelaxedRigidContactsParams` instance"""

        return cls(
            **{
                field: jnp.array(locals().get(field, default), dtype=default.dtype)
                for field, default in map(
                    lambda f: (f, cls.__dataclass_fields__[f].default),
                    filter(lambda f: f != "__mutability__", cls.__dataclass_fields__),
                )
            }
        )

    def valid(self) -> jtp.BoolLike:

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
            and jnp.all(self.max_iterations > 0)
            and jnp.all(self.tolerance > 0.0)
        )


@jax_dataclasses.pytree_dataclass
class RelaxedRigidContactsState(ContactsState):
    """Class storing the state of the relaxed rigid contacts model."""

    def __eq__(self, other: RelaxedRigidContactsState) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def build(cls: type[Self]) -> Self:
        """Create a `RelaxedRigidContactsState` instance"""

        return cls()

    @classmethod
    def zero(cls: type[Self], **kwargs) -> Self:
        """Build a zero `RelaxedRigidContactsState` instance from a `JaxSimModel`."""

        return cls.build()

    def valid(self, **kwargs) -> jtp.BoolLike:
        return True


@jax_dataclasses.pytree_dataclass
class RelaxedRigidContacts(ContactModel):
    """Relaxed rigid contacts model."""

    parameters: RelaxedRigidContactsParams = dataclasses.field(
        default_factory=RelaxedRigidContactsParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    @jax.jit
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
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
            A tuple containing the contact forces.
        """

        # Initialize the model and data this contact model is operating on.
        # This will raise an exception if either the contact model or the
        # contact parameters are not compatible.
        model, data = self.initialize_model_and_data(model=model, data=data)

        link_forces = (
            link_forces
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        )

        joint_force_references = (
            joint_force_references
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

        def _detect_contact(x: jtp.Array, y: jtp.Array, z: jtp.Array) -> jtp.Array:
            x, y, z = jax.tree.map(jnp.squeeze, (x, y, z))

            n̂ = self.terrain.normal(x=x, y=y).squeeze()
            h = jnp.array([0, 0, z - model.terrain.height(x=x, y=y)])

            return jnp.dot(h, n̂)

        # Compute the position and linear velocities (mixed representation) of
        # all collidable points belonging to the robot.
        position, velocity = js.contact.collidable_point_kinematics(
            model=model, data=data
        )

        # Compute the activation state of the collidable points
        δ = jax.vmap(_detect_contact)(*position.T)

        with (
            references.switch_velocity_representation(VelRepr.Mixed),
            data.switch_velocity_representation(VelRepr.Mixed),
        ):
            M = js.model.free_floating_mass_matrix(model=model, data=data)
            Jl_WC = jnp.vstack(
                jax.vmap(lambda J, height: J * (height < 0))(
                    js.contact.jacobian(model=model, data=data)[:, :3, :], δ
                )
            )
            W_H_C = js.contact.transforms(model=model, data=data)
            BW_ν̇_free = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    link_forces=references.link_forces(model=model, data=data),
                    joint_forces=references.joint_force_references(model=model),
                )
            )
            BW_ν = data.generalized_velocity()
            J̇_WC = jnp.vstack(
                jax.vmap(lambda J̇, height: J̇ * (height < 0))(
                    js.contact.jacobian_derivative(model=model, data=data)[:, :3], δ
                ),
            )

            a_ref, R, K, D = self._regularizers(
                model=model,
                penetration=δ,
                velocity=velocity,
                parameters=self.parameters,
            )

        G = Jl_WC @ jnp.linalg.lstsq(M, Jl_WC.T)[0]
        CW_al_free_WC = Jl_WC @ BW_ν̇_free + J̇_WC @ BW_ν

        # Calculate quantities for the linear optimization problem.
        A = G + R
        b = CW_al_free_WC - a_ref

        objective = lambda x, A, b: jnp.sum(jnp.square(A @ x + b))

        def run_optimization(
            init_params: jtp.Array,
            fun: Callable,
            opt: optax.GradientTransformation,
            maxiter: jtp.Int,
            tol: jtp.Float,
            **kwargs,
        ):
            value_and_grad_fn = optax.value_and_grad_from_state(fun)

            def step(carry):
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

            def continuing_criterion(carry):
                _, state = carry
                iter_num = optax.tree_utils.tree_get(state, "count")
                grad = optax.tree_utils.tree_get(state, "grad")
                err = optax.tree_utils.tree_l2_norm(grad)
                return (iter_num == 0) | ((iter_num < maxiter) & (err >= tol))

            init_carry = (init_params, opt.init(init_params))
            final_params, final_state = jax.lax.while_loop(
                continuing_criterion, step, init_carry
            )
            return final_params, final_state

        init_params = (
            K[:, jnp.newaxis] * jnp.zeros_like(position).at[:, 2].set(δ)
            + D[:, jnp.newaxis] * velocity
        ).flatten()

        # Compute the 3D linear force in C[W] frame
        CW_f_Ci, _ = run_optimization(
            init_params=init_params,
            A=A,
            b=b,
            maxiter=self.parameters.max_iterations,
            opt=optax.lbfgs(
                memory_size=10,
            ),
            fun=objective,
            tol=self.parameters.tolerance,
        )

        CW_f_Ci = CW_f_Ci.reshape((-1, 3))

        def mixed_to_inertial(W_H_C: jax.Array, CW_fl: jax.Array) -> jax.Array:
            W_Xf_CW = Adjoint.from_transform(
                W_H_C.at[0:3, 0:3].set(jnp.eye(3)),
                inverse=True,
            ).T
            return W_Xf_CW @ jnp.hstack([CW_fl, jnp.zeros(3)])

        W_f_C = jax.vmap(mixed_to_inertial)(W_H_C, CW_f_Ci)

        return W_f_C, (None,)

    @staticmethod
    def _regularizers(
        model: js.model.JaxSimModel,
        penetration: jtp.Array,
        velocity: jtp.Array,
        parameters: RelaxedRigidContactsParams,
    ) -> tuple:
        """
        Compute the contact jacobian and the reference acceleration.

        Args:
            model: The jaxsim model.
            penetration: The penetration of the collidable points.
            velocity: The velocity of the collidable points.
            parameters: The parameters of the relaxed rigid contacts model.

        Returns:
            A tuple containing the reference acceleration, the regularization matrix, the stiffness, and the damping.
        """

        Ω, ζ, ξ_min, ξ_max, width, mid, p, K, D, μ, *_ = jax_dataclasses.astuple(
            parameters
        )

        def _imp_aref(
            penetration: jtp.Array,
            velocity: jtp.Array,
        ) -> tuple[jtp.Array, jtp.Array]:
            """
            Calculates impedance and offset acceleration in constraint frame.

            Args:
                penetration: penetration in constraint frame
                velocity: velocity in constraint frame

            Returns:
                a_ref: offset acceleration in constraint frame
                R: regularization matrix
                K: computed stiffness
                D: computed damping
            """
            position = jnp.zeros(shape=(3,)).at[2].set(penetration)

            imp_x = jnp.abs(position) / width
            imp_a = (1.0 / jnp.power(mid, p - 1)) * jnp.power(imp_x, p)

            imp_b = 1 - (1.0 / jnp.power(1 - mid, p - 1)) * jnp.power(1 - imp_x, p)

            imp_y = jnp.where(imp_x < mid, imp_a, imp_b)

            imp = jnp.clip(ξ_min + imp_y * (ξ_max - ξ_min), ξ_min, ξ_max)
            imp = jnp.atleast_1d(jnp.where(imp_x > 1.0, ξ_max, imp))

            # When passing negative values, K and D represent a spring and damper, respectively.
            K_f = jnp.where(K < 0, -K / ξ_max**2, 1 / (ξ_max * Ω * ζ) ** 2)
            D_f = jnp.where(D < 0, -D / ξ_max, 2 / (ξ_max * Ω))

            a_ref = -jnp.atleast_1d(D_f * velocity + K_f * imp * position)

            return imp, a_ref, jnp.atleast_1d(K_f), jnp.atleast_1d(D_f)

        def _compute_row(
            *,
            link_idx: jtp.Float,
            penetration: jtp.Array,
            velocity: jtp.Array,
        ) -> tuple[jtp.Array, jtp.Array]:

            # Compute the reference acceleration.
            ξ, a_ref, K, D = _imp_aref(
                penetration=penetration,
                velocity=velocity,
            )

            # Compute the regularization terms.
            R = (
                (2 * μ**2 * (1 - ξ) / (ξ + 1e-12))
                * (1 + μ**2)
                @ jnp.linalg.inv(M_L[link_idx, :3, :3])
            )

            return jax.tree.map(lambda x: x * (penetration < 0), (a_ref, R, K, D))

        M_L = js.model.link_spatial_inertia_matrices(model=model)

        a_ref, R, K, D = jax.tree.map(
            jnp.concatenate,
            (
                *jax.vmap(_compute_row)(
                    link_idx=jnp.array(
                        model.kin_dyn_parameters.contact_parameters.body
                    ),
                    penetration=penetration,
                    velocity=velocity,
                ),
            ),
        )
        return a_ref, jnp.diag(R), K, D
