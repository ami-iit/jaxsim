from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr

from . import common
from .common import ContactModel, ContactsParams

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class RigidContactsParams(ContactsParams):
    """Parameters of the rigid contacts model."""

    # Static friction coefficient
    mu: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    # Baumgarte proportional term
    K: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )

    # Baumgarte derivative term
    D: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )

    def __hash__(self) -> int:
        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray.hash_of_array(self.mu),
                HashedNumpyArray.hash_of_array(self.K),
                HashedNumpyArray.hash_of_array(self.D),
            )
        )

    def __eq__(self, other: RigidContactsParams) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def build(
        cls: type[Self],
        *,
        mu: jtp.FloatLike | None = None,
        K: jtp.FloatLike | None = None,
        D: jtp.FloatLike | None = None,
    ) -> Self:
        """Create a `RigidContactParams` instance."""

        return cls(
            mu=jnp.array(
                mu
                if mu is not None
                else cls.__dataclass_fields__["mu"].default_factory()
            ).astype(float),
            K=jnp.array(
                K if K is not None else cls.__dataclass_fields__["K"].default_factory()
            ).astype(float),
            D=jnp.array(
                D if D is not None else cls.__dataclass_fields__["D"].default_factory()
            ).astype(float),
        )

    def valid(self) -> jtp.BoolLike:
        """Check if the parameters are valid."""
        return bool(
            jnp.all(self.mu >= 0.0)
            and jnp.all(self.K >= 0.0)
            and jnp.all(self.D >= 0.0)
        )


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    regularization_delassus: jax_dataclasses.Static[float] = dataclasses.field(
        default=1e-6, kw_only=True
    )

    _solver_options_keys: jax_dataclasses.Static[tuple[str, ...]] = dataclasses.field(
        default=("solver_tol",), kw_only=True
    )
    _solver_options_values: jax_dataclasses.Static[tuple[Any, ...]] = dataclasses.field(
        default=(1e-3,), kw_only=True
    )

    @property
    def solver_options(self) -> dict[str, Any]:
        """Get the solver options as a dictionary."""

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
        regularization_delassus: jtp.FloatLike | None = None,
        solver_options: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a `RigidContacts` instance with specified parameters.

        Args:
            regularization_delassus:
                The regularization term to add to the diagonal of the Delassus matrix.
            solver_options: The options to pass to the QP solver.
            **kwargs: Extra arguments which are ignored.

        Returns:
            The `RigidContacts` instance.
        """

        if len(kwargs) != 0:
            logging.debug(msg=f"Ignoring extra arguments: {kwargs}")

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
            regularization_delassus=float(
                regularization_delassus
                if regularization_delassus is not None
                else cls.__dataclass_fields__["regularization_delassus"].default
            ),
            _solver_options_keys=tuple(solver_options.keys()),
            _solver_options_values=tuple(solver_options.values()),
            **kwargs,
        )

    @staticmethod
    def compute_impact_velocity(
        inactive_collidable_points: jtp.ArrayLike,
        M: jtp.MatrixLike,
        J_WC: jtp.MatrixLike,
        data: js.data.JaxSimModelData,
    ) -> jtp.Vector:
        """
        Return the new velocity of the system after a potential impact.

        Args:
            inactive_collidable_points: The activation state of the collidable points.
            M: The mass matrix of the system (in mixed representation).
            J_WC: The Jacobian matrix of the collidable points (in mixed representation).
            data: The `JaxSimModelData` instance.
        """

        def impact_velocity(
            inactive_collidable_points: jtp.ArrayLike,
            nu_pre: jtp.ArrayLike,
            M: jtp.MatrixLike,
            J_WC: jtp.MatrixLike,
            data: js.data.JaxSimModelData,
        ):
            # Compute system velocity after impact maintaining zero linear velocity of active points
            with data.switch_velocity_representation(VelRepr.Mixed):
                sl = jnp.s_[:, 0:3, :]
                Jl_WC = J_WC[sl]
                # Zero out the jacobian rows of inactive points
                Jl_WC = jnp.vstack(
                    jnp.where(
                        inactive_collidable_points[:, jnp.newaxis, jnp.newaxis],
                        jnp.zeros_like(Jl_WC),
                        Jl_WC,
                    )
                )

                A = jnp.vstack(
                    [
                        jnp.hstack([M, -Jl_WC.T]),
                        jnp.hstack(
                            [Jl_WC, jnp.zeros((Jl_WC.shape[0], Jl_WC.shape[0]))]
                        ),
                    ]
                )
                b = jnp.hstack([M @ nu_pre, jnp.zeros(Jl_WC.shape[0])])
                x = jnp.linalg.lstsq(A, b)[0]
                nu_post = x[0 : M.shape[0]]

                return nu_post

        with data.switch_velocity_representation(VelRepr.Mixed):
            BW_ν_pre_impact = data.generalized_velocity()

            BW_ν_post_impact = impact_velocity(
                data=data,
                inactive_collidable_points=inactive_collidable_points,
                nu_pre=BW_ν_pre_impact,
                M=M,
                J_WC=J_WC,
            )

        return BW_ν_post_impact

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
            A tuple containing as first element the computed contact forces.
        """

        # Import qpax privately just in this method.
        import qpax

        # Get the indices of the enabled collidable points.
        indices_of_enabled_collidable_points = (
            model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
        )

        n_collidable_points = len(indices_of_enabled_collidable_points)

        link_forces = jnp.atleast_2d(
            jnp.array(link_forces, dtype=float).squeeze()
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        )

        joint_force_references = jnp.atleast_1d(
            jnp.array(joint_force_references, dtype=float).squeeze()
            if joint_force_references is not None
            else jnp.zeros((model.number_of_joints(),))
        )

        # Compute kin-dyn quantities used in the contact model.
        with data.switch_velocity_representation(VelRepr.Mixed):
            BW_ν = data.generalized_velocity()

            M = js.model.free_floating_mass_matrix(model=model, data=data)

            J_WC = js.contact.jacobian(model=model, data=data)
            J̇_WC = js.contact.jacobian_derivative(model=model, data=data)

            W_H_C = js.contact.transforms(model=model, data=data)

        # Compute the position and linear velocities (mixed representation) of
        # all enabled collidable points belonging to the robot.
        position, velocity = js.contact.collidable_point_kinematics(
            model=model, data=data
        )

        # Compute the penetration depth and velocity of the collidable points.
        # Note that this function considers the penetration in the normal direction.
        δ, δ_dot, n̂ = jax.vmap(common.compute_penetration_data, in_axes=(0, 0, None))(
            position, velocity, model.terrain
        )

        # Build a references object to simplify converting link forces.
        references = js.references.JaxSimModelReferences.build(
            model=model,
            data=data,
            velocity_representation=data.velocity_representation,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )

        # Compute the generalized free acceleration.
        with (
            references.switch_velocity_representation(VelRepr.Mixed),
            data.switch_velocity_representation(VelRepr.Mixed),
        ):

            BW_ν̇_free = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    link_forces=references.link_forces(model=model, data=data),
                    joint_force_references=references.joint_force_references(
                        model=model
                    ),
                )
            )

        # Compute the free linear acceleration of the collidable points.
        # Since we use doubly-mixed jacobian, this corresponds to W_p̈_C.
        free_contact_acc = RigidContacts._linear_acceleration_of_collidable_points(
            BW_nu=BW_ν,
            BW_nu_dot=BW_ν̇_free,
            CW_J_WC_BW=J_WC,
            CW_J_dot_WC_BW=J̇_WC,
        ).flatten()

        # Compute stabilization term.
        baumgarte_term = RigidContacts._compute_baumgarte_stabilization_term(
            inactive_collidable_points=(δ <= 0),
            δ=δ,
            δ_dot=δ_dot,
            n=n̂,
            K=data.contacts_params.K,
            D=data.contacts_params.D,
        ).flatten()

        # Compute the Delassus matrix.
        delassus_matrix = RigidContacts._delassus_matrix(M=M, J_WC=J_WC)

        # Initialize regularization term of the Delassus matrix for
        # better numerical conditioning.
        Iε = self.regularization_delassus * jnp.eye(delassus_matrix.shape[0])

        # Construct the quadratic cost function.
        Q = delassus_matrix + Iε
        q = free_contact_acc - baumgarte_term

        # Construct the inequality constraints.
        G = RigidContacts._compute_ineq_constraint_matrix(
            inactive_collidable_points=(δ <= 0), mu=data.contacts_params.mu
        )
        h_bounds = RigidContacts._compute_ineq_bounds(
            n_collidable_points=n_collidable_points
        )

        # Construct the equality constraints.
        A = jnp.zeros((0, 3 * n_collidable_points))
        b = jnp.zeros((0,))

        # Solve the following optimization problem with qpax:
        #
        # min_{x} 0.5 x⊤ Q x + q⊤ x
        #
        # s.t. A x = b
        #      G x ≤ h
        #
        # TODO: add possibility to notify if the QP problem did not converge.
        solution, _, _, _, converged, _ = qpax.solve_qp(  # noqa: F841
            Q=Q, q=q, A=A, b=b, G=G, h=h_bounds, **self.solver_options
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
    def _delassus_matrix(
        M: jtp.MatrixLike,
        J_WC: jtp.MatrixLike,
    ) -> jtp.Matrix:

        sl = jnp.s_[:, 0:3, :]
        J_WC_lin = jnp.vstack(J_WC[sl])

        delassus_matrix = J_WC_lin @ jnp.linalg.pinv(M) @ J_WC_lin.T
        return delassus_matrix

    @staticmethod
    def _compute_ineq_constraint_matrix(
        inactive_collidable_points: jtp.Vector, mu: jtp.FloatLike
    ) -> jtp.Matrix:
        """
        Compute the inequality constraint matrix for a single collidable point.

        Rows 0-3: enforce the friction pyramid constraint,
        Row 4: last one is for the non negativity of the vertical force
        Row 5: contact complementarity condition
        """
        G_single_point = jnp.array(
            [
                [1, 0, -mu],
                [0, 1, -mu],
                [-1, 0, -mu],
                [0, -1, -mu],
                [0, 0, -1],
                [0, 0, 0],
            ]
        )
        G = jnp.tile(G_single_point, (len(inactive_collidable_points), 1, 1))
        G = G.at[:, 5, 2].set(inactive_collidable_points)

        G = jax.scipy.linalg.block_diag(*G)
        return G

    @staticmethod
    def _compute_ineq_bounds(n_collidable_points: int) -> jtp.Vector:

        n_constraints = 6 * n_collidable_points
        return jnp.zeros(shape=(n_constraints,))

    @staticmethod
    def _linear_acceleration_of_collidable_points(
        BW_nu: jtp.ArrayLike,
        BW_nu_dot: jtp.ArrayLike,
        CW_J_WC_BW: jtp.MatrixLike,
        CW_J_dot_WC_BW: jtp.MatrixLike,
    ) -> jtp.Matrix:

        BW_ν = BW_nu
        BW_ν̇ = BW_nu_dot
        CW_J̇_WC_BW = CW_J_dot_WC_BW

        # Compute the linear acceleration of the collidable points.
        # Since we use doubly-mixed jacobians, this corresponds to W_p̈_C.
        CW_a_WC = jnp.vstack(CW_J̇_WC_BW) @ BW_ν + jnp.vstack(CW_J_WC_BW) @ BW_ν̇

        CW_a_WC = CW_a_WC.reshape(-1, 6)
        return CW_a_WC[:, 0:3].squeeze()

    @staticmethod
    def _compute_baumgarte_stabilization_term(
        inactive_collidable_points: jtp.ArrayLike,
        δ: jtp.ArrayLike,
        δ_dot: jtp.ArrayLike,
        n: jtp.ArrayLike,
        K: jtp.FloatLike,
        D: jtp.FloatLike,
    ) -> jtp.Array:

        return jnp.where(
            inactive_collidable_points[:, jnp.newaxis],
            jnp.zeros_like(n),
            (K * δ + D * δ_dot)[:, jnp.newaxis] * n,
        )
