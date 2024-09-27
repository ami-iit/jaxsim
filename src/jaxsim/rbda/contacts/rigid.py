from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr
from jaxsim.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams, ContactsState

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
        """Create a `RigidContactParams` instance"""

        return cls(
            mu=mu or cls.__dataclass_fields__["mu"].default,
            K=K or cls.__dataclass_fields__["K"].default,
            D=D or cls.__dataclass_fields__["D"].default,
        )

    def valid(self) -> jtp.BoolLike:

        return bool(
            jnp.all(self.mu >= 0.0)
            and jnp.all(self.K >= 0.0)
            and jnp.all(self.D >= 0.0)
        )


@jax_dataclasses.pytree_dataclass
class RigidContactsState(ContactsState):
    """Class storing the state of the rigid contacts model."""

    def __eq__(self, other: RigidContactsState) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def build(cls: type[Self]) -> Self:
        """Create a `RigidContactsState` instance"""

        return cls()

    @classmethod
    def zero(cls: type[Self], **kwargs) -> Self:
        """Build a zero `RigidContactsState` instance from a `JaxSimModel`."""

        return cls.build()

    def valid(self, **kwargs) -> jtp.BoolLike:
        return True


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    parameters: RigidContactsParams = dataclasses.field(
        default_factory=RigidContactsParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    @staticmethod
    def detect_contacts(
        W_p_C: jtp.ArrayLike,
        terrain_height: jtp.ArrayLike,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Detect contacts between the collidable points and the terrain.

        Args:
            W_p_C: The position of the collidable points.
            terrain_height: The height of the terrain at the collidable point position.

        Returns:
            A tuple containing the activation state of the collidable points
            and the contact penetration depth h.
        """

        # TODO: reduce code duplication with js.contact.in_contact
        def detect_contact(
            W_p_C: jtp.ArrayLike,
            terrain_height: jtp.FloatLike,
        ) -> tuple[jtp.Bool, jtp.Float]:
            """
            Detect contacts between the collidable points and the terrain.
            """

            # Unpack the position of the collidable point.
            _, _, pz = W_p_C.squeeze()

            inactive = pz > terrain_height

            # Compute contact penetration depth
            h = jnp.maximum(0.0, terrain_height - pz)

            return inactive, h

        inactive_collidable_points, h = jax.vmap(detect_contact)(W_p_C, terrain_height)

        return inactive_collidable_points, h

    @staticmethod
    def compute_impact_velocity(
        inactive_collidable_points: jtp.ArrayLike,
        M: jtp.MatrixLike,
        J_WC: jtp.MatrixLike,
        data: js.data.JaxSimModelData,
    ) -> jtp.Vector:
        """Returns the new velocity of the system after a potential impact.

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
        regularization_term: jtp.FloatLike = 1e-6,
        solver_tol: jtp.FloatLike = 1e-3,
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
            regularization_term:
                The regularization term to add to the diagonal of the Delassus
                matrix for better numerical conditioning.
            solver_tol: The convergence tolerance to consider in the QP solver.

        Returns:
            A tuple containing the contact forces.
        """

        # Initialize the model and data this contact model is operating on.
        # This will raise an exception if either the contact model or the
        # contact parameters are not compatible.
        model, data = self.initialize_model_and_data(model=model, data=data)

        # Import qpax just in this method
        import qpax

        link_forces = (
            link_forces
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        )

        joint_force_references = (
            joint_force_references
            if joint_force_references is not None
            else jnp.zeros((model.number_of_joints(),))
        )

        # Compute kin-dyn quantities used in the contact model
        with data.switch_velocity_representation(VelRepr.Mixed):
            M = js.model.free_floating_mass_matrix(model=model, data=data)
            J_WC = js.contact.jacobian(model=model, data=data)
            W_H_C = js.contact.transforms(model=model, data=data)
            J̇_WC_BW = js.contact.jacobian_derivative(model=model, data=data)
            BW_ν = data.generalized_velocity()

        # Compute the position and linear velocities (mixed representation) of
        # all collidable points belonging to the robot.
        position, velocity = js.contact.collidable_point_kinematics(
            model=model, data=data
        )

        terrain_height = jax.vmap(self.terrain.height)(position[:, 0], position[:, 1])
        n_collidable_points = model.kin_dyn_parameters.contact_parameters.point.shape[0]

        # Compute the activation state of the collidable points
        inactive_collidable_points, h = RigidContacts.detect_contacts(
            W_p_C=position,
            terrain_height=terrain_height,
        )

        delassus_matrix = RigidContacts._delassus_matrix(M=M, J_WC=J_WC)

        # Add regularization for better numerical conditioning
        delassus_matrix = delassus_matrix + regularization_term * jnp.eye(
            delassus_matrix.shape[0]
        )

        references = js.references.JaxSimModelReferences.build(
            model=model,
            data=data,
            velocity_representation=data.velocity_representation,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
        )

        with (
            references.switch_velocity_representation(VelRepr.Mixed),
            data.switch_velocity_representation(VelRepr.Mixed),
        ):
            BW_ν̇_free = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    joint_forces=references.joint_force_references(model=model),
                    link_forces=references.link_forces(model=model, data=data),
                )
            )

        free_contact_acc = RigidContacts._linear_acceleration_of_collidable_points(
            BW_nu=BW_ν,
            BW_nu_dot=BW_ν̇_free,
            CW_J_WC_BW=J_WC,
            CW_J_dot_WC_BW=J̇_WC_BW,
        ).flatten()

        # Compute stabilization term
        ḣ = velocity[:, 2].squeeze()
        baumgarte_term = RigidContacts._compute_baumgarte_stabilization_term(
            inactive_collidable_points=inactive_collidable_points,
            h=h,
            ḣ=ḣ,
            K=self.parameters.K,
            D=self.parameters.D,
        ).flatten()

        free_contact_acc -= baumgarte_term

        # Setup optimization problem
        Q = delassus_matrix
        q = free_contact_acc
        G = RigidContacts._compute_ineq_constraint_matrix(
            inactive_collidable_points=inactive_collidable_points, mu=self.parameters.mu
        )
        h_bounds = RigidContacts._compute_ineq_bounds(
            n_collidable_points=n_collidable_points
        )
        A = jnp.zeros((0, 3 * n_collidable_points))
        b = jnp.zeros((0,))

        # Solve the optimization problem
        solution, *_ = qpax.solve_qp(
            Q=Q, q=q, A=A, b=b, G=G, h=h_bounds, solver_tol=solver_tol
        )

        f_C_lin = solution.reshape(-1, 3)

        # Transform linear contact forces to 6D
        CW_f_C = jnp.hstack(
            (
                f_C_lin,
                jnp.zeros((f_C_lin.shape[0], 3)),
            )
        )

        # Transform the contact forces to inertial-fixed representation
        W_f_C = jax.vmap(
            lambda CW_f_C, W_H_C: ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=CW_f_C,
                transform=W_H_C,
                other_representation=VelRepr.Mixed,
                is_force=True,
            ),
        )(
            CW_f_C,
            W_H_C,
        )

        return W_f_C, ()

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
        def compute_G_single_point(mu: float, c: float) -> jtp.Matrix:
            """
            Compute the inequality constraint matrix for a single collidable point
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
                    [0, 0, c],
                ]
            )
            return G_single_point

        G = jax.vmap(compute_G_single_point, in_axes=(None, 0))(
            mu, inactive_collidable_points
        )
        G = jax.scipy.linalg.block_diag(*G)
        return G

    @staticmethod
    def _compute_ineq_bounds(n_collidable_points: jtp.FloatLike) -> jtp.Vector:
        n_constraints = 6 * n_collidable_points
        return jnp.zeros(shape=(n_constraints,))

    @staticmethod
    def _linear_acceleration_of_collidable_points(
        BW_nu: jtp.ArrayLike,
        BW_nu_dot: jtp.ArrayLike,
        CW_J_WC_BW: jtp.MatrixLike,
        CW_J_dot_WC_BW: jtp.MatrixLike,
    ) -> jtp.Matrix:
        CW_J̇_WC_BW = CW_J_dot_WC_BW
        BW_ν = BW_nu
        BW_ν̇ = BW_nu_dot

        CW_a_WC = jnp.vstack(CW_J̇_WC_BW) @ BW_ν + jnp.vstack(CW_J_WC_BW) @ BW_ν̇
        CW_a_WC = CW_a_WC.reshape(-1, 6)

        return CW_a_WC[:, 0:3].squeeze()

    @staticmethod
    def _compute_baumgarte_stabilization_term(
        inactive_collidable_points: jtp.ArrayLike,
        h: jtp.ArrayLike,
        ḣ: jtp.ArrayLike,
        K: jtp.FloatLike,
        D: jtp.FloatLike,
    ) -> jtp.Array:
        def baumgarte_stabilization(
            inactive: jtp.BoolLike,
            h: jtp.FloatLike,
            ḣ: jtp.FloatLike,
            k_baumgarte: jtp.FloatLike,
            d_baumgarte: jtp.FloatLike,
        ) -> jtp.Array:
            baumgarte_term = jax.lax.cond(
                inactive,
                lambda h, ḣ, K, D: jnp.zeros(shape=(3,)),
                lambda h, ḣ, K, D: jnp.zeros(shape=(3,)).at[2].set(K * h + D * ḣ),
                *(
                    h,
                    ḣ,
                    k_baumgarte,
                    d_baumgarte,
                ),
            )
            return baumgarte_term

        baumgarte_term = jax.vmap(
            baumgarte_stabilization, in_axes=(0, 0, 0, None, None)
        )(inactive_collidable_points, h, ḣ, K, D)

        return baumgarte_term
