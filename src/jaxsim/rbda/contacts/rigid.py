from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses
from jax.numpy.linalg import pinv
from jax.scipy.linalg import block_diag

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import math
from jaxsim.api.common import VelRepr
from jaxsim.terrain.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams, ContactsState


@jax_dataclasses.pytree_dataclass
class RigidContactParams(ContactsParams):
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

    def __eq__(self, other: RigidContactParams) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def build(
        cls,
        mu: jtp.Float | None = None,
        K: jtp.Float | None = None,
        D: jtp.Float | None = None,
    ) -> RigidContactParams:
        """Create a `RigidContactParams` instance"""
        return RigidContactParams(
            mu=mu or cls.__dataclass_fields__["mu"].default,
            K=K or cls.__dataclass_fields__["K"].default,
            D=D or cls.__dataclass_fields__["D"].default,
        )

    def valid(self) -> bool:
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

    @staticmethod
    def build(**kwargs) -> RigidContactsState:
        """Create a `RigidContactsState` instance"""

        return RigidContactsState()

    @staticmethod
    def zero(**kwargs) -> RigidContactsState:
        """Build a zero `RigidContactsState` instance from a `JaxSimModel`."""
        return RigidContactsState.build()

    def valid(self, **kwargs) -> bool:
        return True


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    parameters: RigidContactParams = dataclasses.field(
        default_factory=RigidContactParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    @staticmethod
    def detect_contacts(
        W_o_C: jtp.Array,
        W_o_dot_C: jtp.Array,
        terrain_height: jtp.Array,
        terrain_normal: jtp.Array,
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        """
        Detect contacts between the collidable points and the terrain.

        Args:
            W_o_C: The position of the collidable points.
            W_o_dot_C: The linear velocity of the collidable points.
            terrain_height: The height of the terrain at the collidable point position.
            terrain_normal: The normal of the terrain at the collidable point position.

        Returns:
            A tuple containing the activation state of the collidable points along contact normal penetration depth and velocity.
        """

        # TODO: reduce code duplication with js.contact.in_contact
        def detect_contact(
            W_o_C: jtp.Array,
            W_o_dot_C: jtp.Array,
            terrain_height: jtp.Float,
            terrain_normal: jtp.Vector,
        ) -> tuple[jtp.Vector, tuple[Any, ...]]:
            """
            Detect contacts between the collidable points and the terrain."""

            # Unpack the position of the collidable point.
            _px, _py, pz = W_o_C.squeeze()
            W_o_dot_C = W_o_dot_C.squeeze()

            inactive = pz > terrain_height

            # Compute the terrain normal and the contact depth
            n̂ = terrain_normal.squeeze()
            h = jnp.array([0, 0, terrain_height - pz])

            # Compute the penetration depth normal to the terrain.
            δ = jnp.maximum(0.0, jnp.dot(h, n̂))

            # Compute the penetration normal velocity.
            δ̇ = -jnp.dot(W_o_dot_C, n̂)

            return inactive, (δ, δ̇)

        inactive_collidable_points, (delta, delta_dot) = jax.vmap(detect_contact)(
            W_o_C, W_o_dot_C, terrain_height, terrain_normal
        )

        return inactive_collidable_points, (delta, delta_dot)

    @staticmethod
    def _delassus_matrix(
        M: jtp.Matrix,
        J_WC: jtp.Matrix,
    ):
        sl = jnp.s_[:, 0:3, :]
        J_WC_lin = jnp.vstack(J_WC[sl])

        delassus_matrix = J_WC_lin @ pinv(M) @ J_WC_lin.T
        return delassus_matrix

    @staticmethod
    def _compute_ineq_constraint_matrix(
        inactive_collidable_points: jtp.Vector, mu: jtp.Float
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
        G = block_diag(*G)
        return G

    @staticmethod
    def _compute_ineq_bounds(n_collidable_points: jtp.Float) -> jtp.Vector:
        n_constraints = 6 * n_collidable_points
        return jnp.zeros(shape=(n_constraints,))

    @staticmethod
    def _compute_mixed_nu_dot_free(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        references: js.references.JaxSimModelReferences | None = None,
    ) -> jtp.Array:
        references = (
            references
            if references is not None
            else js.references.JaxSimModelReferences.zero(
                model=model, data=data, velocity_representation=VelRepr.Mixed
            )
        )

        with (
            data.switch_velocity_representation(VelRepr.Mixed),
            references.switch_velocity_representation(VelRepr.Mixed),
        ):
            BW_v_WB = data.base_velocity()
            W_o_dot_B = BW_v_WB[0:3]
            W_omega_WB = BW_v_WB[3:6]
            W_v̇_WB, s̈ = js.ode.system_acceleration(
                model=model,
                data=data,
                joint_forces=references.joint_force_references(model=model),
                link_forces=references.link_forces(model=model, data=data),
            )

        # Convert the inertial-fixed base acceleration to a body-fixed base acceleration.
        W_H_B = data.base_transform()
        W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
        BW_X_W = math.Adjoint.from_transform(W_H_BW, inverse=True)
        term1 = BW_X_W @ W_v̇_WB
        term2 = jnp.zeros(6).at[0:3].set(jnp.cross(W_o_dot_B, W_omega_WB))
        BW_v̇_WB = term1 - term2

        BW_ν̇ = jnp.hstack([BW_v̇_WB, s̈])

        return BW_ν̇

    @staticmethod
    def _linear_acceleration_of_collidable_points(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        BW_ν̇: jax.Array,
    ) -> jax.Array:
        with data.switch_velocity_representation(VelRepr.Mixed):
            CW_J_WC_BW = js.contact.jacobian(
                model=model,
                data=data,
                output_vel_repr=VelRepr.Mixed,
            )
            CW_J̇_WC_BW = js.contact.jacobian_derivative(
                model=model,
                data=data,
                output_vel_repr=VelRepr.Mixed,
            )

            BW_ν = data.generalized_velocity()

        CW_a_WC = jnp.vstack(CW_J̇_WC_BW) @ BW_ν̇ + jnp.vstack(CW_J_WC_BW) @ BW_ν
        CW_a_WC = CW_a_WC.reshape(-1, 6)

        return CW_a_WC[:, 0:3].squeeze()

    @staticmethod
    def _compute_baumgarte_stabilization_term(
        inactive_collidable_points: jax.Array,
        delta: jax.Array,
        delta_dot: jax.Array,
        K: jtp.Float,
        D: jtp.Float,
    ) -> jtp.Array:
        def baumgarte_stabilization(
            inactive: jtp.Bool,
            delta: jax.Array,
            delta_dot: jax.Array,
            k_baumgarte: jtp.Float,
            d_baumgarte: jtp.Float,
        ) -> jtp.Array:
            baumgarte_term = jax.lax.cond(
                inactive,
                lambda δ, δ̇, K, D: jnp.zeros(shape=(3,)),
                lambda δ, δ̇, K, D: jnp.zeros(3).at[2].set(K * δ + D * δ̇),
                *(
                    delta,
                    delta_dot,
                    k_baumgarte,
                    d_baumgarte,
                ),
            )
            return baumgarte_term

        baumgarte_term = jax.vmap(
            baumgarte_stabilization, in_axes=(0, 0, 0, None, None)
        )(
            inactive_collidable_points,
            delta,
            delta_dot,
            K,
            D,
        )

        return baumgarte_term

    @staticmethod
    def compute_impact_velocity(
        inactive_collidable_points: jtp.Array,
        M: jtp.Matrix,
        J_WC: jtp.Matrix,
        data: js.data.JaxSimModelData,
    ):
        """Returns the new velocity of the system after a potential impact.

        Args:
            inactive_collidable_points: The activation state of the collidable points.
            M: The mass matrix of the system.
            J_WC: The Jacobian matrix of the collidable points.
            data: The `JaxSimModelData` instance.
        """

        def impact_velocity(
            inactive_collidable_points: jtp.Array,
            nu_pre: jtp.Array,
            M: jtp.Matrix,
            J_WC: jtp.Matrix,
            data: js.data.JaxSimModelData,
        ):
            # Compute system velocity after impact maintaining zero linear velocity of active points
            with data.switch_velocity_representation(VelRepr.Mixed):
                sl = jnp.s_[:, 0:3, :]
                J_WC = J_WC[sl]
                # Zero out the jacobian rows of inactive points
                J_WC = jnp.vstack(
                    jnp.where(
                        inactive_collidable_points[:, jnp.newaxis, jnp.newaxis],
                        jnp.zeros_like(J_WC),
                        J_WC,
                    )
                )

                A = jnp.vstack(
                    [
                        jnp.hstack([M, -J_WC.T]),
                        jnp.hstack([J_WC, jnp.zeros((J_WC.shape[0], J_WC.shape[0]))]),
                    ]
                )
                b = jnp.hstack([M @ nu_pre, jnp.zeros(J_WC.shape[0])])
                x = jnp.linalg.lstsq(A, b)[0]
                nu_post = x[0 : M.shape[0]]

                return nu_post

        with data.switch_velocity_representation(VelRepr.Mixed):
            BW_nu_pre_impact = data.generalized_velocity()

            BW_nu_post_impact = impact_velocity(
                data=data,
                inactive_collidable_points=inactive_collidable_points,
                nu_pre=BW_nu_pre_impact,
                M=M,
                J_WC=J_WC,
            )

        return BW_nu_post_impact

    @staticmethod
    def _compute_link_forces_inertial_fixed(
        CW_f_C: jtp.Matrix,
        W_H_C: jtp.Matrix,
    ):
        def convert_wrench_mixed_to_inertial(
            W_H_C: jax.Array, CW_f: jax.Array
        ) -> jax.Array:
            W_H_CW = W_H_C.at[0:3, 0:3].set(jnp.eye(3))
            CW_H_W = math.Transform.inverse(W_H_CW)
            CW_X_W = math.Adjoint.from_transform(
                CW_H_W,
            )
            W_Xf_CW = CW_X_W.T
            return W_Xf_CW @ CW_f

        W_f_C = jax.vmap(convert_wrench_mixed_to_inertial)(W_H_C, CW_f_C)
        return W_f_C

    def compute_contact_forces(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        link_external_forces: jtp.MatrixLike | None = None,
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        """
        Compute the contact forces.

        Args:
            position: The position of the collidable point.
            velocity: The linear velocity of the collidable point.
            model: The `JaxSimModel` instance.
            data: The `JaxSimModelData` instance.
            link_external_forces: Optional `(n_links, 6)` matrix of external forces acting on the links,
                expressed in the same representation of data.

        Returns:
            A tuple containing the contact forces.
        """

        # Import qpax just in this method
        import qpax

        link_external_forces = (
            link_external_forces
            if link_external_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        )

        # Compute kin-dyn quantities used in the contact model
        with data.switch_velocity_representation(VelRepr.Mixed):
            M = js.model.free_floating_mass_matrix(model=model, data=data)
            J_WC = js.contact.jacobian(model=model, data=data)
            W_H_C = js.contact.transforms(model=model, data=data)
        terrain_height = jax.vmap(self.terrain.height)(position[:, 0], position[:, 1])
        terrain_normal = jax.vmap(self.terrain.normal)(position[:, 0], position[:, 1])
        n_collidable_points = model.kin_dyn_parameters.contact_parameters.point.shape[0]

        # Compute the activation state of the collidable points
        inactive_collidable_points, (delta, delta_dot) = RigidContacts.detect_contacts(
            W_o_C=position,
            W_o_dot_C=velocity,
            terrain_height=terrain_height,
            terrain_normal=terrain_normal,
        )

        delassus_matrix = RigidContacts._delassus_matrix(M=M, J_WC=J_WC)
        # Make it symmetric if not
        delassus_matrix = jax.lax.cond(
            jnp.allclose(delassus_matrix, delassus_matrix.T),
            lambda G: G,
            lambda G: (G + G.T) / 2,
            delassus_matrix,
        )
        # Add regularization for better numerical conditioning
        delassus_matrix = delassus_matrix + 1e-6 * jnp.eye(delassus_matrix.shape[0])

        references = js.references.JaxSimModelReferences.build(
            model=model,
            data=data,
            velocity_representation=data.velocity_representation,
            link_forces=link_external_forces,
        )

        with references.switch_velocity_representation(VelRepr.Mixed):
            nu_dot_free_mixed = RigidContacts._compute_mixed_nu_dot_free(
                model, data, references=references
            )

        free_contact_acc = RigidContacts._linear_acceleration_of_collidable_points(
            model,
            data,
            nu_dot_free_mixed,
        ).flatten()
        # Compute stabilization term
        baumgarte_term = RigidContacts._compute_baumgarte_stabilization_term(
            inactive_collidable_points=inactive_collidable_points,
            delta=delta,
            delta_dot=delta_dot,
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
        h = RigidContacts._compute_ineq_bounds(n_collidable_points=n_collidable_points)
        A = jnp.zeros((0, 3 * n_collidable_points))
        b = jnp.zeros((0,))

        # Solve the optimization problem
        solution, *_ = qpax.solve_qp(Q=Q, q=q, A=A, b=b, G=G, h=h)

        f_C_lin = solution.reshape(-1, 3)

        # Transform linear contact forces to 6D
        CW_f_C = jnp.hstack(
            (
                f_C_lin,
                jnp.zeros((f_C_lin.shape[0], 3)),
            )
        )

        # Transform the contact forces to inertial-fixed representation
        W_f_C = RigidContacts._compute_link_forces_inertial_fixed(
            CW_f_C=CW_f_C, W_H_C=W_H_C
        )

        return (W_f_C, ())
