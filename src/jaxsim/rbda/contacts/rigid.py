from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses
import qpax
from jax.numpy.linalg import pinv
from jax.scipy.linalg import block_diag

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import math
from jaxsim.api.common import VelRepr
from jaxsim.terrain.terrain import FlatTerrain, Terrain
from jaxsim.utils import Mutability

from .common import ContactModel, ContactsParams, ContactsState


@jax_dataclasses.pytree_dataclass
class RigidContactParams(ContactsParams):
    """Parameters of the rigid contacts model."""

    # Inactive contact points at the previous time step
    inactive_points_prev: jtp.Vector

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
                HashedNumpyArray.hash_of_array(self.inactive_points_prev),
            )
        )

    def __eq__(self, other: RigidContactParams) -> bool:
        return hash(self) == hash(other)

    @staticmethod
    def build(
        inactive_points_prev: jtp.Vector,
        mu: jtp.Float = 0.5,
        K: jtp.Float = 0.0,
        D: jtp.Float = 0.0,
    ) -> RigidContactParams:
        """Create a `RigidContactParams` instance"""
        return RigidContactParams(
            mu=mu, K=K, D=D, inactive_points_prev=inactive_points_prev
        )

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel,
        *,
        static_friction_coefficient: jtp.Float = 0.5,
        K: jtp.Float = 0.0,
        D: jtp.Float = 0.0,
    ) -> RigidContactParams:
        """Build a `RigidContactParams` instance from a `JaxSimModel`."""

        inactive_points_prev = jnp.zeros(
            model.kin_dyn_parameters.contact_parameters.point.shape[0]
        )

        jax.debug.print(
            "==========inactive_points_prev={inactive_points_prev}",
            inactive_points_prev=inactive_points_prev.shape,
        )

        return RigidContactParams.build(
            mu=static_friction_coefficient,
            K=K,
            D=D,
            inactive_points_prev=inactive_points_prev,
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

    def __hash__(self) -> int:
        return hash(tuple(jnp.atleast_1d(self.inactive_points_prev.flatten()).tolist()))

    def __eq__(self, other: RigidContactsState) -> bool:
        return hash(self) == hash(other)

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
    ) -> RigidContactsState:
        """Build a `RigidContactsState` instance from a `JaxSimModel`."""
        return RigidContactsState.build()

    @staticmethod
    def build() -> RigidContactsState:
        """Create a `RigidContactsState` instance"""
        return RigidContactsState()

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> RigidContactsState:
        """Build a zero `RigidContactsState` instance from a `JaxSimModel`."""
        return RigidContactsState.build()

    def valid(self, model: js.model.JaxSimModel) -> bool:
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
    def _detect_contacts(
        W_o_C: jtp.Array,
        W_o_dot_C: jtp.Array,
        terrain_height: jtp.Array,
        terrain_normal: jtp.Array,
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        """
        Detect contacts between the collidable points and the terrain.
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
        references: js.references.JaxSimModelReferences,
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
                link_external_forces_inertial=references.link_forces(
                    model=model, data=data
                ),
            )

        # Convert the inertial-fixed base acceleration to a body-fixed base acceleration.
        W_H_B = data.base_transform()
        W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
        BW_H_W = math.Transform.inverse(W_H_BW)
        BW_X_W = math.Adjoint.from_transform(
            BW_H_W,
        )
        term1 = BW_X_W @ W_v̇_WB
        term2 = jnp.zeros(6).at[0:3].set(jnp.cross(W_o_dot_B, W_omega_WB))
        BW_v̇_WB = term1 - term2

        BW_ν̇ = jnp.hstack([BW_v̇_WB, s̈])

        return BW_ν̇

    @staticmethod
    def _linear_acceleration_of_collidable_points(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        nu_dot_mixed: jax.Array,
    ) -> jax.Array:
        with data.switch_velocity_representation(VelRepr.Mixed):
            CW_J_WC_BW = js.contact.jacobian(
                model=model,
                data=data,
                output_vel_repr=VelRepr.Mixed,
            )
            CW_J_dot_WC_BW = js.contact.jacobian_derivative(
                model=model,
                data=data,
                output_vel_repr=VelRepr.Mixed,
            )

            BW_ν = data.generalized_velocity()

        CW_a_WC = jax.vmap(
            lambda J_dot, J, nu_dot, nu: J_dot @ nu + J @ nu_dot,
            in_axes=(0, 0, None, None),
        )(
            CW_J_dot_WC_BW,
            CW_J_WC_BW,
            nu_dot_mixed,
            BW_ν,
        )
        print(f"{CW_a_WC.shape=}")

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
            inactive: bool,
            delta: jax.Array,
            delta_dot: jax.Array,
            k_baumgarte: float,
            d_baumgarte: float,
        ) -> jtp.Array:
            baumgarte_term = jax.lax.cond(
                inactive,
                lambda in_arg: jnp.zeros(shape=(3)),
                lambda in_arg: jnp.zeros(3)
                .at[2]
                .set(in_arg[2] * in_arg[0] + in_arg[3] * in_arg[1]),
                (
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
    def _compute_impact_velocity(
        inactive_collidable_points: jtp.Array,
        inactive_collidable_points_prev: jtp.Array,
        M: jtp.Matrix,
        J_WC: jtp.Matrix,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
    ):
        """Returns the new velocity of the system after a potential impact."""

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
                J_WC_lin = J_WC[sl]
                # Zero out the jacobian rows of inactive points
                J_WC_lin = jax.vmap(
                    lambda J, inactive: jax.lax.cond(
                        inactive, lambda _: jnp.zeros_like(J), lambda _: J, operand=None
                    )
                )(J_WC_lin, inactive_collidable_points)
                J_WC_lin = jnp.vstack(J_WC_lin)

                I = jnp.eye(M.shape[0])
                nu_post = (
                    I
                    - pinv(M)
                    @ J_WC_lin.T
                    @ pinv(J_WC_lin @ pinv(M) @ J_WC_lin.T)
                    @ J_WC_lin
                ) @ nu_pre
                return nu_post

        new_impacts = jnp.any(
            inactive_collidable_points_prev & ~inactive_collidable_points
        )
        jax.debug.print("new_impacts={new_impacts}", new_impacts=new_impacts)

        nu_pre = data.generalized_velocity()

        nu = jax.lax.cond(
            new_impacts,
            lambda operands: impact_velocity(
                data=operands["data"],
                inactive_collidable_points=operands["inactive_collidable_points"],
                nu_pre=operands["nu_pre"],
                M=operands["M"],
                J_WC=operands["J_WC"],
            ),
            lambda operands: operands["nu_pre"],
            dict(
                model=model,
                data=data,
                inactive_collidable_points=inactive_collidable_points,
                nu_pre=nu_pre,
                M=M,
                J_WC=J_WC,
            ),
        )

        return nu

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
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        # Compute kin-dyn quantities used in the contact model
        with data.switch_velocity_representation(VelRepr.Mixed):
            M = js.model.free_floating_mass_matrix(model=model, data=data)
            J_WC = js.contact.jacobian(model=model, data=data)
            W_H_C = js.contact.transforms(model=model, data=data)
        inactive_collisable_points_prev = self.parameters.inactive_points_prev
        terrain_height = jax.vmap(self.terrain.height)(position[:, 0], position[:, 1])
        terrain_normal = jax.vmap(self.terrain.normal)(position[:, 0], position[:, 1])
        n_collidable_points = model.kin_dyn_parameters.contact_parameters.point.shape[0]

        # Compute the activation state of the collidable points
        inactive_collidable_points, (delta, delta_dot) = RigidContacts._detect_contacts(
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

        nu_dot_free_mixed = RigidContacts._compute_mixed_nu_dot_free(
            model, data, references=None
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

        jax.debug.print(
            "Shapes: Q={Q}, q={q}, A={A}, b={b}, G={G}, h={h}",
            Q=Q.shape,
            q=q.shape,
            A=A.shape,
            b=b.shape,
            G=G.shape,
            h=h.shape,
        )

        # Solve the optimization problem
        solution, s, z, y, converged, iters = qpax.solve_qp(
            Q=Q, q=q, A=A, b=b, G=G, h=h
        )

        jax.debug.print(
            "x={x}, s={s}, z={z}, y={y}, converged={converged}, iters={iters}",
            x=solution,
            s=s,
            z=z,
            y=y,
            converged=converged,
            iters=iters,
        )

        f_C_lin = solution.reshape(-1, 3)

        # Compute the impact velocity
        nu = RigidContacts._compute_impact_velocity(
            model=model,
            data=data,
            inactive_collidable_points=inactive_collidable_points,
            inactive_collidable_points_prev=inactive_collisable_points_prev,
            M=M,
            J_WC=J_WC,
        )

        jax.debug.print(
            "inactive_collidable_points_prev={inactive_collidable_points_prev}",
            inactive_collidable_points_prev=inactive_collisable_points_prev,
        )

        jax.debug.print(
            "inactive_collidable_points={inactive_collidable_points}",
            inactive_collidable_points=inactive_collidable_points,
        )

        with self.mutable_context(
            mutability=Mutability.MUTABLE, restore_after_exception=True
        ):
            self.parameters.inactive_points_prev = inactive_collidable_points

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

        return W_f_C, (nu,)
