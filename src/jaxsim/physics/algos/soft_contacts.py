import abc
import dataclasses
from typing import Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim.physics.model.physics_model
import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.skew import Skew
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.physics.model.physics_model import PhysicsModel

from . import utils


@jax_dataclasses.pytree_dataclass
class SoftContactsState:
    tangential_deformation: jtp.Matrix

    @staticmethod
    def zero(
        physics_model: jaxsim.physics.model.physics_model.PhysicsModel,
    ) -> "SoftContactsState":
        return SoftContactsState(
            tangential_deformation=jnp.zeros(shape=(3, physics_model.gc.body.size))
        )

    def valid(
        self, physics_model: jaxsim.physics.model.physics_model.PhysicsModel
    ) -> bool:
        from jaxsim.simulation.utils import check_valid_shape

        return check_valid_shape(
            what="tangential_deformation",
            shape=self.tangential_deformation.shape,
            expected_shape=(3, physics_model.gc.body.size),
            valid=True,
        )

    def replace(self, validate: bool = True, **kwargs) -> "SoftContactsState":
        with jax_dataclasses.copy_and_mutate(self, validate=validate) as updated_state:
            _ = [updated_state.__setattr__(k, v) for k, v in kwargs.items()]

        return updated_state


def collidable_points_pos_vel(
    model: PhysicsModel,
    q: jtp.Vector,
    qd: jtp.Vector,
    xfb: jtp.Vector = None,
) -> Tuple[jtp.Matrix, jtp.Matrix]:
    # Make sure that shape and size are correct
    xfb, q, qd, _, _, _ = utils.process_inputs(physics_model=model, xfb=xfb, q=q, qd=qd)

    # Initialize buffers of link transforms (W_X_i) and 6D inertial velocities (W_v_Wi)
    W_X_i = jnp.zeros(shape=[model.NB, 6, 6])
    W_v_Wi = jnp.zeros(shape=[model.NB, 6, 1])

    # 6D transform of base velocity
    W_X_0 = Adjoint.from_quaternion_and_translation(
        quaternion=xfb[0:4], translation=xfb[4:7], normalize_quaternion=True
    )
    W_X_i = W_X_i.at[0].set(W_X_0)

    # Store the 6D inertial velocity W_v_W0 of the base link
    W_v_W0 = jnp.vstack(jnp.hstack([xfb[10:13], xfb[7:10]]))
    W_v_Wi = W_v_Wi.at[0].set(W_v_W0)

    # Compute useful resources from the model
    S = model.motion_subspaces(q=q)

    # Get the 6D transform between the parent link λi and the joint's predecessor frame
    pre_X_λi = model.tree_transforms

    # Compute the 6D transform of the joints (from predecessor to successor)
    i_X_pre = model.joint_transforms(q=q)

    # Parent array mapping: i -> λ(i).
    # Exception: λ(0) must not be used, it's initialized to -1.
    λ = model.parent_array()

    # ====================
    # Propagate kinematics
    # ====================

    PropagateTransformsCarry = Tuple[jtp.MatrixJax]
    propagate_transforms_carry: PropagateTransformsCarry = (W_X_i,)

    def propagate_transforms(
        carry: PropagateTransformsCarry, i: jtp.Int
    ) -> Tuple[PropagateTransformsCarry, None]:
        # Unpack the carry
        (W_X_i,) = carry

        # We need the inverse transforms (from parent to child direction)
        pre_Xi_i = Adjoint.inverse(i_X_pre[i])
        λi_Xi_pre = Adjoint.inverse(pre_X_λi[i])

        # Compute the parent to child 6D transform
        λi_X_i = λi_Xi_pre @ pre_Xi_i

        # Compute the world to child 6D transform
        W_Xi_i = W_X_i[λ[i]] @ λi_X_i
        W_X_i = W_X_i.at[i].set(W_Xi_i)

        # Pack and return the carry
        return (W_X_i,), None

    (W_X_i,), _ = jax.lax.scan(
        f=propagate_transforms,
        init=propagate_transforms_carry,
        xs=np.arange(start=1, stop=model.NB),
    )

    # ====================
    # Propagate velocities
    # ====================

    PropagateVelocitiesCarry = Tuple[jtp.MatrixJax]
    propagate_velocities_carry: PropagateVelocitiesCarry = (W_v_Wi,)

    def propagate_velocities(
        carry: PropagateVelocitiesCarry, j_vel_and_j_idx: jtp.VectorJax
    ) -> Tuple[PropagateVelocitiesCarry, None]:
        # Unpack the scanned data
        qd_ii = j_vel_and_j_idx[0]
        ii = jnp.array(j_vel_and_j_idx[1], dtype=int)

        # Given a joint whose velocity is qd[ii], the index of its parent link is ii + 1
        i = ii + 1

        # Unpack the carry
        (W_v_Wi,) = carry

        # Propagate the 6D velocity
        W_vi_Wi = W_v_Wi[λ[i]] + W_X_i[i] @ (S[i] * qd_ii)
        W_v_Wi = W_v_Wi.at[i].set(W_vi_Wi)

        # Pack and return the carry
        return (W_v_Wi,), None

    (W_v_Wi,), _ = jax.lax.scan(
        f=propagate_velocities,
        init=propagate_velocities_carry,
        xs=jnp.vstack([qd, jnp.arange(start=0, stop=qd.size)]).T,
    )

    # ==================================================
    # Compute position and velocity of collidable points
    # ==================================================

    def process_point_kinematics(
        Li_p_C: jtp.VectorJax, parent_body: jtp.Int
    ) -> Tuple[jtp.VectorJax, jtp.VectorJax]:
        # Compute the position of the collidable point
        W_p_Ci = (
            Adjoint.to_transform(adjoint=W_X_i[parent_body]) @ jnp.hstack([Li_p_C, 1])
        )[0:3]

        # Compute the linear part of the mixed velocity Ci[W]_v_{W,Ci}
        CW_vl_WCi = (
            jnp.block([jnp.eye(3), -Skew.wedge(vector=W_p_Ci).squeeze()])
            @ W_v_Wi[parent_body].squeeze()
        )

        return W_p_Ci, CW_vl_WCi

    # Process all the collidable points in parallel
    W_p_Ci, CW_v_WC = jax.vmap(process_point_kinematics)(
        model.gc.point.T, model.gc.body
    )

    return W_p_Ci.transpose(), CW_v_WC.transpose()


@jax_dataclasses.pytree_dataclass
class SoftContactsParams:
    """"""

    K: float = dataclasses.field(default=jnp.array(1e6, dtype=float))
    D: float = dataclasses.field(default=jnp.array(2000, dtype=float))
    mu: float = dataclasses.field(default=jnp.array(0.5, dtype=float))

    @staticmethod
    def build(
        K: float = 1e6, D: float = 2_000, mu: float = 0.5
    ) -> "SoftContactsParams":
        """"""

        return SoftContactsParams(
            K=jnp.array(K, dtype=float),
            D=jnp.array(D, dtype=float),
            mu=jnp.array(mu, dtype=float),
        )


@jax_dataclasses.pytree_dataclass
class SoftContacts:
    parameters: SoftContactsParams = dataclasses.field(
        default_factory=SoftContactsParams
    )

    terrain: Terrain = dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        tangential_deformation: jtp.Vector,
    ) -> Tuple[jtp.Vector, jtp.Vector]:
        # Short name of parameters
        K = self.parameters.K
        D = self.parameters.D
        μ = self.parameters.mu

        # Material 3D tangential deformation and its derivative
        m = tangential_deformation.squeeze()
        ṁ = jnp.zeros_like(m)

        # ========================
        # Normal force computation
        # ========================

        # Unpack the position of the collidable point
        px, py, pz = W_p_C = position.squeeze()
        vx, vy, vz = W_ṗ_C = velocity.squeeze()

        # Compute the terrain normal and the contact depth
        n̂ = self.terrain.normal(x=px, y=py).squeeze()
        h = jnp.array([0, 0, self.terrain.height(x=px, y=py) - pz])

        # Compute the penetration depth normal to the terrain
        δ = jnp.maximum(0.0, jnp.dot(h, n̂))

        # Compute the penetration normal velocity
        δ̇ = -jnp.dot(W_ṗ_C, n̂)

        # Non-linear spring-damper model.
        # This is the force magnitude along the direction normal to the terrain.
        force_normal_mag = jnp.sqrt(δ) * (K * δ + D * δ̇)

        # Prevent negative normal forces that might occur when δ̇ is largely negative
        force_normal_mag = jnp.maximum(0.0, force_normal_mag)

        # Compute the 3D linear force in C[W] frame
        force_normal = force_normal_mag * n̂

        # ====================================
        # No friction and no tangential forces
        # ====================================

        # Compute the adjoint C[W]->W for transforming 6D forces from mixed to inertial.
        # Note: this is equal to the 6D velocities transform: CW_X_W.transpose().
        W_Xf_CW = jnp.block(
            [
                [jnp.eye(3), jnp.zeros(shape=(3, 3))],
                [Skew.wedge(W_p_C), jnp.eye(3)],
            ]
        )

        def with_no_friction():
            # Compute 6D mixed force in C[W]
            CW_f_lin = force_normal
            CW_f = jnp.hstack([force_normal, jnp.zeros_like(CW_f_lin)])

            # Compute lin-ang 6D forces (inertial representation)
            W_f = W_Xf_CW @ CW_f

            return W_f, ṁ

        # =========================
        # Compute tangential forces
        # =========================

        def with_friction():
            # Initialize the tangential deformation rate ṁ.
            # For inactive contacts with m≠0, this is the dynamics of the material
            # relaxation converging exponentially to steady state.
            ṁ = (-K / D) * m

            # Check if the collidable point is below ground.
            # Note: when δ=0, we consider the point still not it contact such that
            #       we prevent divisions by 0 in the computations below.
            active_contact = pz < self.terrain.height(x=px, y=py)

            def above_terrain():
                return jnp.zeros(6), ṁ

            def below_terrain():
                # Decompose the velocity in normal and tangential components
                v_normal = jnp.dot(W_ṗ_C, n̂) * n̂
                v_tangential = W_ṗ_C - v_normal

                # Compute the tangential force. If inside the friction cone, the contact
                f_tangential = -jnp.sqrt(δ) * (K * m + D * v_tangential)

                def sticking_contact():
                    # Sum the normal and tangential forces, and create the 6D force
                    CW_f_stick = force_normal + f_tangential
                    CW_f = jnp.hstack([CW_f_stick, jnp.zeros(3)])

                    # In this case the 3D material deformation is the tangential velocity
                    ṁ = v_tangential

                    # Return the 6D force in the contact frame and
                    # the deformation derivative
                    return CW_f, ṁ

                def slipping_contact():
                    # Project the force to the friction cone boundary
                    f_tangential_projected = (μ * force_normal_mag) * (
                        f_tangential / jnp.linalg.norm(f_tangential)
                    )

                    # Sum the normal and tangential forces, and create the 6D force
                    CW_f_slip = force_normal + f_tangential_projected
                    CW_f = jnp.hstack([CW_f_slip, jnp.zeros(3)])

                    # Correct the material deformation derivative for slipping contacts.
                    # Basically we compute ṁ such that we get `f_tangential` on the cone
                    # given the current (m, δ).
                    ε = 1e-6
                    α = -K * jnp.sqrt(δ)
                    δε = jnp.maximum(δ, ε)
                    βε = -D * jnp.sqrt(δε)
                    ṁ = (f_tangential_projected - α * m) / βε

                    # Return the 6D force in the contact frame and
                    # the deformation derivative
                    return CW_f, ṁ

                CW_f, ṁ = jax.lax.cond(
                    pred=jnp.linalg.norm(f_tangential) > μ * force_normal_mag,
                    true_fun=lambda _: slipping_contact(),
                    false_fun=lambda _: sticking_contact(),
                    operand=None,
                )

                # Express the 6D force in the world frame
                W_f = W_Xf_CW @ CW_f

                # Return the 6D force in the world frame and the deformation derivative
                return W_f, ṁ

            # (W_f, ṁ)
            return jax.lax.cond(
                pred=active_contact,
                true_fun=lambda _: below_terrain(),
                false_fun=lambda _: above_terrain(),
                operand=None,
            )

        # (W_f, ṁ)
        return jax.lax.cond(
            pred=(μ == 0.0),
            true_fun=lambda _: with_no_friction(),
            false_fun=lambda _: with_friction(),
            operand=None,
        )
