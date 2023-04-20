from typing import NamedTuple, Tuple

import jax
import jax.experimental.loops
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.physics.model.physics_model
import jaxsim.typing as jtp
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.conv import Convert
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
    xfb, q, qd, _, _, _ = utils.process_inputs(physics_model=model, xfb=xfb, q=q, qd=qd)

    Xa = jnp.array([jnp.eye(6)] * (model.NB))
    vb = jnp.array([jnp.zeros([6, 1])] * (model.NB))

    # 6D transform of base velocity
    Xa_0 = B_X_W = Adjoint.from_quaternion_and_translation(
        quaternion=xfb[0:4], translation=xfb[4:7], inverse=True
    )
    Xa = Xa.at[0].set(Xa_0)

    vfb = jnp.vstack(jnp.hstack([xfb[10:13], xfb[7:10]]))
    vb_0 = Xa[0] @ vfb
    vb = vb.at[0].set(vb_0)

    Xtree = model.tree_transforms
    S = model.motion_subspaces(q=q)
    XJ = model.joint_transforms(q=q)
    parent_link_of = model.parent_array()

    with jax.experimental.loops.Scope() as s:
        s.Xa = Xa
        s.vb = vb

        for i in s.range(1, model.NB):
            ii = i - 1

            Xup = XJ[i] @ Xtree[i]
            vJ = S[i] * qd[ii] if qd.size != 0 else S[i] * 0

            Xa_i = Xup @ s.Xa[parent_link_of[i]]
            s.Xa = s.Xa.at[i].set(Xa_i)

            vb_i = jnp.vstack(Xup @ s.vb[parent_link_of[i]]) + vJ
            s.vb = s.vb.at[i].set(vb_i)

        Xa = s.Xa
        vb = s.vb

    def process_collidable_points_of_link_with_index(i: int) -> jtp.VectorJax:
        X = jnp.linalg.inv(Xa[i])
        v = X @ vb[i]

        pt = Convert.coordinates_tf(X=X, p=model.gc.point)
        vpt = Convert.velocities_threed(v_6d=v, p=pt)

        return jnp.vstack([pt, vpt])

    parallel_processing = jax.vmap(process_collidable_points_of_link_with_index)
    pos_vel_unpacked = parallel_processing(jnp.array(list(set(model.gc.body))))

    pos_vel = jnp.sum(
        jnp.array(
            [
                jnp.where(
                    model.gc.body == i,
                    pos_vel_unpacked[idx],
                    jnp.zeros_like(pos_vel_unpacked[0]),
                )
                for idx, i in enumerate(set(model.gc.body))
            ]
        ),
        axis=0,
    )

    return pos_vel[0:3, :].squeeze(), pos_vel[3:6, :].squeeze()


@jax_dataclasses.pytree_dataclass
class SoftContactsParams:
    K: float = jnp.array(1e6, dtype=float)
    D: float = jnp.array(2000, dtype=float)
    mu: float = jnp.array(0.5, dtype=float)


def soft_contacts_model(
    positions: jtp.Vector,
    velocities: jtp.Vector,
    tangential_deformation: jtp.Vector,
    soft_contacts_params: SoftContactsParams = SoftContactsParams(),
    terrain: Terrain = FlatTerrain(),
) -> Tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix]:
    p = jnp.array(positions).squeeze()
    pd = jnp.array(velocities).squeeze()
    u = jnp.array(tangential_deformation).squeeze()

    K = soft_contacts_params.K
    D = soft_contacts_params.D
    mu = soft_contacts_params.mu

    # ========================
    # Normal force computation
    # ========================

    # Compute the terrain height and normal
    terrain_height = jax.vmap(terrain.height)(p[0, :], p[1, :])
    terrain_normal = jax.vmap(terrain.normal)(p[0, :], p[1, :])

    # Boolean map to select points in contact
    in_contact = jnp.where(p[2, :] <= terrain_height, True, False)

    # Compute the penetration (<0 for contacts) normal to the terrain
    penetration_vertical = jnp.zeros_like(p).at[2, :].set(p[2, :] - terrain_height)
    penetration_normal = jax.vmap(jnp.dot)(penetration_vertical.T, terrain_normal).T

    # Compute the penetration depth δ (>0 for contacts) and its velocity δ̇
    delta = -penetration_normal
    delta_dot = jax.vmap(jnp.dot)(-pd.T, terrain_normal)

    # Filter only active contacts
    delta = jnp.where(in_contact, delta, 0.0)
    delta_dot = jnp.where(in_contact, delta_dot, 0.0)
    sqrt_delta = jnp.sqrt(delta)

    # Non-linear spring-damper model.
    # This is the force magnitude along the direction normal to the terrain.
    forces_normal_mag = sqrt_delta * (K * delta + D * delta_dot)

    # Compute the 3D linear force in C[W] frame
    forces_normal = terrain_normal.T * forces_normal_mag

    # ====================================
    # No friction and no tangential forces
    # ====================================

    # Compute the adjoint C[W]->W for transforming 6D forces from mixed to inertial.
    # Note: this is equal to the 6D velocities transform: CW_X_W.transpose().
    W_Xf_CW = jax.vmap(
        lambda W_p_C: jnp.block(
            [
                [jnp.eye(3), jnp.zeros(shape=(3, 3))],
                [Skew.wedge(W_p_C), jnp.eye(3)],
            ]
        )
    )(p.T)

    def with_no_friction():
        # Compute 6D mixed forces in C[W]
        CW_f_lin = forces_normal
        CW_f = jnp.vstack([CW_f_lin, jnp.zeros_like(CW_f_lin)])

        # Compute lin-ang 6D forces (inertial representation)
        W_f = jax.vmap(lambda X, f: X @ f)(W_Xf_CW, CW_f.T).T

        return W_f, jnp.zeros_like(u), jnp.zeros_like(forces_normal[0])

    # =========================
    # Compute tangential forces
    # =========================

    def with_friction():
        # Initialize the tangential velocity of the point
        v_perpendicular = jax.vmap(jnp.dot)(pd.T, terrain_normal) * terrain_normal.T
        v_tangential = pd - v_perpendicular
        v_tangential = jnp.where(in_contact, v_tangential, 0.0)

        # Initialize the tangential deformation rate u̇.
        # For inactive contacts with u≠0, this is the dynamics of the material relaxation.
        ud = -K / D * u

        # Assume all contacts are in sticking state
        # -----------------------------------------

        # Compute the contact forces
        f_stick = -K * sqrt_delta * u - D * sqrt_delta * v_tangential
        f_stick = jnp.where(in_contact, f_stick, 0.0)

        # Use the tangential velocity of the contact points to update u̇
        ud = jnp.where(in_contact, v_tangential, ud)

        # Correct forces for contacts in slipping state
        # ---------------------------------------------

        # Test for slipping
        f_cone_boundary = mu * forces_normal_mag
        f_stick_norm_sq = jnp.power(f_stick, 2).sum(axis=0)
        slipping = jnp.where(
            f_stick_norm_sq > jnp.power(f_cone_boundary, 2), True, False
        )

        # Project forces outside the friction cone to the cone boundary
        f_stick_norm_sq_eps = jnp.where(f_stick_norm_sq == 0.0, 1e-6, f_stick_norm_sq)
        forces_tangential = jnp.where(
            slipping,
            f_cone_boundary * (f_stick / jnp.sqrt(f_stick_norm_sq_eps)),
            f_stick,
        )

        # Correct u̇ for slipping contacts.
        # Basically we compute u̇ s.t. we get f_slip on the cone given the current (u, δ).
        sqrt_delta_eps = jnp.where(sqrt_delta == 0.0, 1e-6, sqrt_delta)
        ud_slipping = -(forces_tangential + K * sqrt_delta * u) / (D * sqrt_delta_eps)
        ud = jnp.where(slipping, ud_slipping, ud)

        # Build the output variables
        # --------------------------

        # Compute 6D mixed forces in C[W]
        CW_f_lin = forces_normal + forces_tangential
        CW_f = jnp.vstack([CW_f_lin, jnp.zeros_like(CW_f_lin)])

        # Compute lin-ang 6D forces (inertial representation)
        W_f = jax.vmap(lambda X, f: X @ f)(W_Xf_CW, CW_f.T).T

        return W_f, ud, jnp.zeros_like(f_cone_boundary)

    # Return the forces according to the friction value
    return jax.lax.cond(
        pred=(mu == 0.0),
        true_fun=lambda _: with_no_friction(),
        false_fun=lambda _: with_friction(),
        operand=None,
    )
