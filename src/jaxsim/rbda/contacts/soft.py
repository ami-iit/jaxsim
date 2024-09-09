from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Skew, StandardGravity
from jaxsim.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams, ContactsState


@jax_dataclasses.pytree_dataclass
class SoftContactsParams(ContactsParams):
    """Parameters of the soft contacts model."""

    K: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(1e6, dtype=float)
    )

    D: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(2000, dtype=float)
    )

    mu: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray.hash_of_array(self.K),
                HashedNumpyArray.hash_of_array(self.D),
                HashedNumpyArray.hash_of_array(self.mu),
            )
        )

    def __eq__(self, other: SoftContactsParams) -> bool:

        if not isinstance(other, SoftContactsParams):
            return NotImplemented

        return hash(self) == hash(other)

    @staticmethod
    def build(
        K: jtp.FloatLike = 1e6, D: jtp.FloatLike = 2_000, mu: jtp.FloatLike = 0.5
    ) -> SoftContactsParams:
        """
        Create a SoftContactsParams instance with specified parameters.

        Args:
            K: The stiffness parameter.
            D: The damping parameter of the soft contacts model.
            mu: The static friction coefficient.

        Returns:
            A SoftContactsParams instance with the specified parameters.
        """

        return SoftContactsParams(
            K=jnp.array(K, dtype=float),
            D=jnp.array(D, dtype=float),
            mu=jnp.array(mu, dtype=float),
        )

    @staticmethod
    def build_default_from_jaxsim_model(
        model: js.model.JaxSimModel,
        *,
        standard_gravity: jtp.FloatLike = StandardGravity,
        static_friction_coefficient: jtp.FloatLike = 0.5,
        max_penetration: jtp.FloatLike = 0.001,
        number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
        damping_ratio: jtp.FloatLike = 1.0,
    ) -> SoftContactsParams:
        """
        Create a SoftContactsParams instance with good default parameters.

        Args:
            model: The target model.
            standard_gravity: The standard gravity constant.
            static_friction_coefficient:
                The static friction coefficient between the model and the terrain.
            max_penetration: The maximum penetration depth.
            number_of_active_collidable_points_steady_state:
                The number of contacts supporting the weight of the model
                in steady state.
            damping_ratio: The ratio controlling the damping behavior.

        Returns:
            A `SoftContactsParams` instance with the specified parameters.

        Note:
            The `damping_ratio` parameter allows to operate on the following conditions:
            - ξ > 1.0: over-damped
            - ξ = 1.0: critically damped
            - ξ < 1.0: under-damped
        """

        # Use symbols for input parameters.
        ξ = damping_ratio
        δ_max = max_penetration
        μc = static_friction_coefficient

        # Compute the total mass of the model.
        m = jnp.array(model.kin_dyn_parameters.link_parameters.mass).sum()

        # Rename the standard gravity.
        g = standard_gravity

        # Compute the average support force on each collidable point.
        f_average = m * g / number_of_active_collidable_points_steady_state

        # Compute the stiffness to get the desired steady-state penetration.
        K = f_average / jnp.power(δ_max, 3 / 2)

        # Compute the damping using the damping ratio.
        critical_damping = 2 * jnp.sqrt(K * m)
        D = ξ * critical_damping

        return SoftContactsParams.build(K=K, D=D, mu=μc)

    def valid(self) -> bool:
        """
        Check if the parameters are valid.

        Returns:
            `True` if the parameters are valid, `False` otherwise.
        """

        return (
            jnp.all(self.K >= 0.0)
            and jnp.all(self.D >= 0.0)
            and jnp.all(self.mu >= 0.0)
        )


@jax_dataclasses.pytree_dataclass
class SoftContacts(ContactModel):
    """Soft contacts model."""

    parameters: SoftContactsParams = dataclasses.field(
        default_factory=SoftContactsParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    def compute_contact_forces(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        tangential_deformation: jtp.Vector,
    ) -> tuple[jtp.Vector, tuple[jtp.Vector]]:
        """
        Compute the contact forces and material deformation rate.

        Args:
            position: The position of the collidable point.
            velocity: The linear velocity of the collidable point.
            tangential_deformation: The tangential deformation.

        Returns:
            A tuple containing the contact force and material deformation rate.
        """

        # Short name of parameters
        K = self.parameters.K
        D = self.parameters.D
        μ = self.parameters.mu

        # Material 3D tangential deformation and its derivative
        m = tangential_deformation.squeeze()
        ṁ = jnp.zeros_like(m)

        # Note: all the small hardcoded tolerances in this method have been introduced
        # to allow jax differentiating through this algorithm. They should not affect
        # the accuracy of the simulation, although they might make it less readable.

        # ========================
        # Normal force computation
        # ========================

        # Unpack the position of the collidable point.
        px, py, pz = W_p_C = position.squeeze()
        W_ṗ_C = velocity.squeeze()

        # Compute the terrain normal and the contact depth.
        n̂ = self.terrain.normal(x=px, y=py).squeeze()
        h = jnp.array([0, 0, self.terrain.height(x=px, y=py) - pz])

        # Compute the penetration depth normal to the terrain.
        δ = jnp.maximum(0.0, jnp.dot(h, n̂))

        # Compute the penetration normal velocity.
        δ̇ = -jnp.dot(W_ṗ_C, n̂)

        # Non-linear spring-damper model.
        # This is the force magnitude along the direction normal to the terrain.
        force_normal_mag = jax.lax.select(
            pred=δ >= 1e-9,
            on_true=jnp.sqrt(δ + 1e-12) * (K * δ + D * δ̇),
            on_false=jnp.array(0.0),
        )

        # Prevent negative normal forces that might occur when δ̇ is largely negative.
        force_normal_mag = jnp.maximum(0.0, force_normal_mag)

        # Compute the 3D linear force in C[W] frame.
        force_normal = force_normal_mag * n̂

        # ====================================
        # No friction and no tangential forces
        # ====================================

        # Compute the adjoint C[W]->W for transforming 6D forces from mixed to inertial.
        # Note: this is equal to the 6D velocities transform: CW_X_W.transpose().
        W_Xf_CW = jnp.vstack(
            [
                jnp.block([jnp.eye(3), jnp.zeros(shape=(3, 3))]),
                jnp.block([Skew.wedge(W_p_C), jnp.eye(3)]),
            ]
        )

        def with_no_friction():
            # Compute 6D mixed force in C[W].
            CW_f_lin = force_normal
            CW_f = jnp.hstack([force_normal, jnp.zeros_like(CW_f_lin)])

            # Compute lin-ang 6D forces (inertial representation).
            W_f = W_Xf_CW @ CW_f

            return W_f, (ṁ,)

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
                return jnp.zeros(6), (ṁ,)

            def below_terrain():
                # Decompose the velocity in normal and tangential components.
                v_normal = jnp.dot(W_ṗ_C, n̂) * n̂
                v_tangential = W_ṗ_C - v_normal

                # Compute the tangential force. If inside the friction cone, the contact.
                f_tangential = -jnp.sqrt(δ + 1e-12) * (K * m + D * v_tangential)

                def sticking_contact():
                    # Sum the normal and tangential forces, and create the 6D force.
                    CW_f_stick = force_normal + f_tangential
                    CW_f = jnp.hstack([CW_f_stick, jnp.zeros(3)])

                    # In this case the 3D material deformation is the tangential velocity.
                    ṁ = v_tangential

                    # Return the 6D force in the contact frame and
                    # the deformation derivative.
                    return CW_f, ṁ

                def slipping_contact():
                    # Project the force to the friction cone boundary.
                    f_tangential_projected = (μ * force_normal_mag) * (
                        f_tangential / jnp.maximum(jnp.linalg.norm(f_tangential), 1e-9)
                    )

                    # Sum the normal and tangential forces, and create the 6D force.
                    CW_f_slip = force_normal + f_tangential_projected
                    CW_f = jnp.hstack([CW_f_slip, jnp.zeros(3)])

                    # Correct the material deformation derivative for slipping contacts.
                    # Basically we compute ṁ such that we get `f_tangential` on the cone
                    # given the current (m, δ).
                    ε = 1e-9
                    δε = jnp.maximum(δ, ε)
                    α = -K * jnp.sqrt(δε)
                    β = -D * jnp.sqrt(δε)
                    ṁ = (f_tangential_projected - α * m) / β

                    # Return the 6D force in the contact frame and
                    # the deformation derivative.
                    return CW_f, ṁ

                CW_f, ṁ = jax.lax.cond(
                    pred=f_tangential.dot(f_tangential) > (μ * force_normal_mag) ** 2,
                    true_fun=lambda _: slipping_contact(),
                    false_fun=lambda _: sticking_contact(),
                    operand=None,
                )

                # Express the 6D force in the world frame.
                W_f = W_Xf_CW @ CW_f

                # Return the 6D force in the world frame and the deformation derivative.
                return W_f, (ṁ,)

            # (W_f, (ṁ,))
            return jax.lax.cond(
                pred=active_contact,
                true_fun=lambda _: below_terrain(),
                false_fun=lambda _: above_terrain(),
                operand=None,
            )

        # (W_f, (ṁ,))
        return jax.lax.cond(
            pred=(μ == 0.0),
            true_fun=lambda _: with_no_friction(),
            false_fun=lambda _: with_friction(),
            operand=None,
        )


@jax_dataclasses.pytree_dataclass
class SoftContactsState(ContactsState):
    """
    Class storing the state of the soft contacts model.

    Attributes:
        tangential_deformation:
            The matrix of 3D tangential material deformations corresponding to
            each collidable point.
    """

    tangential_deformation: jtp.Matrix

    def __hash__(self) -> int:
        return hash(
            tuple(jnp.atleast_1d(self.tangential_deformation.flatten()).tolist())
        )

    def __eq__(self, other: SoftContactsState) -> bool:
        if not isinstance(other, SoftContactsState):
            return False

        return hash(self) == hash(other)

    @staticmethod
    def build_from_jaxsim_model(
        model: js.model.JaxSimModel | None = None,
        tangential_deformation: jtp.Matrix | None = None,
    ) -> SoftContactsState:
        """
        Build a `SoftContactsState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the soft contacts state.
            tangential_deformation: The matrix of 3D tangential material deformations.

        Returns:
            The `SoftContactsState` built from the `JaxSimModel`.

        Note:
            If any of the state components are not provided, they are built from the
            `JaxSimModel` and initialized to zero.
        """

        return SoftContactsState.build(
            tangential_deformation=tangential_deformation,
            number_of_collidable_points=len(
                model.kin_dyn_parameters.contact_parameters.body
            ),
        )

    @staticmethod
    def build(
        tangential_deformation: jtp.Matrix | None = None,
        number_of_collidable_points: int | None = None,
    ) -> SoftContactsState:
        """
        Create a `SoftContactsState`.

        Args:
            tangential_deformation:
                The matrix of 3D tangential material deformations corresponding to
                each collidable point.
            number_of_collidable_points: The number of collidable points.

        Returns:
            A `SoftContactsState` instance.
        """

        tangential_deformation = (
            tangential_deformation
            if tangential_deformation is not None
            else jnp.zeros(shape=(number_of_collidable_points, 3))
        )

        if tangential_deformation.shape[1] != 3:
            raise RuntimeError("The tangential deformation matrix must have 3 columns.")

        if (
            number_of_collidable_points is not None
            and tangential_deformation.shape[0] != number_of_collidable_points
        ):
            msg = "The number of collidable points must match the number of rows "
            msg += "in the tangential deformation matrix."
            raise RuntimeError(msg)

        return SoftContactsState(
            tangential_deformation=jnp.array(tangential_deformation).astype(float)
        )

    @staticmethod
    def zero(model: js.model.JaxSimModel) -> SoftContactsState:
        """
        Build a zero `SoftContactsState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the soft contacts state.

        Returns:
            A zero `SoftContactsState` instance.
        """

        return SoftContactsState.build_from_jaxsim_model(model=model)

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `SoftContactsState` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `SoftContactsState` against.

        Returns:
            `True` if the soft contacts state is valid for the given `JaxSimModel`,
            `False` otherwise.
        """

        shape = self.tangential_deformation.shape
        expected = (len(model.kin_dyn_parameters.contact_parameters.body), 3)

        if shape != expected:
            return False

        return True
