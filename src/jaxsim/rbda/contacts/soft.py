from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp
from jaxsim.math import StandardGravity
from jaxsim.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams, ContactsState

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


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

    p: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    q: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray.hash_of_array(self.K),
                HashedNumpyArray.hash_of_array(self.D),
                HashedNumpyArray.hash_of_array(self.mu),
                HashedNumpyArray.hash_of_array(self.p),
                HashedNumpyArray.hash_of_array(self.q),
            )
        )

    def __eq__(self, other: SoftContactsParams) -> bool:

        if not isinstance(other, SoftContactsParams):
            return NotImplemented

        return hash(self) == hash(other)

    @classmethod
    def build(
        cls: type[Self],
        *,
        K: jtp.FloatLike = 1e6,
        D: jtp.FloatLike = 2_000,
        mu: jtp.FloatLike = 0.5,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
    ) -> Self:
        """
        Create a SoftContactsParams instance with specified parameters.

        Args:
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
            A SoftContactsParams instance with the specified parameters.
        """

        return SoftContactsParams(
            K=jnp.array(K, dtype=float),
            D=jnp.array(D, dtype=float),
            mu=jnp.array(mu, dtype=float),
            p=jnp.array(p, dtype=float),
            q=jnp.array(q, dtype=float),
        )

    @classmethod
    def build_default_from_jaxsim_model(
        cls: type[Self],
        model: js.model.JaxSimModel,
        *,
        standard_gravity: jtp.FloatLike = StandardGravity,
        static_friction_coefficient: jtp.FloatLike = 0.5,
        max_penetration: jtp.FloatLike = 0.001,
        number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
        damping_ratio: jtp.FloatLike = 1.0,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
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
            p:
                The exponent p corresponding to the damping-related non-linearity
                of the Hunt/Crossley model.
            q:
                The exponent q corresponding to the spring-related non-linearity
                of the Hunt/Crossley model

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

        return SoftContactsParams.build(K=K, D=D, mu=μc, p=p, q=q)

    def valid(self) -> jtp.BoolLike:
        """
        Check if the parameters are valid.

        Returns:
            `True` if the parameters are valid, `False` otherwise.
        """

        return jnp.hstack(
            [
                self.K >= 0.0,
                self.D >= 0.0,
                self.mu >= 0.0,
                self.p >= 0.0,
                self.q >= 0.0,
            ]
        ).all()


@jax_dataclasses.pytree_dataclass
class SoftContacts(ContactModel):
    """Soft contacts model."""

    parameters: SoftContactsParams = dataclasses.field(
        default_factory=SoftContactsParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def hunt_crossley_contact_model(
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

        # Convert the input vectors to arrays.
        W_p_C = jnp.array(position, dtype=float).squeeze()
        W_ṗ_C = jnp.array(velocity, dtype=float).squeeze()
        m = jnp.array(tangential_deformation, dtype=float).squeeze()

        # Use symbol for the static friction.
        μ = mu

        # Compute the penetration depth, its rate, and the considered terrain normal.
        δ, δ̇, n̂ = SoftContacts.compute_penetration_data(
            p=W_p_C, v=W_ṗ_C, terrain=terrain
        )

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
        # The ε, instead, is needed to make AD happy.
        f_tangential_direction = jnp.where(
            f_tangential.dot(f_tangential) != 0,
            f_tangential / jnp.linalg.norm(f_tangential + ε),
            jnp.zeros(3),
        )

        # Project the tangential force to the friction cone if slipping.
        f_tangential = jnp.where(
            sticking,
            f_tangential,
            jnp.minimum(μ * force_normal_mag, jnp.linalg.norm(f_tangential + ε))
            * f_tangential_direction,
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

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def compute_contact_force(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        parameters: SoftContactsParams,
        terrain: Terrain,
    ) -> tuple[jtp.Vector, jtp.Vector]:

        CW_fl, ṁ = SoftContacts.hunt_crossley_contact_model(
            position=position,
            velocity=velocity,
            tangential_deformation=tangential_deformation,
            terrain=terrain,
            K=parameters.K,
            D=parameters.D,
            mu=parameters.mu,
            p=parameters.p,
            q=parameters.q,
        )

        # Pack a mixed 6D force.
        CW_f = jnp.hstack([CW_fl, jnp.zeros(3)])

        # Compute the 6D force transform from the mixed to the inertial-fixed frame.
        W_Xf_CW = jaxsim.math.Adjoint.from_quaternion_and_translation(
            translation=jnp.array(position), inverse=True
        ).T

        # Compute the 6D force in the inertial-fixed frame.
        W_f = W_Xf_CW @ CW_f

        return W_f, ṁ

    @jax.jit
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
    ) -> tuple[jtp.Vector, tuple[jtp.Vector]]:

        # Initialize the model and data this contact model is operating on.
        # This will raise an exception if either the contact model or the
        # contact parameters are not compatible.
        model, data = self.initialize_model_and_data(model=model, data=data)

        # Compute the position and linear velocities (mixed representation) of
        # all collidable points belonging to the robot.
        W_p_C, W_ṗ_C = js.contact.collidable_point_kinematics(model=model, data=data)

        # Extract the material deformation corresponding to the collidable points.
        assert isinstance(data.state.contact, SoftContactsState)
        m = data.state.contact.tangential_deformation

        # Compute the contact forces for all collidable points.
        # Since we treat them as independent, we can vmap the computation.
        W_f, ṁ = jax.vmap(
            lambda p, v, m: SoftContacts.compute_contact_force(
                position=p,
                velocity=v,
                tangential_deformation=m,
                parameters=self.parameters,
                terrain=self.terrain,
            )
        )(W_p_C, W_ṗ_C, m)

        return W_f, (ṁ,)

    @staticmethod
    @jax.jit
    def compute_penetration_data(
        p: jtp.VectorLike,
        v: jtp.VectorLike,
        terrain: jaxsim.terrain.Terrain,
    ) -> tuple[jtp.Float, jtp.Float, jtp.Vector]:

        # Pre-process the position and the linear velocity of the collidable point.
        W_ṗ_C = jnp.array(v).squeeze()
        px, py, pz = jnp.array(p).squeeze()

        # Compute the terrain normal and the contact depth.
        n̂ = terrain.normal(x=px, y=py).squeeze()
        h = jnp.array([0, 0, terrain.height(x=px, y=py) - pz])

        # Compute the penetration depth normal to the terrain.
        δ = jnp.maximum(0.0, jnp.dot(h, n̂))

        # Compute the penetration normal velocity.
        δ̇ = -jnp.dot(W_ṗ_C, n̂)

        # Enforce the penetration rate to be zero when the penetration depth is zero.
        δ̇ = jnp.where(δ > 0, δ̇, 0.0)

        return δ, δ̇, n̂


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

    def __eq__(self: Self, other: Self) -> bool:

        if not isinstance(other, type(self)):
            return False

        return hash(self) == hash(other)

    @classmethod
    def build_from_jaxsim_model(
        cls: type[Self],
        model: js.model.JaxSimModel | None = None,
        tangential_deformation: jtp.MatrixLike | None = None,
    ) -> Self:
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

        return cls.build(
            tangential_deformation=tangential_deformation,
            number_of_collidable_points=len(
                model.kin_dyn_parameters.contact_parameters.body
            ),
        )

    @classmethod
    def build(
        cls: type[Self],
        *,
        tangential_deformation: jtp.MatrixLike | None = None,
        number_of_collidable_points: int | None = None,
    ) -> Self:
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
            jnp.atleast_2d(tangential_deformation)
            if tangential_deformation is not None
            else jnp.zeros(shape=(number_of_collidable_points, 3))
        ).astype(float)

        if tangential_deformation.shape[1] != 3:
            raise RuntimeError("The tangential deformation matrix must have 3 columns.")

        if (
            number_of_collidable_points is not None
            and tangential_deformation.shape[0] != number_of_collidable_points
        ):
            msg = "The number of collidable points must match the number of rows "
            msg += "in the tangential deformation matrix."
            raise RuntimeError(msg)

        return cls(tangential_deformation=tangential_deformation)

    @classmethod
    def zero(cls: type[Self], *, model: js.model.JaxSimModel) -> Self:
        """
        Build a zero `SoftContactsState` from a `JaxSimModel`.

        Args:
            model: The `JaxSimModel` associated with the soft contacts state.

        Returns:
            A zero `SoftContactsState` instance.
        """

        return cls.build_from_jaxsim_model(model=model)

    def valid(self, *, model: js.model.JaxSimModel) -> jtp.BoolLike:
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
