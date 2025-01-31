from __future__ import annotations

import abc
import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim.utils import JaxsimDataclass

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@functools.partial(jax.jit, static_argnames=("terrain",))
def compute_penetration_data(
    p: jtp.VectorLike,
    v: jtp.VectorLike,
    terrain: jaxsim.terrain.Terrain,
) -> tuple[jtp.Float, jtp.Float, jtp.Vector]:
    """
    Compute the penetration data (depth, rate, and terrain normal) of a collidable point.

    Args:
        p: The position of the collidable point.
        v:
            The linear velocity of the point (linear component of the mixed 6D velocity
            of the implicit frame `C = (W_p_C, [W])` associated to the point).
        terrain: The considered terrain.

    Returns:
        A tuple containing the penetration depth, the penetration velocity,
        and the considered terrain normal.
    """

    # Pre-process the position and the linear velocity of the collidable point.
    W_ṗ_C = jnp.array(v).squeeze()
    px, py, pz = jnp.array(p).squeeze()

    # Compute the terrain normal and the contact depth.
    n̂ = terrain.normal(x=px, y=py).squeeze()
    h = jnp.array([0, 0, terrain.height(x=px, y=py) - pz])

    # Compute the penetration depth normal to the terrain.
    δ = jnp.maximum(0.0, jnp.dot(h, n̂))

    # Compute the penetration normal velocity.
    δ_dot = -jnp.dot(W_ṗ_C, n̂)

    # Enforce the penetration rate to be zero when the penetration depth is zero.
    δ_dot = jnp.where(δ > 0, δ_dot, 0.0)

    return δ, δ_dot, n̂


class ContactsParams(JaxsimDataclass):
    """
    Abstract class representing the parameters of a contact model.

    Note:
        This class is supposed to store only the tunable parameters of the contact
        model, i.e. all those parameters that can be changed during runtime.
        If the contact model has also static parameters, they should be stored
        in the corresponding `ContactModel` class.
    """

    @classmethod
    @abc.abstractmethod
    def build(cls: type[Self], **kwargs) -> Self:
        """
        Create a `ContactsParams` instance with specified parameters.

        Returns:
            The `ContactsParams` instance.
        """
        pass

    @abc.abstractmethod
    def valid(self, **kwargs) -> jtp.BoolLike:
        """
        Check if the parameters are valid.

        Returns:
            True if the parameters are valid, False otherwise.
        """
        pass


class ContactModel(JaxsimDataclass):
    """
    Abstract class representing a contact model.
    """

    @classmethod
    @abc.abstractmethod
    def build(
        cls: type[Self],
        **kwargs,
    ) -> Self:
        """
        Create a `ContactModel` instance with specified parameters.

        Returns:
            The `ContactModel` instance.
        """

        pass

    @abc.abstractmethod
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        **kwargs,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the contact forces.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            **kwargs: Optional additional arguments, specific to the contact model.

        Returns:
            A tuple containing as first element the computed 6D contact force applied to
            the contact points and expressed in the world frame, and as second element
            a dictionary of optional additional information.
        """

        pass

    @classmethod
    def zero_state_variables(cls, model: js.model.JaxSimModel) -> dict[str, jtp.Array]:
        """
        Build zero state variables of the contact model.

        Args:
            model: The robot model considered by the contact model.

        Note:
            There are contact models that require to extend the state vector of the
            integrated ODE system with additional variables. Our integrators are
            capable of operating on a generic state, as long as it is a PyTree.
            This method builds the zero state variables of the contact model as a
            dictionary of JAX arrays.

        Returns:
            A dictionary storing the zero state variables of the contact model.
        """

        return {}

    @property
    def _parameters_class(cls) -> type[ContactsParams]:
        """
        Return the class of the contact parameters.

        Returns:
            The class of the contact parameters.
        """
        import importlib

        return getattr(
            importlib.import_module("jaxsim.rbda.contacts"),
            (
                cls.__name__ + "Params"
                if isinstance(cls, type)
                else cls.__class__.__name__ + "Params"
            ),
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def _hunt_crossley_contact_model(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        terrain: jaxsim.terrain.Terrain,
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
        δ, δ̇, n̂ = compute_penetration_data(p=W_p_C, v=W_ṗ_C, terrain=terrain)

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
