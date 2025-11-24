from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp
from jaxsim import logging

from . import common

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class SoftContactsParams(common.ContactsParams):
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
            return False

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
        **kwargs,
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
            **kwargs: Additional parameters to pass to the contact model.

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
class SoftContacts(common.ContactModel):
    """Soft contacts model."""

    @classmethod
    def build(
        cls: type[Self],
        **kwargs,
    ) -> Self:
        """
        Create a `SoftContacts` instance with specified parameters.

        Args:
            **kwargs: Additional parameters to pass to the contact model.

        Returns:
            The `SoftContacts` instance.
        """

        if len(kwargs) != 0:
            logging.warning(msg=f"Ignoring extra arguments: {kwargs}")

        return cls(**kwargs)

    @classmethod
    def zero_state_variables(cls, model: js.model.JaxSimModel) -> dict[str, jtp.Array]:
        """
        Build zero state variables of the contact model.
        """

        # Initialize the material deformation to zero.
        tangential_deformation = jnp.zeros(
            shape=(len(model.kin_dyn_parameters.contact_parameters.body), 3),
            dtype=float,
        )

        return {"tangential_deformation": tangential_deformation}

    def update_contact_state(
        self: type[Self], old_contact_state: dict[str, jtp.Array]
    ) -> dict[str, jtp.Array]:
        """
        Update the contact state.

        Args:
            old_contact_state: The old contact state.

        Returns:
            The updated contact state.
        """

        return {"tangential_deformation": old_contact_state["m_dot"]}

    def update_velocity_after_impact(
        self: type[Self], model: js.model.JaxSimModel, data: js.data.JaxSimModelData
    ) -> js.data.JaxSimModelData:
        """
        Update the velocity after an impact.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.

        Returns:
            The updated data of the considered model.
        """

        return data

    @staticmethod
    @jax.jit
    def hunt_crossley_contact_model(
        penetration: jtp.VectorLike,
        penetration_rate: jtp.VectorLike,
        velocity: jtp.VectorLike,
        normal: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        K: jtp.FloatLike,
        D: jtp.FloatLike,
        mu: jtp.FloatLike,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact force using the Hunt/Crossley model.

        Args:
            penetration: The penetration of the collision point.
            penetration_rate: The penetration rate of the collision point.
            velocity: The velocity of the contact point.
            normal: The terrain normal at the contact point.
            tangential_deformation: The material deformation of the collidable shape.
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

        # Use symbols for input parameters.
        W_ṗ_C = velocity
        m = tangential_deformation
        δ = penetration
        δ̇ = penetration_rate
        n̂ = normal
        μ = mu

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

    @staticmethod
    @jax.jit
    def compute_contact_force(
        penetration: jtp.Float,
        penetration_rate: jtp.Float,
        position: jtp.Vector,
        velocity: jtp.Vector,
        normal: jtp.Vector,
        tangential_deformation: jtp.Vector,
        parameters: SoftContactsParams,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact force.

        Args:
            penetration: The penetration of the collision point.
            penetration_rate: The penetration rate of the collision point.
            position: The position of the contact point.
            velocity: The velocity of the contact point.
            normal: The terrain normal at the contact point.
            tangential_deformation: The material deformation of the collidable shape.
            parameters: The parameters of the soft contacts model.

        Returns:
            A tuple containing the computed contact force and the derivative of the
            material deformation.
        """

        CW_fl, ṁ = jax.vmap(
            SoftContacts.hunt_crossley_contact_model,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None),
        )(
            penetration,
            penetration_rate,
            velocity,
            normal,
            tangential_deformation,
            parameters.K,
            parameters.D,
            parameters.mu,
            parameters.p,
            parameters.q,
        )

        # Pack a mixed 6D force.
        CW_f = jax.vmap(lambda f: jnp.hstack([f, jnp.zeros(3)]))(f=CW_fl)

        # Compute the 6D force transform from the mixed to the inertial-fixed frame.
        W_Xf_CW = jax.vmap(
            lambda W_p_C: jaxsim.math.Adjoint.from_quaternion_and_translation(
                translation=jnp.array(W_p_C), inverse=True
            ).T
        )(W_p_C=position)

        # Compute the 6D force in the inertial-fixed frame.
        W_f = jnp.einsum("...ij,...j->...i", W_Xf_CW, CW_f)

        return jnp.sum(W_f, axis=0), jnp.mean(ṁ, axis=0)

    @staticmethod
    @jax.jit
    def compute_contact_forces(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the contact forces.

        Args:
            model: The model to consider.
            data: The data of the considered model.

        Returns:
            A tuple containing as first element the computed contact forces, and as
            second element a dictionary with derivative of the material deformation.
        """

        # Compute the position and linear velocities (mixed representation) of
        # all the collidable shapes belonging to the robot and extract the ones
        # for the enabled collidable shapes.
        δ, δ̇, n̂, W_p_C, CW_ṗ_C = jax.vmap(
            lambda shape_transform, shape_type, shape_size, link_transform, link_velocity: common.compute_penetration_data(
                model,
                shape_transform=shape_transform,
                shape_type=shape_type,
                shape_size=shape_size,
                link_transforms=link_transform,
                link_velocities=link_velocity,
            )
        )(
            model.kin_dyn_parameters.contact_parameters.transform,
            model.kin_dyn_parameters.contact_parameters.shape_type,
            model.kin_dyn_parameters.contact_parameters.shape_size,
            data._link_transforms[
                np.array(model.kin_dyn_parameters.contact_parameters.body)
            ],
            data._link_velocities[
                np.array(model.kin_dyn_parameters.contact_parameters.body)
            ],
        )

        # Extract the material deformation corresponding to the collidable shapes.
        m = data.contact_state["tangential_deformation"]

        # Initialize the tangential deformation rate array for every collidable shape.
        ṁ = jnp.zeros_like(m)

        # Compute the contact forces for all the collidable shapes.
        # Since we treat them as independent, we can vmap the computation.
        # We exploit two levels of vmap to vectorize over both the shapes and the points.
        # The outer vmap vectorizes over the shapes, while the inner vmap vectorizes
        # over the maximum points (3) belonging to each shape.
        W_f_per_shape, ṁ = jax.vmap(
            SoftContacts.compute_contact_force,
            in_axes=(0, 0, 0, 0, 0, 0, None),  # vectorize over shapes
        )(δ, δ̇, W_p_C, CW_ṗ_C, n̂, m, model.contact_params)

        # Accumulate forces by parent link using segment_sum
        body_indices = jnp.array(model.kin_dyn_parameters.contact_parameters.body)
        W_f = jax.ops.segment_sum(
            W_f_per_shape, body_indices, num_segments=model.number_of_links()
        )

        return W_f, {"m_dot": ṁ}
