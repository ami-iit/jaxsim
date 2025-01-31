from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import STANDARD_GRAVITY
from jaxsim.terrain import Terrain

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
        standard_gravity: jtp.FloatLike = STANDARD_GRAVITY,
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
        # Note that this is dependent on the non-linear exponent used in
        # the damping term of the Hunt/Crossley model.
        K = f_average / jnp.power(δ_max, 1 + p)

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
class SoftContacts(common.ContactModel):
    """Soft contacts model."""

    @classmethod
    def build(
        cls: type[Self],
        model: js.model.JaxSimModel | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a `SoftContacts` instance with specified parameters.

        Args:
            model:
                The robot model considered by the contact model.
                If passed, it is used to estimate good default parameters.
            **kwargs: Additional parameters to pass to the contact model.

        Returns:
            The `SoftContacts` instance.
        """

        if len(kwargs) != 0:
            logging.debug(msg=f"Ignoring extra arguments: {kwargs}")

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

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def compute_contact_force(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        parameters: SoftContactsParams,
        terrain: Terrain,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact force.

        Args:
            position: The position of the collidable point.
            velocity: The velocity of the collidable point.
            tangential_deformation: The material deformation of the collidable point.
            parameters: The parameters of the soft contacts model.
            terrain: The terrain model.

        Returns:
            A tuple containing the computed contact force and the derivative of the
            material deformation.
        """

        CW_fl, ṁ = common.ContactModel._hunt_crossley_contact_model(
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

        # Get the indices of the enabled collidable points.
        indices_of_enabled_collidable_points = (
            model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
        )

        # Compute the position and linear velocities (mixed representation) of
        # all the collidable points belonging to the robot and extract the ones
        # for the enabled collidable points.
        W_p_C, W_ṗ_C = js.contact.collidable_point_kinematics(model=model, data=data)

        # Extract the material deformation corresponding to the collidable points.
        m = (
            data.contact_state["tangential_deformation"]
            if "tangential_deformation" in data.contact_state
            else jnp.zeros_like(W_p_C)
        )

        m_enabled = m[indices_of_enabled_collidable_points]

        # Initialize the tangential deformation rate array for every collidable point.
        ṁ = jnp.zeros_like(m)

        # Compute the contact forces only for the enabled collidable points.
        # Since we treat them as independent, we can vmap the computation.
        W_f, ṁ_enabled = jax.vmap(
            lambda p, v, m: SoftContacts.compute_contact_force(
                position=p,
                velocity=v,
                tangential_deformation=m,
                parameters=model.contact_params,
                terrain=model.terrain,
            )
        )(W_p_C, W_ṗ_C, m_enabled)

        ṁ = ṁ.at[indices_of_enabled_collidable_points].set(ṁ_enabled)

        return W_f, dict(m_dot=ṁ)
