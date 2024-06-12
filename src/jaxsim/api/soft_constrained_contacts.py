from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxopt

import jaxsim.typing as jtp
from jaxsim.math import Adjoint
from jaxsim.terrain import FlatTerrain, Terrain

from . import link
from .contact import ContactModel, ContactParams, ContactsState
from .data import JaxSimModelData
from .model import (
    JaxSimModel,
    free_floating_bias_forces,
    free_floating_mass_matrix,
    generalized_free_floating_jacobian,
)


@jax_dataclasses.pytree_dataclass
class ConstrainedContactsParams(ContactParams):
    """Parameters of the constrained contacts model."""

    timeconst: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    dampratio: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    d_min: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )
    d_max: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )
    width: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.1, dtype=float)
    )
    mid: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    power: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(1.0, dtype=float)
    )
    friction: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )
    stiffness: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )
    damping: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )

    @classmethod
    def build(
        cls,
        timeconst: float,
        dampratio: float,
        d_min: float,
        d_max: float,
        width: float,
        mid: float,
        power: float,
        friction: float,
        stiffness: float,
        damping: float,
    ) -> ConstrainedContactsParams:
        """
        Create a ConstrainedContactsParams instance with specified parameters.

        Args:
            timeconst: The time constant.
            dampratio: The damping ratio.
            d_min: The minimum damping.
            d_max: The maximum damping.
            width: The width of the damping function.
            mid: The mid value of the damping function.
            power: The power of the damping function.
            friction: The dynamic friction coefficient.
        Returns:
            ConstrainedContactsParams: A ConstrainedContactsParams instance with the specified parameters.
        """
        # Fallback to default dataclasses values
        timeconst = (
            jnp.array(timeconst, dtype=float)
            if timeconst is not None
            else cls.__dataclass_fields__["timeconst"].default
        )
        dampratio = (
            jnp.array(dampratio, dtype=float)
            if dampratio is not None
            else cls.__dataclass_fields__["dampratio"].default
        )
        d_min = (
            jnp.array(d_min, dtype=float)
            if d_min is not None
            else cls.__dataclass_fields__["d_min"].default
        )
        d_max = (
            jnp.array(d_max, dtype=float)
            if d_max is not None
            else cls.__dataclass_fields__["d_max"].default
        )
        width = (
            jnp.array(width, dtype=float)
            if width is not None
            else cls.__dataclass_fields__["width"].default
        )
        mid = (
            jnp.array(mid, dtype=float)
            if mid is not None
            else cls.__dataclass_fields__["mid"].default
        )
        power = (
            jnp.array(power, dtype=float)
            if power is not None
            else cls.__dataclass_fields__["power"].default
        )
        friction = (
            jnp.array(friction, dtype=float)
            if friction is not None
            else cls.__dataclass_fileds["friction"].default
        )
        stiffness = (
            jnp.array(stiffness, dtype=float)
            if stiffness is not None
            else cls.__dataclass_fields__["stiffness"].default
        )
        damping = (
            jnp.array(damping, dtype=float)
            if damping is not None
            else cls.__dataclass_fields__["damping"].default
        )

        return cls(
            timeconst=timeconst,
            dampratio=dampratio,
            d_min=d_min,
            d_max=d_max,
            width=width,
            mid=mid,
            power=power,
            friction=friction,
            stiffness=stiffness,
            damping=damping,
        )

    @classmethod
    def build_default_from_jaxsim_model(
        cls,
        model: JaxSimModel,
        *args,
        **kwargs,
    ) -> ConstrainedContactsParams:
        """
        Create a ConstrainedContactsParams instance.

        Args:
            model: The target model.

        Returns:
            A `ConstrainedContactsParams` instance with the specified parameters.
        """
        return cls.build()

    def __iter__(self):
        return iter(
            [
                self.timeconst,
                self.dampratio,
                self.d_min,
                self.d_max,
                self.width,
                self.mid,
                self.power,
                self.friction,
                self.stiffness,
                self.damping,
            ]
        )


@jax_dataclasses.pytree_dataclass
class ConstrainedContacts(ContactModel):
    """Constrained contacts model."""

    parameters: ConstrainedContactsParams = jax_dataclasses.field(
        default_factory=ConstrainedContactsParams
    )

    terrain: Terrain = jax_dataclasses.field(default_factory=FlatTerrain)

    def contact_model(
        self,
        position: jtp.Vector,
        velocity: jtp.Vector,
        model: JaxSimModel,
        data: jtp.Vector,
        tau: jtp.Vector | None = None,
        tangential_deformation: jtp.Vector | None = None,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces.

        Args:
            model (JaxSimModel): The jaxsim model.
            position (jtp.Vector): The position of the collidable point.
            velocity (jtp.Vector): The linear velocity of the collidable point.
            tangential_deformation (jtp.Vector, optional): The tangential deformation. Defaults to None.

        Returns:
            tuple[jtp.Vector, jtp.Vector]: A tuple containing the contact force and material deformation rate.
        """
        # Extract the parameters.
        (
            timeconst,
            ζ,
            ξ_min,
            ξ_max,
            width,
            m,
            p,
            μ,
            stiffness,
            damping,
        ) = self.parameters

        def _imp_aref(
            position: jtp.Array, velocity: jtp.Array
        ) -> tuple[jtp.Array, jtp.Array]:
            """
            Calculates impedance and offset acceleration in constraint frame.

            Args:
                position: position in constraint frame
                velocity: velocity in constraint frame

            Returns:
                ξ: constraint impedance
                a_ref: offset acceleration in constraint frame
            """

            ξ_x = jnp.abs(position) / width
            ξ_a = (1.0 / jnp.power(m, p - 1)) * jnp.power(ξ_x, p)

            ξ_b = 1 - (1.0 / jnp.power(1 - m, p - 1)) * jnp.power(1 - ξ_x, p)

            ξ_y = jnp.where(ξ_x < m, ξ_a, ξ_b)

            ξ = jnp.clip(ξ_min + ξ_y * (ξ_max - ξ_min), ξ_min, ξ_max)
            ξ = jnp.atleast_1d(jnp.where(ξ_x > 1.0, ξ_max, ξ))

            # When passing negative values, K and D represent spring and damper constants, respectively.
            K = jnp.where(
                stiffness < 0, -stiffness / ξ_max**2, 1 / (ξ_max * timeconst * ζ) ** 2
            )
            D = jnp.where(damping < 0, -damping / ξ_max, 2 / (ξ_max * timeconst))

            a_ref = jnp.atleast_1d(D * velocity + K * ξ * position)

            return ξ, a_ref

        def _contact_jacobian(model: JaxSimModel, data: JaxSimModelData) -> tuple:
            """Compute the contact jacobian and the reference acceleration.

            Args:
                model (JaxSimModel): The jaxsim model.
                position (jtp.Vector): The position of the collidable point.

            Returns:
                tuple: A tuple containing the contact jacobian, the reference acceleration, and the contact radius.
            """

            def _compute_row(link_idx: jtp.Float):
                # Compute the contact jacobian.
                L_Xv_C = Adjoint.from_rotation_and_translation(
                    rotation=jnp.eye(3),
                    translation=model.kin_dyn_parameters.contact_parameters.point[
                        link_idx
                    ],
                ).T

                J = (
                    L_Xv_C
                    @ generalized_free_floating_jacobian(model=model, data=data)[
                        link_idx
                    ]
                )[:3]

                # Compute the reference acceleration.
                imp, a_ref = _imp_aref(position=position, velocity=velocity)

                # Compute the regularization terms.
                R = (
                    (2 * self.parameters.friction**2 * (1 - imp) / (imp + 1e-8))
                    * (1 + self.parameters.friction**2)
                    / link.mass(model=model, link_index=link_idx)
                )

                return jax.tree.map(lambda x: x * (position < 0), (J, a_ref, R))

            J, a_ref, R = jax.tree.map(
                jnp.concatenate,
                jax.vmap(_compute_row)(
                    jnp.array(model.kin_dyn_parameters.contact_parameters.body),
                    δ,
                    δ̇,
                ),
            )
            return J, a_ref, R

        τ_constraints = jnp.zeros_like(data.joint_positions())
        S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T
        h = free_floating_bias_forces(model=model, data=data)

        δ, δ̇ = position[:, 2], velocity[:, 2]

        J, a_ref, r = _contact_jacobian(model=model, data=data)

        R = jnp.diag(r)

        # Compute the smooth contact force.
        qf_smooth = S @ (jnp.atleast_1d(tau - τ_constraints)) - h

        M_inv = jnp.linalg.inv(free_floating_mass_matrix(model=model, data=data))

        reference_acc = jnp.concatenate(
            jax.vmap(lambda ref: (jnp.zeros(shape=(3,)).at[2].set(ref)))(a_ref)
        )

        # Calculate quantities for the linear optimization problem.
        A = J @ M_inv @ J.T + R
        b = J @ M_inv @ qf_smooth + jnp.hstack(velocity) - a_ref

        objective = lambda x: jnp.sum(0.5 * (A @ x + b) ** 2)

        # Compute the 3D linear force in C[W] frame
        opt = jaxopt.ProjectedGradient(
            fun=objective,
            projection=jaxopt.projection.projection_non_negative,
            maxiter=100,
            implicit_diff=False,
            maxls=20,
        )

        F = J.T @ opt.run(jnp.zeros_like(b)).params

        return F, None


@jax_dataclasses.pytree_dataclass
class ConstrainedContactsState(ContactsState):
    """
    Class storing the state of the constrained contacts model.
    """

    @staticmethod
    def build(model: JaxSimModel | None = None, **kwargs) -> ConstrainedContactsState:
        return ConstrainedContactsState()

    @staticmethod
    def build_from_jaxsim_model(
        model: JaxSimModel,
        **kwargs,
    ) -> ConstrainedContactsState:
        """
        Create a ConstrainedContactsState instance.

        Args:
            model: The target model.

        Returns:
            A `ConstrainedContactsState` instance with the specified parameters.
        """
        return ConstrainedContactsState.build()

    @staticmethod
    def zero(model: JaxSimModel) -> ConstrainedContactsState:
        return ConstrainedContactsState.build_from_jaxsim_model(model=model)
