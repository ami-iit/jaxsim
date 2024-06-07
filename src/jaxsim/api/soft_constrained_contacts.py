from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxopt

import jaxsim.typing as jtp
from jaxsim.terrain import FlatTerrain, Terrain

from . import link
from .contact import ContactModel, ContactParams, ContactsState
from .data import JaxSimModelData
from .model import (
    JaxSimModel,
    free_floating_mass_matrix,
    generalized_free_floating_jacobian,
    link_bias_accelerations,
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
    dmin: float = jax_dataclasses.field(
        default_factory=lambda: jnp.array(0.0, dtype=float)
    )
    dmax: float = jax_dataclasses.field(
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

    @classmethod
    def build(
        cls,
        timeconst: float,
        dampratio: float,
        dmin: float,
        dmax: float,
        width: float,
        mid: float,
        power: float,
        friction: float,
    ) -> ConstrainedContactsParams:
        """
        Create a ConstrainedContactsParams instance with specified parameters.

        Args:
            timeconst: The time constant.
            dampratio: The damping ratio.
            dmin: The minimum damping.
            dmax: The maximum damping.
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
        dmin = (
            jnp.array(dmin, dtype=float)
            if dmin is not None
            else cls.__dataclass_fields__["dmin"].default
        )
        dmax = (
            jnp.array(dmax, dtype=float)
            if dmax is not None
            else cls.__dataclass_fields__["dmax"].default
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

        return cls(
            timeconst=timeconst,
            dampratio=dampratio,
            dmin=dmin,
            dmax=dmax,
            width=width,
            mid=mid,
            power=power,
            friction=friction,
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
                self.dmin,
                self.dmax,
                self.width,
                self.mid,
                self.power,
                self.friction,
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

        τ_constraints = jnp.zeros(shape=(model.dofs(),))

        def _imp_aref(
            position: jtp.Array, velocity: jtp.Array
        ) -> tuple[jtp.Array, jtp.Array]:
            """Calculates impedance and offset acceleration in constraint frame.

            Args:
                position: position in constraint frame
                velocity: velocity in constraint frame

            Returns:
                impedance: constraint impedance
                a_ref: offset acceleration in constraint frame
            """
            # Check https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
            timeconst, ζ, dmin, dmax, width, mid, p, _ = self.parameters

            imp_x = jnp.abs(position) / width
            imp_a = (1.0 / jnp.power(mid, p - 1)) * jnp.power(imp_x, p)

            imp_b = 1 - (1.0 / jnp.power(1 - mid, p - 1)) * jnp.power(1 - imp_x, p)

            imp_y = jnp.where(imp_x < mid, imp_a, imp_b)

            imp = jnp.clip(dmin + imp_y * (dmax - dmin), dmin, dmax)
            imp = jnp.where(imp_x > 1.0, dmax, imp)

            D = 2 / (dmax * timeconst)
            K = 1 / (dmax * timeconst * ζ) ** 2

            a_ref = -D * velocity - K * imp * position

            return imp, a_ref

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
                J = generalized_free_floating_jacobian(model=model, data=data)

                # Compute the reference acceleration.
                imp, a_ref = _imp_aref(position=position, velocity=velocity)

                # Compute the regularization terms.
                R = (
                    (2 * self.parameters.friction**2 * (1 - imp) / (imp + 1e-8))
                    * (1 + self.parameters.friction**2)
                    / link.mass(model=model, link_index=link_idx)
                )

                # TODO: Compute the smooth contact force
                qf_smooth = jnp.atleast_1d(tau - τ_constraints)

                return jax.tree.map(
                    lambda x: x * (position < 0), (J, a_ref, R, qf_smooth)
                )

            J, a_ref, R, qf_smooth = jax.tree.map(
                jnp.concatenate,
                jax.vmap(_compute_row)(
                    jnp.array(model.kin_dyn_parameters.contact_parameters.body),
                    δ,
                    δ̇,
                ),
            )
            return J, a_ref, R, qf_smooth

        δ, δ̇ = position[:, 2], velocity[:, 2]

        J, a_ref, r, qf_smooth = _contact_jacobian(model=model, data=data)

        R = jnp.diag(r)

        M_inv = jnp.linalg.inv(free_floating_mass_matrix(model=model, data=data))

        reference_acc = jnp.concatenate(
            jax.vmap(lambda ref: (jnp.zeros(shape=(6,)).at[2].set(ref)))(a_ref)
        )

        # Calculate quantities for the linear optimization problem.
        A = J @ M_inv @ J.T
        b = (
            J @ M_inv @ qf_smooth
            + link_bias_accelerations(model=model, data=data)[
                jnp.array(list(set(model.kin_dyn_parameters.contact_parameters.body)))
            ]
            - reference_acc
        )  # ? - qf_smooth

        objective = lambda x: jnp.sum(0.5 * (A @ x + b) ** 2)

        # Compute the 3D linear force in C[W] frame
        opt = jaxopt.ProjectedGradient(
            fun=objective,
            projection=jaxopt.projection.projection_non_negative,
            maxiter=8,
            implicit_diff=False,
            maxls=5,
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
