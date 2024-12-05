from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim
import jaxsim.api as js
import jaxsim.exceptions
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.api.common import ModelDataWithVelocityRepresentation
from jaxsim.math import StandardGravity
from jaxsim.terrain import Terrain

from . import common
from .soft import SoftContacts, SoftContactsParams

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class ViscoElasticContactsParams(common.ContactsParams):
    """Parameters of the visco-elastic contacts model."""

    K: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(1e6, dtype=float)
    )

    D: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(2000, dtype=float)
    )

    static_friction: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    p: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    q: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array(0.5, dtype=float)
    )

    @classmethod
    def build(
        cls: type[Self],
        K: jtp.FloatLike = 1e6,
        D: jtp.FloatLike = 2_000,
        static_friction: jtp.FloatLike = 0.5,
        p: jtp.FloatLike = 0.5,
        q: jtp.FloatLike = 0.5,
    ) -> Self:
        """
        Create a SoftContactsParams instance with specified parameters.

        Args:
            K: The stiffness parameter.
            D: The damping parameter of the soft contacts model.
            static_friction: The static friction coefficient.
            p:
                The exponent p corresponding to the damping-related non-linearity
                of the Hunt/Crossley model.
            q:
                The exponent q corresponding to the spring-related non-linearity
                of the Hunt/Crossley model.

        Returns:
            A ViscoElasticParams instance with the specified parameters.
        """

        return ViscoElasticContactsParams(
            K=jnp.array(K, dtype=float),
            D=jnp.array(D, dtype=float),
            static_friction=jnp.array(static_friction, dtype=float),
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
    ) -> Self:
        """
        Create a ViscoElasticContactsParams instance with good default parameters.

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
                of the Hunt/Crossley model.

        Returns:
            A `ViscoElasticContactsParams` instance with the specified parameters.

        Note:
            The `damping_ratio` parameter allows to operate on the following conditions:
            - ξ > 1.0: over-damped
            - ξ = 1.0: critically damped
            - ξ < 1.0: under-damped
        """

        # Call the SoftContact builder instead of duplicating the logic.
        soft_contacts_params = SoftContactsParams.build_default_from_jaxsim_model(
            model=model,
            standard_gravity=standard_gravity,
            static_friction_coefficient=static_friction_coefficient,
            max_penetration=max_penetration,
            number_of_active_collidable_points_steady_state=number_of_active_collidable_points_steady_state,
            damping_ratio=damping_ratio,
        )

        return ViscoElasticContactsParams.build(
            K=soft_contacts_params.K,
            D=soft_contacts_params.D,
            static_friction=soft_contacts_params.mu,
            p=p,
            q=q,
        )

    def valid(self) -> jtp.BoolLike:
        """
        Check if the parameters are valid.

        Returns:
            `True` if the parameters are valid, `False` otherwise.
        """

        return (
            jnp.all(self.K >= 0.0)
            and jnp.all(self.D >= 0.0)
            and jnp.all(self.static_friction >= 0.0)
            and jnp.all(self.p >= 0.0)
            and jnp.all(self.q >= 0.0)
        )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray.hash_of_array(self.K),
                HashedNumpyArray.hash_of_array(self.D),
                HashedNumpyArray.hash_of_array(self.static_friction),
                HashedNumpyArray.hash_of_array(self.p),
                HashedNumpyArray.hash_of_array(self.q),
            )
        )

    def __eq__(self, other: ViscoElasticContactsParams) -> bool:

        if not isinstance(other, ViscoElasticContactsParams):
            return False

        return hash(self) == hash(other)


@jax_dataclasses.pytree_dataclass
class ViscoElasticContacts(common.ContactModel):
    """Visco-elastic contacts model."""

    max_squarings: jax_dataclasses.Static[int] = dataclasses.field(default=25)

    @classmethod
    def build(
        cls: type[Self],
        model: js.model.JaxSimModel | None = None,
        max_squarings: jtp.IntLike | None = None,
        **kwargs,
    ) -> Self:
        """
        Create a `ViscoElasticContacts` instance with specified parameters.

        Args:
            model:
                The robot model considered by the contact model.
                If passed, it is used to estimate good default parameters.
            max_squarings:
                The maximum number of squarings performed in the matrix exponential.

        Returns:
            The `ViscoElasticContacts` instance.
        """

        if len(kwargs) != 0:
            logging.debug(msg=f"Ignoring extra arguments: {kwargs}")

        return cls(
            max_squarings=int(
                max_squarings
                if max_squarings is not None
                else cls.__dataclass_fields__["max_squarings"].default
            ),
        )

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

    @jax.jit
    def compute_contact_forces(
        self,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        dt: jtp.FloatLike | None = None,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
    ) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
        """
        Compute the contact forces.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            dt: The time step to consider. If not specified, it is read from the model.
            link_forces:
                The 6D forces to apply to the links expressed in the frame corresponding
                to the velocity representation of `data`.
            joint_force_references: The joint force references to apply.

        Note:
            This contact model, contrarily to most other contact models, requires the
            knowledge of the integration step. It is not straightforward to assess how
            this contact model behaves when used with high-order Runge-Kutta schemes.
            For the time being, it is recommended to use a simple forward Euler scheme.
            The main benefit of this model is that the stiff contact dynamics is computed
            separately from the rest of the system dynamics, which allows to use simple
            integration schemes without altering significantly the simulation stability.

        Returns:
            A tuple containing as first element the computed 6D contact force applied to
            the contact point and expressed in the world frame, and as second element
            a dictionary of optional additional information.
        """

        # Extract the indices corresponding to the enabled collidable points.
        indices_of_enabled_collidable_points = (
            model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
        )

        # Initialize the time step.
        dt = dt if dt is not None else model.time_step

        # Compute the average contact linear forces in mixed representation by
        # integrating the contact dynamics in the continuous time domain.
        CW_f̅l, CW_fl̿, m_tf = (
            ViscoElasticContacts._compute_contact_forces_with_exponential_integration(
                model=model,
                data=data,
                dt=jnp.array(dt).astype(float),
                link_forces=link_forces,
                joint_force_references=joint_force_references,
                indices_of_enabled_collidable_points=indices_of_enabled_collidable_points,
                max_squarings=self.max_squarings,
            )
        )

        # ============================================
        # Compute the inertial-fixed 6D contact forces
        # ============================================

        # Compute the transforms of the mixed frames `C[W] = (W_p_C, [W])`
        # associated to each collidable point.
        W_H_C = js.contact.transforms(model=model, data=data)[
            indices_of_enabled_collidable_points, :, :
        ]

        # Vmapped transformation from mixed to inertial-fixed representation.
        compute_forces_inertial_fixed_vmap = jax.vmap(
            lambda CW_fl_C, W_H_C: (
                ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                    array=jnp.zeros(6).at[0:3].set(CW_fl_C),
                    other_representation=jaxsim.VelRepr.Mixed,
                    transform=W_H_C,
                    is_force=True,
                )
            )
        )

        # Express the linear contact forces in the inertial-fixed frame.
        W_f̅_C, W_f̿_C = jax.vmap(
            lambda CW_fl: compute_forces_inertial_fixed_vmap(CW_fl, W_H_C)
        )(jnp.stack([CW_f̅l, CW_fl̿]))

        return W_f̅_C, dict(W_f_avg2_C=W_f̿_C, m_tf=m_tf)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("max_squarings",))
    def _compute_contact_forces_with_exponential_integration(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        dt: jtp.FloatLike,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
        indices_of_enabled_collidable_points: jtp.VectorLike | None = None,
        max_squarings: int = 25,
    ) -> tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix]:
        """
        Compute the average contact forces by integrating the contact dynamics.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            dt: The integration time step.
            link_forces: The 6D forces to apply to the links.
            joint_force_references: The joint force references to apply.
            indices_of_enabled_collidable_points:
                The indices of the enabled collidable points.
            max_squarings:
                The maximum number of squarings performed in the matrix exponential.

        Returns:
            A tuple containing:
            - The average contact forces.
            - The average of the average contact forces.
            - The tangential deformation at the final state.
        """

        # ==========================
        # Populate missing arguments
        # ==========================

        indices = (
            indices_of_enabled_collidable_points
            if indices_of_enabled_collidable_points is not None
            else jnp.arange(
                len(model.kin_dyn_parameters.contact_parameters.body)
            ).astype(int)
        )

        # ==================================
        # Compute the contact point dynamics
        # ==================================

        p_t0, v_t0 = js.contact.collidable_point_kinematics(model, data)
        m_t0 = data.state.extended["tangential_deformation"][indices, :]

        p_t0 = p_t0[indices, :]
        v_t0 = v_t0[indices, :]

        # Compute the linearized contact dynamics.
        # Note that it linearizes the (non-linear) contact model at (p, v, m)[t0].
        A, b, A_sc, b_sc = ViscoElasticContacts._contact_points_dynamics(
            model=model,
            data=data,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
            indices_of_enabled_collidable_points=indices,
            p_t0=p_t0,
            v_t0=v_t0,
            m_t0=m_t0,
        )

        # =============================================
        # Compute the integrals of the contact dynamics
        # =============================================

        # Pack the initial state of the contact points.
        x_t0 = jnp.hstack([p_t0.flatten(), v_t0.flatten(), m_t0.flatten()])

        # Pack the augmented matrix used to compute the single and double integral
        # of the exponential integration.
        A̅ = jnp.vstack(
            [
                jnp.hstack(
                    [
                        A,
                        jnp.vstack(b),
                        jnp.vstack(x_t0),
                        jnp.vstack(jnp.zeros_like(x_t0)),
                    ]
                ),
                jnp.hstack([jnp.zeros(A.shape[1]), 0, 1, 0]),
                jnp.hstack([jnp.zeros(A.shape[1]), 0, 0, 1]),
                jnp.hstack([jnp.zeros(A.shape[1]), 0, 0, 0]),
            ]
        )

        # Compute the matrix exponential.
        exp_tA = jax.scipy.linalg.expm(
            (dt * A̅).astype(float), max_squarings=max_squarings
        )

        # Integrate the contact dynamics in the continuous time domain.
        x_int, x_int2 = (
            jnp.hstack([jnp.eye(A.shape[0]), jnp.zeros(shape=(A.shape[0], 3))])
            @ exp_tA
            @ jnp.vstack([jnp.zeros(shape=(A.shape[0] + 1, 2)), jnp.eye(2)])
        ).T

        jaxsim.exceptions.raise_runtime_error_if(
            condition=jnp.isnan(x_int).any(),
            msg="NaN integration, try to increase `max_squarings` or decreasing `dt`",
        )

        # ==========================
        # Compute the contact forces
        # ==========================

        # Compute the average contact forces.
        CW_f̅, _ = jnp.split(
            (A_sc @ x_int / dt + b_sc).reshape(-1, 3),
            indices_or_sections=2,
        )

        # Compute the average of the average contact forces.
        CW_f̿, _ = jnp.split(
            (A_sc @ x_int2 * 2 / (dt**2) + b_sc).reshape(-1, 3),
            indices_or_sections=2,
        )

        # Extract the tangential deformation at the final state.
        x_tf = x_int / dt
        m_tf = jnp.split(x_tf, 3)[2].reshape(-1, 3)

        return CW_f̅, CW_f̿, m_tf

    @staticmethod
    @jax.jit
    def _contact_points_dynamics(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
        indices_of_enabled_collidable_points: jtp.VectorLike | None = None,
        p_t0: jtp.MatrixLike | None = None,
        v_t0: jtp.MatrixLike | None = None,
        m_t0: jtp.MatrixLike | None = None,
    ) -> tuple[jtp.Matrix, jtp.Vector, jtp.Matrix, jtp.Vector]:
        """
        Compute the dynamics of the contact points.

        Note:
            This function projects the system dynamics to the contact space and
            returns the matrices of a linear system to simulate its evolution.
            Since the active contact model can be non-linear, this function also
            linearizes the contact model at the initial state.

        Args:
            model: The robot model considered by the contact model.
            data: The data of the considered model.
            link_forces: The 6D forces to apply to the links.
            joint_force_references: The joint force references to apply.
            indices_of_enabled_collidable_points:
                The indices of the enabled collidable points.
            p_t0: The initial position of the collidable points.
            v_t0: The initial velocity of the collidable points.
            m_t0: The initial tangential deformation of the collidable points.

        Returns:
            A tuple containing:
            - The `A` matrix of the linear system that models the contact dynamics.
            - The `b` vector of the linear system that models the contact dynamics.
            - The `A_sc` matrix of the linear system that approximates the contact model.
            - The `b_sc` vector of the linear system that approximates the contact model.
        """

        indices_of_enabled_collidable_points = (
            indices_of_enabled_collidable_points
            if indices_of_enabled_collidable_points is not None
            else jnp.arange(
                len(model.kin_dyn_parameters.contact_parameters.body)
            ).astype(int)
        )

        p_t0 = jnp.atleast_2d(
            p_t0
            if p_t0 is not None
            else js.contact.collidable_point_positions(model=model, data=data)[
                indices_of_enabled_collidable_points, :
            ]
        )

        v_t0 = jnp.atleast_2d(
            v_t0
            if v_t0 is not None
            else js.contact.collidable_point_velocities(model=model, data=data)[
                indices_of_enabled_collidable_points, :
            ]
        )

        m_t0 = jnp.atleast_2d(
            m_t0
            if m_t0 is not None
            else data.state.extended["tangential_deformation"][
                indices_of_enabled_collidable_points, :
            ]
        )

        # We expect that the 6D forces of the `link_forces` argument are expressed
        # in the frame corresponding to the velocity representation of `data`.
        references = js.references.JaxSimModelReferences.build(
            model=model,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
            data=data,
            velocity_representation=data.velocity_representation,
        )

        # ===========================
        # Linearize the contact model
        # ===========================

        # Linearize the contact model at the initial state of all considered
        # contact points.
        A_sc_points, b_sc_points = jax.vmap(
            lambda p, v, m: ViscoElasticContacts._linearize_contact_model(
                position=p,
                velocity=v,
                tangential_deformation=m,
                parameters=data.contacts_params,
                terrain=model.terrain,
            )
        )(p_t0, v_t0, m_t0)

        # Since x = [p1, p2, ..., v1, v2, ..., m1, m2, ...], we need to split the A_sc of
        # individual points since otherwise we'd get x = [ p1, v1, m1, p2, v2, m2, ...].
        A_sc_p, A_sc_v, A_sc_m = jnp.split(A_sc_points, indices_or_sections=3, axis=-1)

        # We want to have in output first the forces and then the material deformation rates.
        # Therefore, we need to extract the components is A_sc_* separately.
        A_sc = jnp.vstack(
            [
                jnp.hstack(
                    [
                        jax.scipy.linalg.block_diag(*A_sc_p[:, 0:3, :]),
                        jax.scipy.linalg.block_diag(*A_sc_v[:, 0:3, :]),
                        jax.scipy.linalg.block_diag(*A_sc_m[:, 0:3, :]),
                    ],
                ),
                jnp.hstack(
                    [
                        jax.scipy.linalg.block_diag(*A_sc_p[:, 3:6, :]),
                        jax.scipy.linalg.block_diag(*A_sc_v[:, 3:6, :]),
                        jax.scipy.linalg.block_diag(*A_sc_m[:, 3:6, :]),
                    ]
                ),
            ]
        )

        # We need to do the same for the b_sc.
        b_sc = jnp.hstack(
            [b_sc_points[:, 0:3].flatten(), b_sc_points[:, 3:6].flatten()]
        )

        # ===========================================================
        # Compute the A and b matrices of the contact points dynamics
        # ===========================================================

        with data.switch_velocity_representation(jaxsim.VelRepr.Mixed):

            BW_ν = data.generalized_velocity()

            M = js.model.free_floating_mass_matrix(model=model, data=data)

            CW_Jl_WC = js.contact.jacobian(
                model=model,
                data=data,
                output_vel_repr=jaxsim.VelRepr.Mixed,
            )[indices_of_enabled_collidable_points, 0:3, :]

            CW_J̇l_WC = js.contact.jacobian_derivative(
                model=model, data=data, output_vel_repr=jaxsim.VelRepr.Mixed
            )[indices_of_enabled_collidable_points, 0:3, :]

        # Compute the Delassus matrix.
        ψ = jnp.vstack(CW_Jl_WC) @ jnp.linalg.lstsq(M, jnp.vstack(CW_Jl_WC).T)[0]

        I_nc = jnp.eye(v_t0.flatten().size)
        O_nc = jnp.zeros(shape=(p_t0.flatten().size, p_t0.flatten().size))

        # Pack the A matrix.
        A = jnp.vstack(
            [
                jnp.hstack([O_nc, I_nc, O_nc]),
                ψ @ jnp.split(A_sc, 2, axis=0)[0],
                jnp.split(A_sc, 2, axis=0)[1],
            ]
        )

        # Short names for few variables.
        ν = BW_ν
        J = jnp.vstack(CW_Jl_WC)
        J̇ = jnp.vstack(CW_J̇l_WC)

        # Compute the free system acceleration components.
        with (
            data.switch_velocity_representation(jaxsim.VelRepr.Mixed),
            references.switch_velocity_representation(jaxsim.VelRepr.Mixed),
        ):

            BW_v̇_free_WB, s̈_free = js.ode.system_acceleration(
                model=model,
                data=data,
                link_forces=references.link_forces(model=model, data=data),
                joint_force_references=references.joint_force_references(model=model),
            )

        # Pack the free system acceleration in mixed representation.
        ν̇_free = jnp.hstack([BW_v̇_free_WB, s̈_free])

        # Compute the acceleration of collidable points.
        # This is the true derivative of ṗ only in mixed representation.
        p̈ = J @ ν̇_free + J̇ @ ν

        # Pack the b array.
        b = jnp.hstack(
            [
                jnp.zeros_like(p_t0.flatten()),
                p̈ + ψ @ jnp.split(b_sc, indices_or_sections=2)[0],
                jnp.split(b_sc, indices_or_sections=2)[1],
            ]
        )

        return A, b, A_sc, b_sc

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def _linearize_contact_model(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        parameters: ViscoElasticContactsParams,
        terrain: Terrain,
    ) -> tuple[jtp.Matrix, jtp.Vector]:
        """
        Linearize the Hunt/Crossley contact model at the initial state.

        Args:
            position: The position of the contact point.
            velocity: The velocity of the contact point.
            tangential_deformation: The tangential deformation of the contact point.
            parameters: The parameters of the contact model.
            terrain: The considered terrain.

        Returns:
            A tuple containing the `A` matrix and the `b` vector of the linear system
            corresponding to the contact dynamics linearized at the initial state.
        """

        # Initialize the state at which the model is linearized.
        p0 = jnp.array(position, dtype=float).squeeze()
        v0 = jnp.array(velocity, dtype=float).squeeze()
        m0 = jnp.array(tangential_deformation, dtype=float).squeeze()

        # ============
        # Compute A_sc
        # ============

        compute_contact_force_non_linear_model = functools.partial(
            ViscoElasticContacts._compute_contact_force_non_linear_model,
            parameters=parameters,
            terrain=terrain,
        )

        # Compute with AD the functions to get the Jacobians of CW_fl.
        df_dp_fun, df_dv_fun, df_dm_fun = (
            jax.jacrev(
                lambda p0, v0, m0: compute_contact_force_non_linear_model(
                    position=p0, velocity=v0, tangential_deformation=m0
                )[0],
                argnums=num,
            )
            for num in (0, 1, 2)
        )

        # Compute with AD the functions to get the Jacobians of ṁ.
        dṁ_dp_fun, dṁ_dv_fun, dṁ_dm_fun = (
            jax.jacrev(
                lambda p0, v0, m0: compute_contact_force_non_linear_model(
                    position=p0, velocity=v0, tangential_deformation=m0
                )[1],
                argnums=num,
            )
            for num in (0, 1, 2)
        )

        # Compute the Jacobians of the contact forces w.r.t. the state.
        df_dp = jnp.vstack(df_dp_fun(p0, v0, m0))
        df_dv = jnp.vstack(df_dv_fun(p0, v0, m0))
        df_dm = jnp.vstack(df_dm_fun(p0, v0, m0))

        # Compute the Jacobians of the material deformation rate w.r.t. the state.
        dṁ_dp = jnp.vstack(dṁ_dp_fun(p0, v0, m0))
        dṁ_dv = jnp.vstack(dṁ_dv_fun(p0, v0, m0))
        dṁ_dm = jnp.vstack(dṁ_dm_fun(p0, v0, m0))

        # Pack the A matrix.
        A_sc = jnp.vstack(
            [
                jnp.hstack([df_dp, df_dv, df_dm]),
                jnp.hstack([dṁ_dp, dṁ_dv, dṁ_dm]),
            ]
        )

        # ============
        # Compute b_sc
        # ============

        # Compute the output of the non-linear model at the initial state.
        x0 = jnp.hstack([p0, v0, m0])
        f0, ṁ0 = compute_contact_force_non_linear_model(
            position=p0, velocity=v0, tangential_deformation=m0
        )

        # Pack the b vector.
        b_sc = jnp.hstack([f0, ṁ0]) - A_sc @ x0

        return A_sc, b_sc

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("terrain",))
    def _compute_contact_force_non_linear_model(
        position: jtp.VectorLike,
        velocity: jtp.VectorLike,
        tangential_deformation: jtp.VectorLike,
        parameters: ViscoElasticContactsParams,
        terrain: Terrain,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        """
        Compute the contact forces using the non-linear Hunt/Crossley model.

        Args:
            position: The position of the contact point.
            velocity: The velocity of the contact point.
            tangential_deformation: The tangential deformation of the contact point.
            parameters: The parameters of the contact model.
            terrain: The considered terrain.

        Returns:
            A tuple containing:
            - The linear contact force in the mixed contact frame.
            - The rate of material deformation.
        """

        # Compute the linear contact force in mixed representation using
        # the non-linear Hunt/Crossley model.
        # The following function also returns the rate of material deformation.
        CW_fl, ṁ = SoftContacts.hunt_crossley_contact_model(
            position=position,
            velocity=velocity,
            tangential_deformation=tangential_deformation,
            terrain=terrain,
            K=parameters.K,
            D=parameters.D,
            mu=parameters.static_friction,
            p=parameters.p,
            q=parameters.q,
        )

        return CW_fl, ṁ

    @staticmethod
    @jax.jit
    def integrate_data_with_average_contact_forces(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        *,
        dt: jtp.FloatLike,
        link_forces: jtp.MatrixLike | None = None,
        joint_force_references: jtp.VectorLike | None = None,
        average_link_contact_forces_inertial: jtp.MatrixLike | None = None,
        average_of_average_link_contact_forces_mixed: jtp.MatrixLike | None = None,
    ) -> js.data.JaxSimModelData:
        """
        Advance the system state by integrating the dynamics.

        Args:
            model: The model to consider.
            data: The data of the considered model.
            dt: The integration time step.
            link_forces:
                The 6D forces to apply to the links expressed in the frame corresponding
                to the velocity representation of `data`.
            joint_force_references: The joint force references to apply.
            average_link_contact_forces_inertial:
                The average contact forces computed with the exponential integrator and
                expressed in the inertial-fixed frame.
            average_of_average_link_contact_forces_mixed:
                The average of the average contact forces computed with the exponential
                integrator and expressed in the mixed frame.

        Returns:
            The data object storing the system state at the final time.
        """

        s_t0 = data.joint_positions()
        W_p_B_t0 = data.base_position()
        W_Q_B_t0 = data.base_orientation(dcm=False)

        ṡ_t0 = data.joint_velocities()
        with data.switch_velocity_representation(jaxsim.VelRepr.Mixed):
            W_ṗ_B_t0 = data.base_velocity()[0:3]
            W_ω_WB_t0 = data.base_velocity()[3:6]

        with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):
            W_ν_t0 = data.generalized_velocity()

        # We expect that the 6D forces of the `link_forces` argument are expressed
        # in the frame corresponding to the velocity representation of `data`.
        references = js.references.JaxSimModelReferences.build(
            model=model,
            link_forces=link_forces,
            joint_force_references=joint_force_references,
            data=data,
            velocity_representation=data.velocity_representation,
        )

        W_f̅_L = (
            jnp.array(average_link_contact_forces_inertial)
            if average_link_contact_forces_inertial is not None
            else jnp.zeros_like(references._link_forces)
        ).astype(float)

        LW_f̿_L = (
            jnp.array(average_of_average_link_contact_forces_mixed)
            if average_of_average_link_contact_forces_mixed is not None
            else W_f̅_L
        ).astype(float)

        # Compute the system inertial acceleration, used to integrate the system velocity.
        # It considers the average contact forces computed with the exponential integrator.
        with (
            data.switch_velocity_representation(jaxsim.VelRepr.Inertial),
            references.switch_velocity_representation(jaxsim.VelRepr.Inertial),
        ):

            W_ν̇_pr = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    joint_force_references=references.joint_force_references(
                        model=model
                    ),
                    link_forces=W_f̅_L + references.link_forces(model=model, data=data),
                )
            )

        # Compute the system mixed acceleration, used to integrate the system position.
        # It considers the average of the average contact forces computed with the
        # exponential integrator.
        with (
            data.switch_velocity_representation(jaxsim.VelRepr.Mixed),
            references.switch_velocity_representation(jaxsim.VelRepr.Mixed),
        ):

            BW_ν̇_pr2 = jnp.hstack(
                js.ode.system_acceleration(
                    model=model,
                    data=data,
                    joint_force_references=references.joint_force_references(
                        model=model
                    ),
                    link_forces=LW_f̿_L + references.link_forces(model=model, data=data),
                )
            )

        # Integrate the system velocity using the inertial-fixed acceleration.
        W_ν_plus = W_ν_t0 + dt * W_ν̇_pr

        # Integrate the system position using the mixed velocity.
        q_plus = jnp.hstack(
            [
                # Note: here both ṗ and p̈ -> need mixed representation.
                W_p_B_t0 + dt * W_ṗ_B_t0 + 0.5 * dt**2 * BW_ν̇_pr2[0:3],
                jaxsim.math.Quaternion.integration(
                    dt=dt,
                    quaternion=W_Q_B_t0,
                    omega=(W_ω_WB_t0 + 0.5 * dt * BW_ν̇_pr2[3:6]),
                    omega_in_body_fixed=False,
                ).squeeze(),
                s_t0 + dt * ṡ_t0 + 0.5 * dt**2 * BW_ν̇_pr2[6:],
            ]
        )

        # Create the data at the final time.
        data_tf = data.copy()
        data_tf = data_tf.reset_joint_positions(q_plus[7:])
        data_tf = data_tf.reset_base_position(q_plus[0:3])
        data_tf = data_tf.reset_base_quaternion(q_plus[3:7])
        data_tf = data_tf.reset_joint_velocities(W_ν_plus[6:])
        data_tf = data_tf.reset_base_velocity(
            W_ν_plus[0:6], velocity_representation=jaxsim.VelRepr.Inertial
        )

        return data_tf.replace(
            velocity_representation=data.velocity_representation, validate=False
        )


@jax.jit
def step(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    dt: jtp.FloatLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    joint_force_references: jtp.VectorLike | None = None,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    """
    Step the system dynamics with the visco-elastic contact model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        dt: The time step to consider. If not specified, it is read from the model.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        joint_force_references: The joint force references to consider.

    Returns:
        A tuple containing the new data of the model
        and an empty dictionary of auxiliary data.
    """

    assert isinstance(model.contact_model, ViscoElasticContacts)
    assert isinstance(data.contacts_params, ViscoElasticContactsParams)

    # Compute the contact forces in inertial-fixed representation.
    # TODO: understand what's wrong in other representations.
    data_inertial_fixed = data.replace(
        velocity_representation=jaxsim.VelRepr.Inertial, validate=False
    )

    # Create the references object.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        link_forces=link_forces,
        joint_force_references=joint_force_references,
        velocity_representation=data.velocity_representation,
    )

    # Initialize the time step.
    dt = dt if dt is not None else model.time_step

    # Compute the contact forces with the exponential integrator.
    W_f̅_C, aux_data = model.contact_model.compute_contact_forces(
        model=model,
        data=data_inertial_fixed,
        dt=jnp.array(dt).astype(float),
        link_forces=references.link_forces(model=model, data=data),
        joint_force_references=references.joint_force_references(model=model),
    )

    # Extract the final material deformation and the average of average forces
    # from the dictionary containing auxiliary data.
    m_tf = aux_data["m_tf"]
    W_f̿_C = aux_data["W_f_avg2_C"]

    # ===============================
    # Compute the link contact forces
    # ===============================

    # Get the link contact forces by summing the forces of contact points belonging
    # to the same link.
    W_f̅_L, W_f̿_L = jax.vmap(
        lambda W_f_C: model.contact_model.link_forces_from_contact_forces(
            model=model, data=data_inertial_fixed, contact_forces=W_f_C
        )
    )(jnp.stack([W_f̅_C, W_f̿_C]))

    # Compute the link transforms.
    W_H_L = (
        js.model.forward_kinematics(model=model, data=data)
        if data.velocity_representation is not jaxsim.VelRepr.Inertial
        else jnp.zeros(shape=(model.number_of_links(), 4, 4))
    )

    # For integration purpose, we need the average of average forces expressed in
    # mixed representation.
    LW_f̿_L = jax.vmap(
        lambda W_f_L, W_H_L: (
            ModelDataWithVelocityRepresentation.inertial_to_other_representation(
                array=W_f_L,
                other_representation=jaxsim.VelRepr.Mixed,
                transform=W_H_L,
                is_force=True,
            )
        )
    )(W_f̿_L, W_H_L)

    # ==========================
    # Integrate the system state
    # ==========================

    # Integrate the system dynamics using the average contact forces.
    data_tf: js.data.JaxSimModelData = (
        model.contact_model.integrate_data_with_average_contact_forces(
            model=model,
            data=data_inertial_fixed,
            dt=dt,
            link_forces=references.link_forces(model=model, data=data),
            joint_force_references=references.joint_force_references(model=model),
            average_link_contact_forces_inertial=W_f̅_L,
            average_of_average_link_contact_forces_mixed=LW_f̿_L,
        )
    )

    # Store the tangential deformation at the final state.
    # Note that this was integrated in the continuous time domain, therefore it should
    # be much more accurate than the one computed with the discrete soft contacts.
    with data_tf.mutable_context():

        # Extract the indices corresponding to the enabled collidable points.
        # The visco-elastic contact model computed only their contact forces.
        indices_of_enabled_collidable_points = (
            model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
        )

        data_tf.state.extended |= {
            "tangential_deformation": data_tf.state.extended["tangential_deformation"]
            .at[indices_of_enabled_collidable_points]
            .set(m_tf)
        }

    # Restore the original velocity representation.
    data_tf = data_tf.replace(
        velocity_representation=data.velocity_representation, validate=False
    )

    return data_tf, {}
