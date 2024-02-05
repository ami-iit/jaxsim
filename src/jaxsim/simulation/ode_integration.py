import enum
import functools
from typing import Any, Dict, Tuple, Union

import jax.flatten_util
from jax.experimental.ode import odeint

import jaxsim.typing as jtp
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.physics.algos.terrain import FlatTerrain, Terrain
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.simulation import integrators, ode


class IntegratorType(enum.IntEnum):
    RungeKutta4 = enum.auto()
    EulerForward = enum.auto()
    EulerSemiImplicit = enum.auto()
    EulerSemiImplicitManifold = enum.auto()


_integrator_registry = {
    IntegratorType.RungeKutta4: integrators.odeint_rk4,
    IntegratorType.EulerForward: integrators.odeint_euler,
    IntegratorType.EulerSemiImplicit: integrators.odeint_euler_semi_implicit,
    IntegratorType.EulerSemiImplicitManifold: integrators.odeint_euler_semi_implicit_manifold_one_step,
}


@jax.jit
def ode_integration_rk4_adaptive(
    x0: jtp.Array,
    t: integrators.TimeHorizon,
    physics_model: PhysicsModel,
    *args,
    **kwargs,
) -> jtp.Array:
    # Close function over its inputs and parameters
    dx_dt_closure = lambda x, ts: ode.dx_dt(x, ts, physics_model, *args)

    return odeint(dx_dt_closure, x0, t, **kwargs)


@functools.partial(
    jax.jit, static_argnames=["num_sub_steps", "integrator_type", "return_aux"]
)
def ode_integration_fixed_step(
    x0: ode.ode_data.ODEState,
    t: integrators.TimeHorizon,
    physics_model: PhysicsModel,
    integrator_type: IntegratorType,
    soft_contacts_params: SoftContactsParams = SoftContactsParams(),
    terrain: Terrain = FlatTerrain(),
    ode_input: ode.ode_data.ODEInput | None = None,
    *args,
    num_sub_steps: int = 1,
    return_aux: bool = False,
) -> Union[ode.ode_data.ODEState, Tuple[ode.ode_data.ODEState, Dict]]:
    # Close func over additional inputs and parameters
    dx_dt_closure = lambda x, ts: ode.dx_dt(
        x, ts, physics_model, soft_contacts_params, ode_input, terrain, *args
    )

    # Integrate over the horizon
    out = _integrator_registry[integrator_type](
        func=dx_dt_closure,
        y0=x0,
        t=t,
        num_sub_steps=num_sub_steps,
        return_aux=return_aux,
    )

    # Return output pytree and, optionally, the aux dict
    state = out if not return_aux else out[0]
    return (state, out[1]) if return_aux else state
