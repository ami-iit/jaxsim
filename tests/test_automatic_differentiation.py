import os

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

import jaxsim.api as js
import jaxsim.rbda
from jaxsim import VelRepr

# All JaxSim algorithms, excluding the variable-step integrators, should support
# being automatically differentiated until second order, both in FWD and REV modes.
# However, checking the second-order derivatives is particularly slow and makes
# CI tests take too long. Therefore, we only check first-order derivatives.
AD_ORDER = os.environ.get("JAXSIM_TEST_AD_ORDER", 1)


def get_random_data_and_references(
    model: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    key: jax.Array,
) -> tuple[js.data.JaxSimModelData, js.references.JaxSimModelReferences]:

    key, subkey = jax.random.split(key, num=2)

    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    key, subkey1, subkey2 = jax.random.split(key, num=3)

    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base.
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    return data, references


def test_ad_aba(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()
    ṡ = data.joint_velocities(model=model)
    xfb = data.state.physics_model.xfb()

    # Inputs.
    f = references.link_forces(model=model)
    τ = references.joint_force_references(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    aba = lambda xfb, s, ṡ, tau, f_ext: jaxsim.rbda.aba(
        model=model.physics_model, xfb=xfb, q=s, qd=ṡ, tau=tau, f_ext=f_ext
    )

    # Check derivatives against finite differences.
    check_grads(
        f=aba,
        args=(xfb, s, ṡ, τ, f),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_rnea(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()
    ṡ = data.joint_velocities(model=model)
    xfb = data.state.physics_model.xfb()

    # Inputs.
    f = references.link_forces(model=model)

    # ====
    # Test
    # ====

    key, subkey1, subkey2 = jax.random.split(key, num=3)
    W_v̇_WB = jax.random.uniform(subkey1, shape=(6,), minval=-1)
    s̈ = jax.random.uniform(subkey2, shape=(model.dofs(),), minval=-1)

    # Get a closure exposing only the parameters to be differentiated.
    rnea = lambda xfb, s, ṡ, s̈, W_v̇_WB, f_ext: jaxsim.rbda.rnea(
        model=model.physics_model, xfb=xfb, q=s, qd=ṡ, qdd=s̈, a0fb=W_v̇_WB, f_ext=f_ext
    )

    # Check derivatives against finite differences.
    check_grads(
        f=rnea,
        args=(xfb, s, ṡ, s̈, W_v̇_WB, f),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_crba(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    crba = lambda s: jaxsim.rbda.crba(model=model.physics_model, q=s)

    # Check derivatives against finite differences.
    check_grads(
        f=crba,
        args=(s,),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_fk(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()
    xfb = data.state.physics_model.xfb()

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    fk = lambda xfb, s: jaxsim.rbda.forward_kinematics_model(
        model=model.physics_model, xfb=xfb, q=s
    )

    # Check derivatives against finite differences.
    check_grads(
        f=fk,
        args=(xfb, s),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_jacobian(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()

    # ====
    # Test
    # ====

    # Get the link indices.
    link_indices = js.link.names_to_idxs(model=model, link_names=model.link_names())

    # Get a closure exposing only the parameters to be differentiated.
    # We differentiate the jacobian of the last link, likely among those
    # farther from the base.
    jacobian = lambda s: jaxsim.rbda.jacobian(
        model=model.physics_model, q=s, body_index=link_indices[-1]
    )

    # Check derivatives against finite differences.
    check_grads(
        f=jacobian,
        args=(s,),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_soft_contacts(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    key, subkey1, subkey2, subkey3 = jax.random.split(prng_key, num=4)
    p = jax.random.uniform(subkey1, shape=(3,), minval=-1)
    v = jax.random.uniform(subkey2, shape=(3,), minval=-1)
    m = jax.random.uniform(subkey3, shape=(3,), minval=-1)

    # Get the soft contacts parameters.
    parameters = js.contact.estimate_good_soft_contacts_parameters(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    soft_contacts = lambda p, v, m: jaxsim.rbda.SoftContacts(
        parameters=parameters
    ).contact_model(position=p, velocity=v, tangential_deformation=m)

    # Check derivatives against finite differences.
    check_grads(
        f=soft_contacts,
        args=(p, v, m),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_integration(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=key
    )

    # Perturbation used for computing finite differences.
    ε = jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3)

    # State in VelRepr.Inertial representation.
    s = data.joint_positions()
    ṡ = data.joint_velocities(model=model)
    xfb = data.state.physics_model.xfb()
    m = data.state.soft_contacts.tangential_deformation

    # Inputs.
    f = references.link_forces(model=model)
    τ = references.joint_force_references(model=model)

    # ====
    # Test
    # ====

    import jaxsim.integrators

    # Note that only fixes-step integrators support both FWD and RWD gradients.
    # Select a second-order Heun scheme with quaternion integrated on SO(3).
    # Note that it's always preferable using the SO(3) versions on AD applications so
    # that the gradient of the integrated dynamics always considers unary quaternions.
    integrator = jaxsim.integrators.fixed_step.Heun2SO3.build(
        dynamics=js.ode.wrap_system_dynamics_for_integration(
            model=model,
            data=data,
            system_dynamics=js.ode.system_dynamics,
        ),
    )

    # Initialize the integrator.
    t0, dt = 0.0, 0.001
    integrator_state = integrator.init(x0=data.state, t0=t0, dt=dt)

    # Function exposing only the parameters to be differentiated.
    def step(
        xfb: jax.typing.ArrayLike,
        s: jax.typing.ArrayLike,
        ṡ: jax.typing.ArrayLike,
        m: jax.typing.ArrayLike,
        tau: jax.typing.ArrayLike,
        f_ext: jax.typing.ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        data_x0 = data.replace(
            state=js.ode_data.ODEState.build(
                physics_model_state=js.ode_data.PhysicsModelState.build(
                    joint_positions=s,
                    joint_velocities=ṡ,
                    base_position=xfb[4:7],
                    base_quaternion=xfb[0:4],
                    base_linear_velocity=xfb[7:10],
                    base_angular_velocity=xfb[10:13],
                ),
                soft_contacts_state=js.ode_data.SoftContactsState.build(
                    tangential_deformation=m
                ),
            ),
        )

        data_xf, _ = js.model.step(
            dt=dt,
            model=model,
            data=data_x0,
            integrator=integrator,
            integrator_state=integrator_state,
            joint_forces=tau,
            external_forces=f_ext,
        )

        s_xf = data_xf.joint_positions()
        ṡ_xf = data_xf.joint_velocities()
        xfb_xf = data_xf.state.physics_model.xfb()
        m_xf = data_xf.state.soft_contacts.tangential_deformation

        return xfb_xf, s_xf, ṡ_xf, m_xf

    # Check derivatives against finite differences.
    check_grads(
        f=step,
        args=(xfb, s, ṡ, m, τ, f),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
        # This check (at least on ErgoCub) needs larger tolerances
        rtol=0.0001,
    )
