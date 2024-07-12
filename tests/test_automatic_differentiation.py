import os

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import VelRepr
from jaxsim.rbda.contacts.soft import SoftContacts, SoftContactsParams

# All JaxSim algorithms, excluding the variable-step integrators, should support
# being automatically differentiated until second order, both in FWD and REV modes.
# However, checking the second-order derivatives is particularly slow and makes
# CI tests take too long. Therefore, we only check first-order derivatives.
AD_ORDER = os.environ.get("JAXSIM_TEST_AD_ORDER", 1)

# Define the step size used to compute finite differences depending on the
# floating point resolution.
ε = os.environ.get(
    "JAXSIM_TEST_FD_STEP_SIZE",
    jnp.finfo(jnp.array(0.0)).resolution ** (1 / 3),
)


def get_random_data_and_references(
    model: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    key: jax.Array,
) -> tuple[js.data.JaxSimModelData, js.references.JaxSimModelReferences]:

    key, subkey = jax.random.split(key, num=2)

    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    _, subkey1, subkey2 = jax.random.split(key, num=3)

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

    _, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # Get the standard gravity constant.
    g = jaxsim.math.StandardGravity

    # State in VelRepr.Inertial representation.
    W_p_B = data.base_position()
    W_Q_B = data.base_orientation(dcm=False)
    s = data.joint_positions(model=model)
    W_v_WB = data.base_velocity()
    ṡ = data.joint_velocities(model=model)

    # Inputs.
    W_f_L = references.link_forces(model=model)
    τ = references.joint_force_references(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    aba = lambda W_p_B, W_Q_B, s, W_v_WB, ṡ, τ, W_f_L, g: jaxsim.rbda.aba(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B / jnp.linalg.norm(W_Q_B),
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        joint_forces=τ,
        link_forces=W_f_L,
        standard_gravity=g,
    )

    # Check derivatives against finite differences.
    check_grads(
        f=aba,
        args=(W_p_B, W_Q_B, s, W_v_WB, ṡ, τ, W_f_L, g),
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
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # Get the standard gravity constant.
    g = jaxsim.math.StandardGravity

    # State in VelRepr.Inertial representation.
    W_p_B = data.base_position()
    W_Q_B = data.base_orientation(dcm=False)
    s = data.joint_positions(model=model)
    W_v_WB = data.base_velocity()
    ṡ = data.joint_velocities(model=model)

    # Inputs.
    W_f_L = references.link_forces(model=model)

    # ====
    # Test
    # ====

    _, subkey1, subkey2 = jax.random.split(key, num=3)
    W_v̇_WB = jax.random.uniform(subkey1, shape=(6,), minval=-1)
    s̈ = jax.random.uniform(subkey2, shape=(model.dofs(),), minval=-1)

    # Get a closure exposing only the parameters to be differentiated.
    rnea = lambda W_p_B, W_Q_B, s, W_v_WB, ṡ, W_v̇_WB, s̈, W_f_L, g: jaxsim.rbda.rnea(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B / jnp.linalg.norm(W_Q_B),
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        base_linear_acceleration=W_v̇_WB[0:3],
        base_angular_acceleration=W_v̇_WB[3:6],
        joint_accelerations=s̈,
        link_forces=W_f_L,
        standard_gravity=g,
    )

    # Check derivatives against finite differences.
    check_grads(
        f=rnea,
        args=(W_p_B, W_Q_B, s, W_v_WB, ṡ, W_v̇_WB, s̈, W_f_L, g),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_crba(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data, _ = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # State in VelRepr.Inertial representation.
    s = data.joint_positions(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    crba = lambda s: jaxsim.rbda.crba(model=model, joint_positions=s)

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

    _, subkey = jax.random.split(prng_key, num=2)
    data, _ = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # State in VelRepr.Inertial representation.
    W_p_B = data.base_position()
    W_Q_B = data.base_orientation(dcm=False)
    s = data.joint_positions(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    fk = lambda W_p_B, W_Q_B, s: jaxsim.rbda.forward_kinematics_model(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B / jnp.linalg.norm(W_Q_B),
        joint_positions=s,
    )

    # Check derivatives against finite differences.
    check_grads(
        f=fk,
        args=(W_p_B, W_Q_B, s),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_jacobian(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data, _ = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # State in VelRepr.Inertial representation.
    s = data.joint_positions(model=model)

    # ====
    # Test
    # ====

    # Get the link indices.
    link_indices = js.link.names_to_idxs(model=model, link_names=model.link_names())

    # Get a closure exposing only the parameters to be differentiated.
    # We differentiate the jacobian of the last link, likely among those
    # farther from the base.
    jacobian = lambda s: jaxsim.rbda.jacobian(
        model=model, joint_positions=s, link_index=link_indices[-1]
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

    _, subkey1, subkey2, subkey3 = jax.random.split(prng_key, num=4)
    p = jax.random.uniform(subkey1, shape=(3,), minval=-1)
    v = jax.random.uniform(subkey2, shape=(3,), minval=-1)
    m = jax.random.uniform(subkey3, shape=(3,), minval=-1)

    # Get the soft contacts parameters.
    parameters = js.contact.estimate_good_soft_contacts_parameters(model=model)

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    def close_over_inputs_and_parameters(
        p: jtp.VectorLike,
        v: jtp.VectorLike,
        m: jtp.VectorLike,
        params: SoftContactsParams,
    ) -> tuple[jtp.Vector, jtp.Vector]:
        W_f_Ci, (CW_ṁ,) = SoftContacts(parameters=params).compute_contact_forces(
            position=p, velocity=v, tangential_deformation=m
        )
        return W_f_Ci, CW_ṁ

    # Check derivatives against finite differences.
    check_grads(
        f=close_over_inputs_and_parameters,
        args=(p, v, m, parameters),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
        # On GPU, the tolerance needs to be increased.
        rtol=0.02 if "gpu" in {d.platform for d in p.devices()} else None,
    )


def test_ad_integration(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data, references = get_random_data_and_references(
        model=model, velocity_representation=VelRepr.Inertial, key=subkey
    )

    # State in VelRepr.Inertial representation.
    W_p_B = data.base_position()
    W_Q_B = data.base_orientation(dcm=False)
    s = data.joint_positions(model=model)
    W_v_WB = data.base_velocity()
    ṡ = data.joint_velocities(model=model)
    m = data.state.contact.tangential_deformation

    # Inputs.
    W_f_L = references.link_forces(model=model)
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
        W_p_B: jtp.Vector,
        W_Q_B: jtp.Vector,
        s: jtp.Vector,
        W_v_WB: jtp.Vector,
        ṡ: jtp.Vector,
        m: jtp.Vector,
        τ: jtp.Vector,
        W_f_L: jtp.Matrix,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

        # When JAX tests against finite differences, the injected ε will make the
        # quaternion non-unitary, which will cause the AD check to fail.
        W_Q_B = W_Q_B / jnp.linalg.norm(W_Q_B)

        data_x0 = data.replace(
            state=js.ode_data.ODEState.build(
                physics_model_state=js.ode_data.PhysicsModelState.build(
                    base_position=W_p_B,
                    base_quaternion=W_Q_B,
                    joint_positions=s,
                    base_linear_velocity=W_v_WB[0:3],
                    base_angular_velocity=W_v_WB[3:6],
                    joint_velocities=ṡ,
                ),
                contact=js.ode_data.SoftContactsState.build(tangential_deformation=m),
            ),
        )

        data_xf, _ = js.model.step(
            dt=dt,
            model=model,
            data=data_x0,
            integrator=integrator,
            integrator_state=integrator_state,
            joint_forces=τ,
            link_forces=W_f_L,
        )

        xf_W_p_B = data_xf.base_position()
        xf_W_Q_B = data_xf.base_orientation(dcm=False)
        xf_s = data_xf.joint_positions(model=model)
        xf_W_v_WB = data_xf.base_velocity()
        xf_ṡ = data_xf.joint_velocities(model=model)
        xf_m = data_xf.state.contact.tangential_deformation

        return xf_W_p_B, xf_W_Q_B, xf_s, xf_W_v_WB, xf_ṡ, xf_m

    # Check derivatives against finite differences.
    check_grads(
        f=step,
        args=(W_p_B, W_Q_B, s, W_v_WB, ṡ, m, τ, W_f_L),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )
