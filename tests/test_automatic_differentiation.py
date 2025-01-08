import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import VelRepr
from jaxsim.rbda.contacts import SoftContacts, SoftContactsParams

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
    W_p_B = data.base_position
    W_Q_B = data.base_orientation
    s = data.joint_positions
    W_v_WB = data.kyn_dyn.base_velocity
    ṡ = data.joint_velocities
    i_X_λ = data.kyn_dyn.joint_transforms
    S = data.kyn_dyn.motion_subspaces

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
        joint_transforms=i_X_λ,
        motion_subspaces=S,
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
    W_p_B = data.base_position
    W_Q_B = data.base_orientation
    s = data.joint_positions
    W_v_WB = data.kyn_dyn.base_velocity
    ṡ = data.joint_velocities
    i_X_λ = data.kyn_dyn.joint_transforms
    S = data.kyn_dyn.motion_subspaces

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
        joint_transforms=i_X_λ,
        motion_subspaces=S,
    )

    # Check derivatives against finite differences.
    check_grads(
        f=rnea,
        args=(
            W_p_B,
            W_Q_B,
            s,
            W_v_WB,
            ṡ,
            W_v̇_WB,
            s̈,
            W_f_L,
            g,
        ),
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
    s = data.joint_positions
    i_X_λ = data.kyn_dyn.joint_transforms
    S = data.kyn_dyn.motion_subspaces

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    crba = lambda s: jaxsim.rbda.crba(
        model=model,
        joint_positions=s,
        joint_transforms=i_X_λ,
        motion_subspaces=S,
    )

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
    W_p_B = data.base_position
    W_Q_B = data.base_orientation
    s = data.joint_positions
    i_X_λ = data.kyn_dyn.joint_transforms

    # ====
    # Test
    # ====

    # Get a closure exposing only the parameters to be differentiated.
    fk = lambda W_p_B, W_Q_B: jaxsim.rbda.forward_kinematics_model(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B / jnp.linalg.norm(W_Q_B),
        joint_positions=s,
        joint_transforms=i_X_λ,
    )

    # Check derivatives against finite differences.
    check_grads(
        f=fk,
        args=(W_p_B, W_Q_B),
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
    s = data.joint_positions
    i_X_λ = data.kyn_dyn.joint_transforms
    S = data.kyn_dyn.motion_subspaces

    # ====
    # Test
    # ====

    # Get the link indices.
    link_indices = jnp.arange(model.number_of_links())

    # Get a closure exposing only the parameters to be differentiated.
    # We differentiate the jacobian of the last link, likely among those
    # farther from the base.
    jacobian = lambda s: jaxsim.rbda.jacobian(
        model=model,
        joint_positions=s,
        link_index=link_indices[-1],
        joint_transforms=i_X_λ,
        motion_subspaces=S,
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
    parameters = js.contact.estimate_good_contact_parameters(model=model)

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

        W_f_Ci, CW_ṁ = SoftContacts.compute_contact_force(
            position=p,
            velocity=v,
            tangential_deformation=m,
            parameters=params,
            terrain=model.terrain,
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
    W_p_B = data.base_position
    W_Q_B = data.base_orientation
    s = data.joint_positions
    W_v_WB = data.kyn_dyn.base_velocity
    ṡ = data.joint_velocities
    m = data.state.extended["tangential_deformation"]

    # Inputs.
    W_f_L = references.link_forces(model=model)
    τ = references.joint_force_references(model=model)

    # ====
    # Test
    # ====

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
                extended_state={"tangential_deformation": m},
            ),
        )

        # Update the kyn_dyn cache.
        data_x0.update_kyn_dyn(model=model)

        data_xf, _ = js.model.step(
            model=model,
            data=data_x0,
            joint_force_references=τ,
            link_forces=W_f_L,
        )

        xf_W_p_B = data_xf.base_position
        xf_W_Q_B = data_xf.state.physics_model.base_quaternion
        xf_s = data_xf.joint_positions
        xf_W_v_WB = data_xf.kyn_dyn.base_velocity
        xf_ṡ = data_xf.joint_velocities
        xf_m = data_xf.state.extended["tangential_deformation"]

        return xf_W_p_B, xf_W_Q_B, xf_s, xf_W_v_WB, xf_ṡ, xf_m

    # Check derivatives against finite differences.
    check_grads(
        f=step,
        args=(W_p_B, W_Q_B, s, W_v_WB, ṡ, m, τ, W_f_L),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )


def test_ad_safe_norm(
    prng_key: jax.Array,
):

    _, subkey = jax.random.split(prng_key, num=2)
    array = jax.random.uniform(subkey, shape=(4,), minval=-5, maxval=5)

    # ====
    # Test
    # ====

    # Test that the safe_norm function is compatible with batching.
    array = jnp.stack([array, array])
    assert jaxsim.math.safe_norm(array, axis=1).shape == (2,)

    # Test that the safe_norm function is correctly computing the norm.
    assert np.allclose(jaxsim.math.safe_norm(array), np.linalg.norm(array))

    # Function exposing only the parameters to be differentiated.
    def safe_norm(array: jtp.Array) -> jtp.Array:

        return jaxsim.math.safe_norm(array)

    # Check derivatives against finite differences.
    check_grads(
        f=safe_norm,
        args=(array,),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )

    # Check derivatives against finite differences when the array is zero.
    check_grads(
        f=safe_norm,
        args=(jnp.zeros_like(array),),
        order=AD_ORDER,
        modes=["rev", "fwd"],
        eps=ε,
    )
