import os
import time

os.environ["JAXSIM_DISABLE_EXCEPTIONS"] = "0"
os.environ["JAX_ENABLE_X64"] = "0"

import jaxsim.api as js
import pathlib
from jaxsim import VelRepr
import jax.numpy as jnp
import jax
from jaxsim.rbda.contacts.relaxed_rigid import (
    RelaxedRigidContacts,
    RelaxedRigidContactsParams,
)
import pandas as pd
import mujoco
from mujoco import mjx


try:
    os.environ["ROBOT_DESCRIPTION_COMMIT"] = "v0.7.1"

    import robot_descriptions.ergocub_description

finally:
    _ = os.environ.pop("ROBOT_DESCRIPTION_COMMIT", None)

model_urdf_path = pathlib.Path(
    robot_descriptions.ergocub_description.URDF_PATH.replace(
        "ergoCubSN000", "ergoCubSN001"
    )
)

model_full = js.model.JaxSimModel.build_from_model_description(
    model_description=model_urdf_path,
    contact_model=RelaxedRigidContacts.build(solver_options={"maxiter": 6}),
    time_step=0.002,
    contact_params=RelaxedRigidContactsParams.build(
        d_min=0.9,
        d_max=0.95,
        width=1e-3,
        midpoint=0.5,
        power=2.0,
        time_constant=0.02,
        damping_coefficient=1.0,
        mu=5e-3,
    ),
)

reduced_joints = tuple(
    j
    for j in model_full.joint_names()
    if "camera" not in j
    # Remove head and hands.
    and "neck" not in j
    and "wrist" not in j
    and "thumb" not in j
    and "index" not in j
    and "middle" not in j
    and "ring" not in j
    and "pinkie" not in j
)

js_model = js.model.reduce(model=model_full, considered_joints=reduced_joints)
js_model = jax.device_put(js_model)
js_data = js.data.JaxSimModelData.build(model=js_model)
js_data = jax.device_put(js_data)


mj_model = mujoco.MjModel.from_xml_path("muj_model.xml")
mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
mj_model.opt.iterations = 6
mj_model.opt.ls_iterations = 6
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetData(mj_model, mj_data)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)


def benchmark_jaxsim(num_envs):
    data = jax.vmap(
        lambda W_p_B: js.data.JaxSimModelData.build(
            model=js_model,
            velocity_representation=VelRepr.Mixed,
            base_position=W_p_B,
        )
    )(jnp.repeat(js_data.base_position, num_envs, axis=0).reshape((num_envs, 3)))


    length = 1000
    def scan(d):
        @jax.vmap
        def scan_body(d, _):
            d = js.model.step(js_model, d)
            return d, None
        d, _ = jax.lax.scan(scan_body, d, None, length=length)
        return d

    t1 = time.perf_counter()
    scan = jax.jit(scan).lower(data).compile()
    compile_time = time.perf_counter() - t1
    
    start = time.perf_counter()
    out = scan(data)
    jax.block_until_ready(out)
    end = time.perf_counter()

    return (end - start) / length, compile_time


def benchmark_mjx(num_envs):
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, num_envs)
    data = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (27,))))(rng)

    length = 1000
    def scan(d):
        @jax.vmap
        def scan_body(d, _):
            d = mjx.step(mjx_model, d)
            return d, None
        d, _ = jax.lax.scan(scan_body, d, None, length=length)
        return d

    t1 = time.perf_counter()
    scan = jax.jit(scan).lower(data).compile()
    compile_time = time.perf_counter() - t1
    
    start = time.perf_counter()
    out = scan(data)
    jax.block_until_ready(out)
    end = time.perf_counter()

    return (end - start) / length, compile_time

if __name__ == "__main__":
    js_times, mjx_times = [],[]
    batch_sizes =[8, 32, 128, 512, 2048, 8192]  
    for num_envs in batch_sizes:
        js_run_time, js_compile_time = benchmark_jaxsim(num_envs)
        mjx_run_time, mjx_compile_time = benchmark_mjx(num_envs)
        js_sps = (1 / js_run_time) * num_envs
        js_times.append(js_sps)
        mjx_sps = (1 / mjx_run_time) * num_envs
        mjx_times.append(mjx_sps)

        print('-'*20, 'num_envs:', num_envs, '-'*20)
        print(f"JaxSim, time: {int(js_sps)} SPS, compile_time: {js_compile_time}")
        print(f"MJX, time: {int(mjx_sps)} SPS, compile_time: {mjx_compile_time}")

    df = pd.DataFrame([js_times, mjx_times], index=["JaxSim", "MJX"], columns=batch_sizes)
    print(df)
    df.to_csv("bench.csv")


