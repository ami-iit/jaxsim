# JaxSim

**JaxSim** is a **differentiable physics engine** and **multibody dynamics library** built with JAX, tailored for control and robotic learning applications.

<div align="center">
<br/>
<table>
  <tr>
    <th><img src="https://github.com/user-attachments/assets/f9661fae-9a85-41dd-9a58-218758ec8c9c"></th>
    <th><img src="https://github.com/user-attachments/assets/62b88b9d-45ea-4d22-99d2-f24fc842dd29"></th>
  </tr>
</table>
<br/>
</div>

## Features
- Reduced-coordinate physics engine for **fixed-base** and **floating-base** robots.
- Multibody dynamics library for model-based control algorithms.
- Fully Python-based, leveraging [jax][jax] following a functional programming paradigm.
- Seamless execution on CPUs, GPUs, and TPUs.
- Supports JIT compilation and automatic vectorization for high performance.
- Compatible with SDF models and URDF (via [sdformat][sdformat] conversion).

## Usage

### Using JaxSim as simulator


```python
import pathlib

import icub_models
import jax.numpy as jnp

import jaxsim.api as js

# Load the iCub model
model_path = icub_models.get_model_file("iCubGazeboV2_5")

joints = ('torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
          'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
          'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch',
          'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
          'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch',
          'r_ankle_roll')

# Build and reduce the model
model_description = pathlib.Path(model_path)

full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_description, time_step=0.0001, is_urdf=True
)

model = js.model.reduce(model=full_model, considered_joints=joints)

# Get the number of degrees of freedom
ndof = model.dofs()

# Initialize data and simulation
# Note that the default data representation is mixed velocity representation
data = js.data.JaxSimModelData.build(
    model=model, base_position=jnp.array([0.0, 0.0, 1.0])
)

T = jnp.arange(start=0, stop=1.0, step=model.time_step)

tau = jnp.zeros(ndof)

# Simulate
for _ in T:
    data = js.model.step(
        model=model, data=data, link_forces=None, joint_force_references=tau
    )
```

### Using JaxSim as a multibody dynamics library
``` python
import pathlib

import icub_models
import jax.numpy as jnp

import jaxsim.api as js

# Load the iCub model
model_path = icub_models.get_model_file("iCubGazeboV2_5")

joints = ('torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
          'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
          'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch',
          'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
          'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch',
          'r_ankle_roll')

# Build and reduce the model
model_description = pathlib.Path(model_path)

full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_description, time_step=0.0001, is_urdf=True
)

model = js.model.reduce(model=full_model, considered_joints=joints)

# Initialize model data
data = js.data.JaxSimModelData.build(
    model=model,
    base_position=jnp.array([0.0, 0.0, 1.0]),
)

# Frame and dynamics computations
frame_index = js.frame.name_to_idx(model=model, frame_name="l_foot")

# Frame transformation
W_H_F = js.frame.transform(
    model=model, data=data, frame_index=frame_index
)

# Frame Jacobian
W_J_F = js.frame.jacobian(
    model=model, data=data, frame_index=frame_index
)

# Dynamics properties
M = js.model.free_floating_mass_matrix(model=model, data=data)  # Mass matrix
h = js.model.free_floating_bias_forces(model=model, data=data)  # Bias forces
g = js.model.free_floating_gravity_forces(model=model, data=data)  # Gravity forces
C = js.model.free_floating_coriolis_matrix(model=model, data=data)  # Coriolis matrix

# Print dynamics results
print(f"{M.shape=} \n{h.shape=} \n{g.shape=} \n{C.shape=}")
```

### Additional features

- Full support for automatic differentiation of RBDAs (forward and reverse modes) with JAX.
- Support for automatically differentiating against kinematics and dynamics parameters.
- All fixed-step integrators are forward and reverse differentiable.
- Check the example folder for additional use cases!

[jax]: https://github.com/google/jax/
[sdformat]: https://github.com/gazebosim/sdformat
[notation]: https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2
[passive_viewer_mujoco]: https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer

> [!WARNING]
> This project is still experimental, APIs could change between releases without notice.

> [!NOTE]
> JaxSim currently focuses on locomotion applications.
> Only contacts between bodies and smooth ground surfaces are supported.

## Installation

<details>
<summary>With <code>conda</code></summary>

You can install the project using [`conda`][conda] as follows:

```bash
conda install jaxsim -c conda-forge
```

You can enforce GPU support, if needed, by also specifying `"jaxlib = * = *cuda*"`.

</details>

<details>
<summary>With <code>pixi</code></summary>

> ### Note
> The minimum version of `pixi` required is `0.39.0`.

You can add the jaxsim dependency in [`pixi`][pixi] project as follows:

```bash
pixi add jaxsim
```

If you are on Linux and you want to use a `cuda`-powered version of `jax`, remember to add the appropriate line in the [`system-requirements`](https://pixi.sh/latest/reference/pixi_manifest/#the-system-requirements-table) table, i.e. adding

~~~toml
[system-requirements]
cuda = "12"
~~~

if you are using a `pixi.toml` file or

~~~toml
[tool.pixi.system-requirements]
cuda = "12"
~~~

if you are using a `pyproject.toml` file.

</details>

<details>
<summary>With <code>pip</code></summary>

You can install the project using [`pypa/pip`][pip], preferably in a [virtual environment][venv], as follows:

```bash
pip install jaxsim
```

Check [`pyproject.toml`](pyproject.toml) for the complete list of optional dependencies.
You can obtain a full installation using `jaxsim[all]`.

If you need GPU support, follow the official [installation instructions][jax_gpu] of JAX.

</details>

<details>
<summary>Contributors installation (with <code>conda</code>)</summary>

If you want to contribute to the project, we recommend creating the following `jaxsim` conda environment first:

```bash
conda env create -f environment.yml
```

Then, activate the environment and install the project in editable mode:

```bash
conda activate jaxsim
pip install --no-deps -e .
```

</details>

<details>
<summary>Contributors installation (with <code>pixi</code>)</summary>

> ### Note
> The minimum version of `pixi` required is `0.39.0`.

You can install the default dependencies of the project using [`pixi`][pixi] as follows:

```bash
pixi install
```

See `pixi task list` for a list of available tasks.

</details>

[conda]: https://anaconda.org/
[pip]: https://github.com/pypa/pip/
[pixi]: https://pixi.sh/
[venv]: https://docs.python.org/3/tutorial/venv.html
[jax_gpu]: https://github.com/google/jax/#installation

## Documentation

The JaxSim API documentation is available at [jaxsim.readthedocs.io][readthedocs].

[readthedocs]: https://jaxsim.readthedocs.io/


## Overview

<details>
<summary>Structure of the Python package</summary>

```
# tree -L 2 -I "__pycache__" -I "__init__*" -I "__main__*" src/jaxsim

src/jaxsim
|-- api..........................# Package containing the main functional APIs.
|   |-- actuation_model.py.......# |-- APIs for computing quantities related to the actuation model.
|   |-- common.py................# |-- Common utilities used in the current package.
|   |-- com.py...................# |-- APIs for computing quantities related to the center of mass.
|   |-- contact_model.py.........# |-- APIs for computing quantities related to the contact model.
|   |-- contact.py...............# |-- APIs for computing quantities related to the collidable points.
|   |-- data.py..................# |-- Class storing the data of a simulated model.
|   |-- frame.py.................# |-- APIs for computing quantities related to additional frames.
|   |-- integrators.py...........# |-- APIs for integrating the system dynamics.
|   |-- joint.py.................# |-- APIs for computing quantities related to the joints.
|   |-- kin_dyn_parameters.py....# |-- Class storing kinematic and dynamic parameters of a model.
|   |-- link.py..................# |-- APIs for computing quantities related to the links.
|   |-- model.py.................# |-- Class defining a simulated model and APIs for computing related quantities.
|   |-- ode.py...................# |-- APIs for computing quantities related to the system dynamics.
|   `-- references.py............# `-- Helper class to create references (link forces and joint torques).
|-- exceptions.py................# Module containing functions to raise exceptions from JIT-compiled functions.
|-- logging.py...................# Module containing logging utilities.
|-- math.........................# Package containing mathematical utilities.
|   |-- adjoint.py...............# |-- APIs for creating and manipulating 6D transformations.
|   |-- cross.py.................# |-- APIs for computing cross products of 6D quantities.
|   |-- inertia.py...............# |-- APIs for creating and manipulating 6D inertia matrices.
|   |-- joint_model.py...........# |-- APIs defining the supported joint model and the corresponding transformations.
|   |-- quaternion.py............# |-- APIs for creating and manipulating quaternions.
|   |-- rotation.py..............# |-- APIs for creating and manipulating rotation matrices.
|   |-- skew.py..................# |-- APIs for creating and manipulating skew-symmetric matrices.
|   |-- transform.py.............# |-- APIs for creating and manipulating homogeneous transformations.
|   |-- utils.py.................# |-- Common utilities used in the current package.
|-- mujoco.......................# Package containing utilities to interact with the Mujoco passive viewer.
|   |-- loaders.py...............# |-- Utilities for converting JaxSim models to Mujoco models.
|   |-- model.py.................# |-- Class providing high-level methods to compute quantities using Mujoco.
|   `-- visualizer.py............# `-- Class that simplifies opening the passive viewer and recording videos.
|-- parsers......................# Package containing utilities to parse model descriptions (SDF and URDF models).
|   |-- descriptions/............# |-- Package containing the intermediate representation of a model description.
|   |-- kinematic_graph.py.......# |-- Definition of the kinematic graph associated with a parsed model description.
|   `-- rod/.....................# `-- Package to create the intermediate representation from model descriptions using ROD.
|-- rbda.........................# Package containing the low-level rigid body dynamics algorithms.
|   |-- aba.py...................# |-- The Articulated Body Algorithm.
|   |-- collidable_points.py.....# |-- Kinematics of collidable points.
|   |-- contacts/................# |-- Package containing the supported contact models.
|   |-- crba.py..................# |-- The Composite Rigid Body Algorithm.
|   |-- forward_kinematics.py....# |-- Forward kinematics of the model.
|   |-- jacobian.py..............# |-- Full Jacobian and full Jacobian derivative.
|   |-- rnea.py..................# |-- The Recursive Newton-Euler Algorithm.
|   `-- utils.py.................# `-- Common utilities used in the current package.
|-- terrain......................# Package containing resources to specify the terrain.
|   `-- terrain.py...............# `-- Classes defining the supported terrains.
|-- typing.py....................# Module containing type hints.
`-- utils........................# Package of common utilities.
    |-- jaxsim_dataclass.py......# |-- Utilities to operate on pytree dataclasses.
    |-- tracing.py...............# |-- Utilities to use when JAX is tracing functions.
    `-- wrappers.py..............# `-- Utilities to wrap objects for specific use cases on pytree dataclass attributes.
```

</details>

## Credits

The RBDAs are based on the theory of the [Rigid Body Dynamics Algorithms][RBDA]
book by Roy Featherstone.
The algorithms and some simulation features were inspired by its accompanying [code][spatial_v2].

[RBDA]: https://link.springer.com/book/10.1007/978-1-4899-7560-7
[spatial_v2]: http://royfeatherstone.org/spatial/index.html#spatial-software

The development of JaxSim started in late 2021, inspired by early versions of [`google/brax`][brax].
At that time, Brax was implemented in maximal coordinates, and we wanted a physics engine in reduced coordinates.
We are grateful to the Brax team for their work and for showing the potential of [JAX][jax] in this field.

Brax v2 was later implemented with reduced coordinates, following an approach comparable to JaxSim.
The development then shifted to [MJX][mjx], which provides a JAX-based implementation of the Mujoco APIs.

The main differences between MJX/Brax and JaxSim are as follows:

- JaxSim supports out-of-the-box all SDF models with [Pose Frame Semantics][PFS].
- JaxSim only supports collisions between points rigidly attached to bodies and a compliant ground surface.

[brax]: https://github.com/google/brax
[mjx]: https://mujoco.readthedocs.io/en/3.0.0/mjx.html
[PFS]: http://sdformat.org/tutorials?tut=pose_frame_semantics

## Contributing

We welcome contributions from the community.
Please read the [contributing guide](./CONTRIBUTING.md) to get started.

## Citing

```bibtex
@software{ferigo_jaxsim_2022,
  author = {Diego Ferigo and Filippo Luca Ferretti and Silvio Traversaro and Daniele Pucci},
  title = {{JaxSim}: A Differentiable Physics Engine and Multibody Dynamics Library for Control and Robot Learning},
  url = {http://github.com/ami-iit/jaxsim},
  year = {2022},
}
```

Theoretical aspects of JaxSim are based on Chapters 7 and 8 of the following Ph.D. thesis:

```bibtex
@phdthesis{ferigo_phd_thesis_2022,
  title = {Simulation Architectures for Reinforcement Learning applied to Robotics},
  author = {Diego Ferigo},
  school = {University of Manchester},
  type = {PhD Thesis},
  month = {July},
  year = {2022},
}
```

## People

| Authors | Maintainers |
|:------:|:-----------:|
| [<img src="https://avatars.githubusercontent.com/u/469199?v=4" width="40">][df] [<img src="https://avatars.githubusercontent.com/u/102977828?v=4" width="40">][ff] | [<img src="https://avatars.githubusercontent.com/u/102977828?v=4" width="40">][ff] [<img src="https://avatars.githubusercontent.com/u/57228872?v=4" width="40">][ac] |

[df]: https://github.com/diegoferigo
[ff]: https://github.com/flferretti
[ac]: https://github.com/xela-95

## License

[BSD3](https://choosealicense.com/licenses/bsd-3-clause/)
