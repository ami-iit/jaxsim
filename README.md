# JaxSim

JaxSim is a **differentiable physics engine** and **multibody dynamics library** designed for applications in control and robot learning, implemented with JAX.

Its design facilitates research and accelerates prototyping in the intersection of robotics and artificial intelligence.

## Features

- Physics engine in reduced coordinates supporting fixed-base and floating-base robots.
- Multibody dynamics library providing all the necessary components for developing model-based control algorithms.
- Completely developed in Python with [`google/jax`][jax] following a functional programming paradigm.
- Transparent support for running on CPUs, GPUs, and TPUs.
- Full support for JIT compilation for increased performance.
- Full support for automatic vectorization for massive parallelization of open-loop and closed-loop architectures.
- Support for SDF models and, upon conversion with [sdformat][sdformat], URDF models.
- Visualization based on the [passive viewer][passive_viewer_mujoco] of Mujoco.

### JaxSim as a simulator

- Wide range of fixed-step explicit Runge-Kutta integrators.
- Support for variable-step integrators implemented as embedded Runge-Kutta schemes.
- Improved stability by optionally integrating the base orientation on the $\text{SO}(3)$ manifold.
- Soft contacts model supporting full friction cone and sticking-slipping transition.
- Collision detection between points rigidly attached to bodies and uneven ground surfaces.

### JaxSim as a multibody dynamics library

- Provides rigid body dynamics algorithms (RBDAs) like RNEA, ABA, CRBA, and Jacobians.
- Provides all the quantities included in the Euler-Poincarè formulation of the equations of motion.
- Supports body-fixed, inertial-fixed, and mixed [velocity representations][notation].
- Exposes all the necessary quantities to develop controllers in centroidal coordinates.

### JaxSim for robot learning

- Being developed with JAX, all the RBDAs support automatic differentiation both in forward and reverse modes.
- Support for automatically differentiating against kinematics and dynamics parameters.
- All fixed-step integrators are forward and reverse differentiable.
- All variable-step integrators are forward differentiable.
- Ideal for sampling synthetic data for reinforcement learning (RL).
- Ideal for designing physics-informed neural networks (PINNs) with loss functions requiring model-based quantities.
- Ideal for combining model-based control with learning-based components.

[jax]: https://github.com/google/jax/
[sdformat]: https://github.com/gazebosim/sdformat
[notation]: https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2
[passive_viewer_mujoco]: https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer

> [!WARNING]
> This project is still experimental, APIs could change between releases without notice.

> [!NOTE]
> JaxSim currently focuses on locomotion applications.
> Only contacts between bodies and smooth ground surfaces are supported.

## Documentation

The JaxSim API documentation is available at [jaxsim.readthedocs.io][readthedocs].

[readthedocs]: https://jaxsim.readthedocs.io/

## Installation

<details>
<summary>With conda</summary>

You can install the project using [`conda`][conda] as follows:

```bash
conda install jaxsim -c conda-forge
```

You can enforce GPU support, if needed, by also specifying `"jaxlib = * = *cuda*"`.

</details>

<details>
<summary>With pip</summary>

You can install the project using [`pypa/pip`][pip], preferably in a [virtual environment][venv], as follows:

```bash
pip install jaxsim
```

Check [`setup.cfg`](setup.cfg) for the complete list of optional dependencies.
You can obtain a full installation using `jaxsim[all]`.

If you need GPU support, follow the official [installation instructions][jax_gpu] of JAX.

</details>

<details>
<summary>Contributors installation</summary>

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

[conda]: https://anaconda.org/
[pip]: https://github.com/pypa/pip/
[venv]: https://docs.python.org/3/tutorial/venv.html
[jax_gpu]: https://github.com/google/jax/#installation

## Credits

The RBDAs are based on the theory of the [Rigid Body Dynamics Algorithms][RBDA]
book by Roy Featherstone.
The algorithms and some simulation features were inspired by its accompanying [code][spatial_v2].

[RBDA]: https://link.springer.com/book/10.1007/978-1-4899-7560-7
[spatial_v2]: http://royfeatherstone.org/spatial/index.html#spatial-software

The development of JaxSim started in late 2021, inspired by early versions of [`google/brax`][brax].
At that time, Brax was implemented in maximal coordinates, and we wanted a physics engine in reduced coordinates.
We are grateful to the Brax team for their work and showing the potential of [JAX][jax] in this field.

Brax v2 was later implemented reduced coordinates, following an approach comparable to JaxSim.
The development then shifted to [MJX][mjx], which today provides a JAX-based implementation of the Mujoco APIs.

The main differences between MJX/Brax and JaxSim are as follows:

- JaxSim supports out-of-the-box all SDF models with [Pose Frame Semantics][PFS].
- JaxSim only supports collisions between points rigidly attached to bodies and a compliant ground surface.
  Our contact model requires careful tuning of its spring-damper parameters, but being an instantaneous
  function of the state $(\mathbf{q}, \boldsymbol{\nu})$, it doesn't require running any optimization algorithm
  when stepping the simulation forward.
- JaxSim mitigates the stiffness of the contact-aware system dynamics by providing variable-step integrators.

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

## People

| Author | Maintainers |
|:------:|:-----------:|
| [<img src="https://avatars.githubusercontent.com/u/469199?v=4" width="40">][df] | [<img src="https://avatars.githubusercontent.com/u/102977828?v=4" width="40">][ff] [<img src="https://avatars.githubusercontent.com/u/469199?v=4" width="40">][df] |

[df]: https://github.com/diegoferigo
[ff]: https://github.com/flferretti

## License

[BSD3](https://choosealicense.com/licenses/bsd-3-clause/)
