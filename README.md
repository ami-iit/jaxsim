# JAXsim

**A scalable physics engine and multibody dynamics library implemented with JAX. With JIT batteries 🔋**

> [!WARNING]
> This project is still experimental, APIs could change without notice.

> [!NOTE]
> This simulator currently focuses on locomotion applications. Only contacts with ground are supported.

## Features

- Physics engine in reduced coordinates implemented with [JAX][jax] in Python.
- JIT compilation of Python code for increased performance.
- Transparent support to execute logic on CPUs, GPUs, and TPUs.
- Parallel multi-body simulations on hardware accelerators for significantly increased throughput.
- Support for SDF models (and, upon conversion, URDF models).
- Collision detection between bodies and uneven ground surface.
- Soft contacts model supporting full friction cone and sticking / slipping transition.
- Complete support for inertial properties of rigid bodies.
- Revolute, prismatic, and fixed joints support.
- Integrators: forward Euler, semi-implicit Euler, Runge-Kutta 4.
- High-level classes for object-oriented programming.
- High-level classes to compute multi-body dynamics quantities from the simulation state.
- High-level classes wrapping the low-level functional RBDAs with support of [multiple velocities representations][notation].
- Default validation of JAX pytrees to prevent JIT re-compilations.
- Preliminary support for automatic differentiation of RBDAs.

[jax]: https://github.com/google/jax/
[notation]: https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2

## Documentation

The JAXsim API documentation is available at [jaxsim.readthedocs.io](https://jaxsim.readthedocs.io/).

## Installation

You can install the project using [`conda`][conda]:

```bash
conda install jaxsim -c conda-forge
```

Alternatively, you can use [`pypa/pip`][pip], preferably in a [virtual environment][venv]:

```bash
pip install jaxsim
```

Check [`setup.cfg`](setup.cfg) for the complete list of optional dependencies.
Install all of them with `jaxsim[all]`.

**Note:** For GPU support, follow the official [installation instructions][jax_gpu] of JAX.

[conda]: https://anaconda.org/
[pip]: https://github.com/pypa/pip/
[venv]: https://docs.python.org/3/tutorial/venv.html
[jax_gpu]: https://github.com/google/jax/#installation

## Quickstart

Explore and learn how to use the library through practical demonstrations available in the [examples](./examples) folder.

## Credits

The physics module of JAXsim is based on the theory of the [Rigid Body Dynamics Algorithms][RBDA]
book by Roy Featherstone.
We structured part of our logic following its accompanying [code][spatial_v2].
The physics engine is developed entirely in Python using [JAX][jax].

[RBDA]: https://link.springer.com/book/10.1007/978-1-4899-7560-7
[spatial_v2]: http://royfeatherstone.org/spatial/index.html#spatial-software

The inspiration for developing JAXsim originally stemmed from early versions of [`google/brax`][brax].
Here below we summarize the differences between the projects:

- JAXsim simulates multibody dynamics in reduced coordinates, while brax v1 uses maximal coordinates.
- The new v2 APIs of brax (and the new [MJX][mjx]) were then implemented in reduced coordinates, following an approach comparable to JAXsim, with major differences in contact handling.
- The rigid-body algorithms used in JAXsim allow to efficiently compute quantities based on the Euler-Poincarè
  formulation of the equations of motion, necessary for model-based robotics research.
- JAXsim supports SDF (and, indirectly, URDF) models, assuming the model is described with the
  recent [Pose Frame Semantics][PFS].
- Contrarily to brax, JAXsim only supports collision detection between bodies and a compliant ground surface.
- The RBDAs of JAXsim support automatic differentiation, but this functionality has not been thoroughly tested.

[brax]: https://github.com/google/brax
[mjx]: https://mujoco.readthedocs.io/en/3.0.0/mjx.html
[PFS]: http://sdformat.org/tutorials?tut=pose_frame_semantics

## Contributing

Pull requests are welcome. 
For major changes, please open an issue first to discuss what you would like to change.

## Citing

```bibtex
@software{ferigo_jaxsim_2022,
  author = {Diego Ferigo and Silvio Traversaro and Daniele Pucci},
  title = {{JAXsim}: A Physics Engine in Reduced Coordinates and Multibody Dynamics Library for Control and Robot Learning},
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
