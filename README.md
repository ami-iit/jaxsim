# JAXsim

**A scalable physics engine implemented with JAX. With JIT batteries 🔋**

⚠ This project is still experimental, APIs could change without notice. ️⚠

⚠ This simulator currently focuses on locomotion applications. Only contacts with ground are supported. ️⚠

## Features

- Physics engine in reduced coordinates implemented with [JAX][jax] in Python.
- Supported JIT compilation of Python code for increased performance.
- Transparent support to execute the simulation on CPUs, GPUs, and TPUs.
- Possibility to run parallel multi-body simulations on hardware accelerators for significantly increased throughput.
- Support of SDF models (and, upon conversion, URDF models).
- Collision detection between bodies and uneven ground surface.
- Continuous soft contacts model with no friction cone approximations.
- Full support of inertial properties of bodies.
- Revolute, prismatic, and fixed joints support.
- Integrators: forward Euler, semi-implicit Euler, Runge-Kutta 4.
- High-level classes to compute multi-body dynamics quantities from simulation state.
- High-level classes supporting both object-oriented and functional programming.
- Optional validation of JAX pytrees to prevent JIT re-compilation. 

Planned features:

- Reinforcement Learning module developed in JAX.
- Finalization of differentiable physics through the simulation.

[jax]: https://github.com/google/jax/

## Installation

You can install the project with [`pypa/pip`][pip], preferably in a [virtual environment][venv]:

```bash
pip install jaxsim
```

Have a look to [`setup.cfg`](setup.cfg) for a complete list of optional dependencies.
You can install all of them by specifying `jaxsim[all]`.

**Note:** if you need GPU support, please follow the official [installation instruction][jax_gpu] of JAX.

[pip]: https://github.com/pypa/pip/
[venv]: https://docs.python.org/3.8/tutorial/venv.html
[jax_gpu]: https://github.com/google/jax/#installation

## Credits

The physics module of JAXsim is based on the theory of the [Rigid Body Dynamics Algorithms][RBDA]
book authored by Roy Featherstone.
We structured part of our logic following its accompanying [code][spatial_v2].
The physics engine is developed entirely in Python using [JAX][jax].

[RBDA]: https://link.springer.com/book/10.1007/978-1-4899-7560-7
[spatial_v2]: http://royfeatherstone.org/spatial/index.html#spatial-software

The inspiration of developing JAXsim stems from [`google/brax`][brax].
Here below we summarize the differences between the projects:

- JAXsim simulates multibody dynamics in reduced coordinates, while `brax` uses maximal coordinates.
- The rigid body algorithms used in JAXsim allow to efficiently compute quantities based on the Euler-Poincarè
  formulation of the equations of motion, necessary for model-based robotics research.
- JAXsim supports SDF (and, indirectly, URDF) models, under the assumption that the model is described with the
  recent [Pose Frame Semantics][PFS].
- Contrarily to `brax`, JAXsim only supports collision detection between bodies and a compliant ground surface.
- While supported thanks to the usage of JAX, differentiating through the simulator has not yet been studied.

[brax]: https://github.com/google/brax
[PFS]: http://sdformat.org/tutorials?tut=pose_frame_semantics

## Contributing

Pull requests are welcome. 
For major changes, please open an issue first to discuss what you would like to change.

## Citing

```bibtex
@software{ferigo_jaxsim_2022,
  author = {Diego Ferigo and Silvio Traversaro and Daniele Pucci},
  title = {{JAXsim}: A Physics Engine in Reduced Coordinates for Control and Robot Learning},
  url = {http://github.com/ami-iit/jaxsin},
  year = {2022},
}
```

## Maintainers

| [<img src="https://github.com/diegoferigo.png" width="40">][df] | [@diegoferigo][df] |
|:---------------------------------------------------------------:|:------------------:|

[df]: https://github.com/diegoferigo

## License

[BSD3](https://choosealicense.com/licenses/bsd-3-clause/)
