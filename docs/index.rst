JAXsim
#######

A scalable physics engine and multibody dynamics library implemented with JAX. With JIT batteries 🔋

.. warning::
    This project is still experimental, APIs could change without notice.

.. note::
    This simulator currently focuses on locomotion applications. Only contacts with ground are supported.

Features
--------

- Physics engine in reduced coordinates implemented with `JAX <https://github.com/google/jax/>`_ in Python.
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
- High-level classes to compute multi-body dynamics quantities from simulation state.
- High-level classes wrapping the low-level functional RBDAs with support of `multiple velocities representations <https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2>`_.
- Default validation of JAX pytrees to prevent JIT re-compilations.
- Preliminary support for automatic differentiation of RBDAs.

.. toctree::
  :hidden:
  :maxdepth: 1
  :caption: User Guide

  guide/install

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: JAXsim API

  modules/high_level
  modules/math
  modules/parsers
  modules/physics
  modules/simulation
  modules/typing
  modules/utils


Examples
--------

Explore and learn how to use the library through practical demonstrations available in the `examples <https://github.com/ami-iit/jaxsim/tree/main/examples>`__ folder.

Credits
-------

The physics module of JAXsim is based on the theory of the `Rigid Body Dynamics Algorithms <https://link.springer.com/book/10.1007/978-1-4899-7560-7>`_ book by Roy Featherstone.
We structured part of our logic following its accompanying `code <http://royfeatherstone.org/spatial/index.html#spatial-software>`_.
The physics engine is developed entirely in Python using `JAX <https://github.com/google/jax/>`_.

The inspiration for developing JAXsim originally stemmed from early versions of `google/brax <https://github.com/google/brax>`_.
Here below we summarize the differences between the projects:

- JAXsim simulates multibody dynamics in reduced coordinates, while brax v1 uses maximal coordinates.
- The new v2 APIs of brax (and the new `MJX <https://mujoco.readthedocs.io/en/3.0.0/mjx.html>`_) were then implemented in reduced coordinates, following an approach comparable to JAXsim, with major differences in contact handling.
- The rigid-body algorithms used in JAXsim allow to efficiently compute quantities based on the Euler-Poincarè
  formulation of the equations of motion, necessary for model-based robotics research.
- JAXsim supports SDF (and, indirectly, URDF) models, assuming the model is described with the
  recent `Pose Frame Semantics <http://sdformat.org/tutorials?tut=pose_frame_semantics>`_.
- Contrarily to brax, JAXsim only supports collision detection between bodies and a compliant ground surface.
- The RBDAs of JAXsim support automatic differentiation, but this functionality has not been thoroughly tested.

Contributing
------------

Pull requests are welcome. 
For major changes, please open an issue first to discuss what you would like to change.

Citing
------

.. code-block:: bibtex

    @software{ferigo_jaxsim_2022,
      author = {Diego Ferigo and Silvio Traversaro and Daniele Pucci},
      title = {{JAXsim}: A Physics Engine in Reduced Coordinates and Multibody Dynamics Library for Control and Robot Learning},
      url = {http://github.com/ami-iit/jaxsim},
      year = {2022},
    }

People
------

Author and Maintainer
'''''''''''''''''''''

`Diego Ferigo <https://github.com/diegoferigo>`_

Maintainer
''''''''''

`Filippo Luca Ferretti <https://github.com/flferretti>`_

License
-------

`BSD3 <https://choosealicense.com/licenses/bsd-3-clause/>`_
