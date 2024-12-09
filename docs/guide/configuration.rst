Configuration
=============

.. _configuration:

Environment Variables
---------------------

JaxSim uses environment variables to configure the application.

Collision Dynamics
~~~~~~~~~~~~~~~~~~

The environment variables used to configure the collision dynamics start with
``JAXSIM_COLLISION_``. The following variables are available:

- ``JAXSIM_COLLISION_SPHERE_POINTS``: The number of collision points to use to approximate the sphere. *The default is ``50``.*
- ``JAXSIM_COLLISION_MESH_ENABLED``: Enable or disable the mesh collision. *The default is ``False``.*
- ``JAXSIM_COLLISION_USE_BOTTOM_ONLY``: Use only the bottom half of the box or sphere for collision detection. *The default is ``False``.*

Testing
~~~~~~~

The environment variables used to configure the testing start with
``JAXSIM_TEST_``. The following variables are available:

- ``JAXSIM_TEST_SEED``: The seed to use for the random number generator. *The default is ``0``.*
- ``JAXSIM_TEST_AD_ORDER``: The gradient order to use for the automatic differentiation tests. *The default is ``1``.*
- ``JAXSIM_TEST_FD_STEP_SIZE``: The step size to use for the finite difference tests. *The default is the cube root of the machine epsilon.*

Joint Dynamics
~~~~~~~~~~~~~~

The environment variables used to configure the joint dynamics start with
``JAXSIM_JOINT_``. The following variables are available:

- ``JAXSIM_JOINT_POSITION_LIMIT_DAMPER``: The damper value for the joint position limit. *The default is ``0``.*
- ``JAXSIM_JOINT_POSITION_LIMIT_SPRING``: The spring value for the joint position limit. *The default is ``0``.*

Logging
~~~~~~~

The environment variable ``JAXSIM_LOGGING_LEVEL`` can be used to set the logging level.
The default is ``DEBUG`` for development and ``WARNING`` for production.
