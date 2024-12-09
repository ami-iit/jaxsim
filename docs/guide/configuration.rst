Configuration
=============

JaxSim utilizes environment variables for application configuration. Below is a detailed overview of the various configuration categories and their respective variables.


Collision Dynamics
~~~~~~~~~~~~~~~~~~

Environment variables starting with ``JAXSIM_COLLISION_`` are used to configure collision dynamics. The available variables are:

- ``JAXSIM_COLLISION_SPHERE_POINTS``: Specifies the number of collision points to approximate the sphere.
  *Default:* ``50``.

- ``JAXSIM_COLLISION_MESH_ENABLED``: Enables or disables mesh-based collision detection.
  *Default:* ``False``.

- ``JAXSIM_COLLISION_USE_BOTTOM_ONLY``: Limits collision detection to only the bottom half of the box or sphere.
  *Default:* ``False``.


Testing
~~~~~~~

For testing configurations, environment variables beginning with ``JAXSIM_TEST_`` are used. The following variables are available:

- ``JAXSIM_TEST_SEED``: Defines the seed for the random number generator.
  _Default: ``0``._

- ``JAXSIM_TEST_AD_ORDER``: Specifies the gradient order for automatic differentiation tests.
  _Default: ``1``._

- ``JAXSIM_TEST_FD_STEP_SIZE``: Sets the step size for finite difference tests.
  _Default: the cube root of the machine epsilon._


Joint Dynamics
~~~~~~~~~~~~~~
Joint dynamics are configured using environment variables starting with ``JAXSIM_JOINT_``. Available variables include:

- ``JAXSIM_JOINT_POSITION_LIMIT_DAMPER``: Defines the damper value for joint position limits.
  _Default: ``0``._

- ``JAXSIM_JOINT_POSITION_LIMIT_SPRING``: Defines the spring value for joint position limits.
  _Default: ``0``._


Logging
~~~~~~~

The logging configuration is controlled by the following environment variable:

- ``JAXSIM_LOGGING_LEVEL``: Determines the logging level.
  *Default:* ``DEBUG`` for development, ``WARNING`` for production.
