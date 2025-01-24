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

.. note::
  The bottom half is defined as the half of the box or sphere with the lowest z-coordinate in the collision link frame.


Testing
~~~~~~~

For testing configurations, environment variables beginning with ``JAXSIM_TEST_`` are used. The following variables are available:

- ``JAXSIM_TEST_SEED``: Defines the seed for the random number generator.

  *Default:* ``0``.

- ``JAXSIM_TEST_AD_ORDER``: Specifies the gradient order for automatic differentiation tests.

  *Default:* ``1``.

- ``JAXSIM_TEST_FD_STEP_SIZE``: Sets the step size for finite difference tests.

  *Default:* the cube root of the machine epsilon.


Joint Dynamics
~~~~~~~~~~~~~~
Joint dynamics are configured using environment variables starting with ``JAXSIM_JOINT_``. Available variables include:

- ``JAXSIM_JOINT_POSITION_LIMIT_DAMPER``: Overrides the damper value for joint position limits of the SDF model.

- ``JAXSIM_JOINT_POSITION_LIMIT_SPRING``: Overrides the spring value for joint position limits of the SDF model.


Logging and Exceptions
~~~~~~~~~~~~~~~~~~~~~~

The logging and exceptions configurations is controlled by the following environment variables:

- ``JAXSIM_LOGGING_LEVEL``: Determines the logging level.

  *Default:* ``DEBUG`` for development, ``WARNING`` for production.

- ``JAXSIM_ENABLE_EXCEPTIONS``: Enables the runtime checks and exceptions. Note that enabling exceptions might lead to device-to-host transfer of data, increasing the computational time required.

  *Default:* ``False``.

.. note::
    Runtime exceptions are disabled by default on TPU.
