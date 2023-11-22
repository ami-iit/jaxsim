Installation
============

.. _installation:

Prerequisites
-------------

JAXsim requires Python 3.10 or later. 

Basic Installation
------------------

You can install the project with `pypa/pip`, preferably in a `virtual environment`_:

.. code-block:: bash

   pip install jaxsim

Have a look to `setup.cfg` for a complete list of optional dependencies.
You can install all of them by specifying ``jaxsim[all]``.

.. note::

    If you need GPU support, please follow the official `installation instruction`_ of JAX.

.. _pypa/pip: https://github.com/pypa/pip/
.. _virtual environment: https://docs.python.org/3.8/tutorial/venv.html
.. _installation instruction: https://github.com/google/jax/#installation
