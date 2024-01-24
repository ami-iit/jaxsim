from . import logging
from ._version import __version__


# Follow upstream development in https://github.com/google/jax/pull/13304
def _jnp_options() -> None:
    import os

    from jax.config import config

    # Enable by default
    if not ("JAX_ENABLE_X64" in os.environ and os.environ["JAX_ENABLE_X64"] == "0"):
        logging.info("Enabling JAX to use 64bit precision")
        config.update("jax_enable_x64", True)

        import jax.numpy as jnp
        import numpy as np

        if jnp.empty(0, dtype=float).dtype != jnp.empty(0, dtype=np.float64).dtype:
            logging.warning("Failed to enable 64bit precision in JAX")


def _np_options() -> None:
    import numpy as np

    np.set_printoptions(precision=5, suppress=True, linewidth=150, threshold=10_000)


def _is_editable() -> bool:
    import importlib.util
    import pathlib
    import site

    # Get the ModuleSpec of jaxsim
    jaxsim_spec = importlib.util.find_spec(name="jaxsim")

    # This can be None. If it's None, assume non-editable installation.
    if jaxsim_spec.origin is None:
        return False

    # Get the folder containing the jaxsim package
    jaxsim_package_dir = str(pathlib.Path(jaxsim_spec.origin).parent.parent)

    # The installation is editable if the package dir is not in any {site|dist}-packages
    return jaxsim_package_dir not in site.getsitepackages()


# Initialize the logging verbosity
if _is_editable():
    logging.configure(level=logging.LoggingLevel.DEBUG)
else:
    logging.configure(level=logging.LoggingLevel.WARNING)

# Configure JAX
_jnp_options()

# Initialize the numpy print options
_np_options()

del _jnp_options
del _np_options
del _is_editable

from . import high_level, logging, math, simulation, sixd
from .high_level.common import VelRepr
from .simulation.ode_integration import IntegratorType
from .simulation.simulator import JaxSim
