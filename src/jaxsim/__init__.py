from . import logging
from ._version import __version__


# Follow upstream development in https://github.com/google/jax/pull/13304
def _jnp_options() -> None:
    import os

    import jax

    # Enable by default 64bit precision in JAX.
    if os.environ.get("JAX_ENABLE_X64", "1") != "0":

        logging.info("Enabling JAX to use 64bit precision")
        jax.config.update("jax_enable_x64", True)

        import jax.numpy as jnp
        import numpy as np

        if jnp.empty(0, dtype=float).dtype != jnp.empty(0, dtype=np.float64).dtype:
            logging.warning("Failed to enable 64bit precision in JAX")

    else:
        logging.warning(
            "Using 32bit precision in JaxSim is still experimental, please avoid to use variable step integrators."
        )


def _np_options() -> None:
    import numpy as np

    np.set_printoptions(precision=5, suppress=True, linewidth=150, threshold=10_000)


def _is_editable() -> bool:

    import importlib.util
    import pathlib
    import site

    # Get the ModuleSpec of jaxsim.
    jaxsim_spec = importlib.util.find_spec(name="jaxsim")

    # This can be None. If it's None, assume non-editable installation.
    if jaxsim_spec.origin is None:
        return False

    # Get the folder containing the jaxsim package.
    jaxsim_package_dir = str(pathlib.Path(jaxsim_spec.origin).parent.parent)

    # The installation is editable if the package dir is not in any {site|dist}-packages.
    return jaxsim_package_dir not in site.getsitepackages()


def _get_default_logging_level(env_var: str) -> logging.LoggingLevel:
    """
    Get the default logging level.

    Args:
        env_var: The environment variable to check.

    Returns:
        The logging level to set.
    """

    import os

    # Define the default logging level depending on the installation mode.
    default_logging_level = (
        logging.LoggingLevel.DEBUG
        if _is_editable()  # noqa: F821
        else logging.LoggingLevel.WARNING
    )

    # Allow to override the default logging level with an environment variable.
    try:
        return logging.LoggingLevel[
            os.environ.get(env_var, default_logging_level.name).upper()
        ]

    except KeyError as exc:
        msg = f"Invalid logging level defined in {env_var}='{os.environ[env_var]}'"
        raise RuntimeError(msg) from exc


# Configure the logger with the default logging level.
logging.configure(level=_get_default_logging_level(env_var="JAXSIM_LOGGING_LEVEL"))


# Configure JAX.
_jnp_options()

# Initialize the numpy print options.
_np_options()

del _jnp_options
del _np_options
del _get_default_logging_level
del _is_editable

from . import terrain  # isort:skip
from . import api, integrators, logging, math, rbda
from .api.common import VelRepr
