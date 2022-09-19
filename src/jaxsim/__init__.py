from . import high_level, logging, math, sixd


def _np_options():
    import numpy as np

    np.set_printoptions(precision=5, suppress=True, linewidth=150, threshold=10_000)


def _is_editable():

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

# Initialize the numpy print options
_np_options()

del _np_options
del _is_editable
