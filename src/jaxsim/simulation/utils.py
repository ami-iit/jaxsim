from typing import Tuple

from jaxsim import logging


def check_valid_shape(
    what: str, shape: Tuple, expected_shape: Tuple, valid: bool
) -> bool:
    valid_shape = shape == expected_shape

    if not valid_shape:
        logging.debug(f"Shape of {what} differs: {shape}, {expected_shape}")
        raise

    return valid
