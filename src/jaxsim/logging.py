import enum
import logging

import coloredlogs

LOGGER_NAME = "jaxsim"


class LoggingLevel(enum.IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def _logger() -> logging.Logger:
    return logging.getLogger(name=LOGGER_NAME)


def set_logging_level(level: int | LoggingLevel = LoggingLevel.WARNING):
    if isinstance(level, int):
        level = LoggingLevel(level)

    _logger().setLevel(level=level.value)


def get_logging_level() -> LoggingLevel:
    level = _logger().getEffectiveLevel()
    return LoggingLevel(level)


def configure(level: LoggingLevel = LoggingLevel.WARNING) -> None:
    info("Configuring the 'jaxsim' logger")

    handler = logging.StreamHandler()
    fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"
    handler.setFormatter(fmt=coloredlogs.ColoredFormatter(fmt=fmt))
    _logger().addHandler(hdlr=handler)

    # Do not propagate the messages to handlers of parent loggers
    # (preventing duplicate logging)
    _logger().propagate = False

    set_logging_level(level=level)


def debug(msg: str = "") -> None:
    _logger().debug(msg=msg)


def info(msg: str = "") -> None:
    _logger().info(msg=msg)


def warning(msg: str = "") -> None:
    _logger().warning(msg=msg)


def error(msg: str = "") -> None:
    _logger().error(msg=msg)


def critical(msg: str = "") -> None:
    _logger().critical(msg=msg)


def exception(msg: str = "") -> None:
    _logger().exception(msg=msg)
