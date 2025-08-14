import inspect
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # This ensures that loggers from external libraries are not disabled
    "formatters": {
        "standard": {
            "format": "\033[32m%(asctime)-8s %(levelname)-8s %(filename)-10s %(name)s: %(message)s\033[0m",
        },
        "verbose": {
            "format": "%(asctime)-8s %(levelname)-8s %(filename)-10s:%(lineno)-4d%(funcName)-19s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "verbose",
            "filename": "jaxns.log",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        # You can add more module-specific configurations here
        "jaxns": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
}

# logging.getLogger('transitions.core').setLevel(logging.CRITICAL)

logging.config.dictConfig(LOGGING_CONFIG)
jaxns_logger = logging.getLogger("jaxns")


def get_grandparent_info(relative_depth: int = 7):
    """
    Get the file, line number and function name of the caller of the caller of this function.

    Args:
        relative_depth: the number of frames to go back from the caller of this function. Default is 6. Should be
        enough to get out of a jax.tree.map call.

    Returns:
        str: a string with the file, line number and function name of the caller of the caller of this function.
    """
    # Get the grandparent frame (caller of the caller)
    s = []
    for depth in range(1, min(1 + relative_depth, len(inspect.stack()) - 1) + 1):
        caller_frame = inspect.stack()[depth]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno
        caller_func = caller_frame.function
        s.append(f"{os.path.basename(caller_file)}:{caller_line} in {caller_func}")
    s = s[::-1]
    s = f"at {' -> '.join(s)}"
    return s
