import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # This ensures that loggers from external libraries (like Ray) are not disabled
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
            "filename": "application.log",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "jaxns": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
jaxns_logger = logging.getLogger("jaxns")
