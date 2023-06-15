import logging
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger('jaxns')


def deprecated(other: Optional[Callable]) -> Callable:
    def decorator(f: Callable):
        @wraps
        def g(*args, **kwargs):
            if other is not None:
                logger.warning(f"{f.__name__} is deprecated, use `{other.__name__}`.")
            else:
                logger.warning(f"{f.__name__} is deprecated.")
            return f(*args, **kwargs)

        if other is not None:
            g.__doc__ = (f"Deprecated. Prefer {other.__name__}.\n\n{g.__doc__}")
        else:
            g.__doc__ = (f"Deprecated.\n\n{g.__doc__}")

        return g

    return decorator
