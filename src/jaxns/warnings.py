import warnings
from functools import wraps
from typing import Callable, Optional


def deprecated(other: Optional[Callable]) -> Callable:
    def decorator(f: Callable):
        @wraps
        def g(*args, **kwargs):
            if other is not None:
                warnings.warn(f"{f.__name__} is deprecated, use `{other.__name__}`.")
            else:
                warnings.warn(f"{f.__name__} is deprecated.")
            return f(*args, **kwargs)

        if other is not None:
            g.__doc__ = (f"Deprecated. Prefer {other.__name__}.\n\n{g.__doc__}")
        else:
            g.__doc__ = (f"Deprecated.\n\n{g.__doc__}")

        return g

    return decorator
