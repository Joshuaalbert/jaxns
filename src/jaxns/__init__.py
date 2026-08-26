"""Lazy public conveniences for ordinary JAXNS model authoring.

Core runtime classes remain in their defining modules. ``Prior`` and
``special_priors`` are the small modeling vocabulary most users need before
they can construct those runtime objects, so they are also available here.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxctx.priors.prior import Prior as Prior

    from jaxns import special_priors as special_priors

__all__ = [
    "Prior",
    "special_priors",
]


def __getattr__(name: str) -> object:
    """Resolve common modeling objects without making root import expensive."""
    # The CLI imports the package before ``jaxns.cli``. Keep these aliases
    # lazy so configuration validation does not acquire JAX, TFP, ZeroMQ, or
    # model-serialization imports merely because Prior is public at the root.
    if name == "Prior":
        value = import_module("jaxctx.priors.prior").Prior
    elif name == "special_priors":
        value = import_module("jaxns.special_priors")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public objects in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
