import jax
import jax.numpy as jnp
from jax import lax
import math

__all__ = [
    "quick_unit",
    "quick_unit_inverse"
]



_A = 2.0 / math.pi
_SQRT_A = math.sqrt(_A)

def quick_unit(x: jax.Array) -> jax.Array:
    """
    Smooth, fast CDF-like map approximating Φ(x).

    Args:
        x: [...] in (-inf, inf)

    Returns:
        [...] y in (0, 1)
    """
    return 0.5 * (1.0 + _SQRT_A * x / jnp.sqrt(1.0 + _A * x * x))


def quick_unit_inverse(y: jax.Array) -> jax.Array:
    """
    Approximate probit (inverse CDF) for quick_unit.

    Args:
        y: [...] (0, 1)

    Returns:
        [...] x in (-inf, inf)
    """
    eps = 1e-12  # avoid hitting sqrt(0) at exactly 0 or 1
    y = jnp.clip(y, eps, 1.0 - eps)
    t = 2.0 * y - 1.0
    return t / (_SQRT_A * jnp.sqrt(1.0 - t * t))
