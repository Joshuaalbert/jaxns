import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "quick_unit",
    "quick_unit_inverse"
]


def quick_unit(x: jax.Array) -> jax.Array:
    """
    Quick approximation to the sigmoid.

    Args:
        x: jax.Array value in (-inf, inf) open interval

    Returns:
        value in (0, 1) in open interval
    """
    return 0.5 * (x / (1 + lax.abs(x)) + 1)


def quick_unit_inverse(y: jax.Array) -> jax.Array:
    """
    Inverse of quick_unit.

    Args:
        y: jax.Array value in (0, 1) open interval

    Returns:
        value in (-inf, inf) in open interval
    """
    twoy = y + y

    return jnp.where(
        y >= 0.5,
        (1 - twoy) / (twoy - 2),
        1 - lax.reciprocal(twoy)
    )
