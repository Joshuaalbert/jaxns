import dataclasses
from typing import Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def apply_interp(x: jax.Array, i0: jax.Array, alpha0: jax.Array, i1: jax.Array, alpha1: jax, axis: int = 0):
    """
    Apply interpolation alpha given axis.

    Args:
        x: nd-array
        i0: [N] or scalar
        alpha0: [N] or scalar
        i1: [N] or scalar
        alpha1: [N] or scalar
        axis: axis to take along

    Returns:
        [N] or scalar interpolated along axis
    """

    def take(i):
        num_dims = len(np.shape(x))
        # [0] [1] [2 3 4], num_dims=5, axis=1
        slices = [slice(None)] * axis + [i] + [slice(None)] * (num_dims - axis - 1)
        # return jnp.take(x, i, axis=axis)
        return x[tuple(slices)]

    return left_broadcast_multiply(take(i0), alpha0, axis=axis) + left_broadcast_multiply(
        take(i1), alpha1, axis=axis)


def left_broadcast_multiply(x, y, axis: int = 0):
    """
    Left broadcast multiply of two arrays.
    Equivalent to right-padding before multiply

    Args:
        x: [..., a,b,c,...]
        y: [a, b]

    Returns:
        [..., a, b, c, ...]
    """
    needed_length = len(np.shape(x)[axis:])
    len_y = len(np.shape(y))
    extra = needed_length - len_y
    if extra < 0:
        raise ValueError(f"Shape mismatch {np.shape(x)} x {np.shape(y)}.")
    y = lax.reshape(y, np.shape(y) + (1,) * extra)
    return x * y


def get_interp_indices_and_weights(x, xp, regular_grid: bool = False) -> Tuple[
    Tuple[Union[int, jax.Array, float, jax.Array]], Tuple[Union[int, jax.Array, float, jax.Array]]]:
    """
    One-dimensional linear interpolation. Outside bounds is also linear from nearest two points.

    Args:
        x: the x-coordinates at which to evaluate the interpolated values
        xp: the x-coordinates of the data points, must be increasing

    Returns:
        the interpolated values, same shape as `x`
    """

    x = jnp.asarray(x, dtype=jnp.float_)
    xp = jnp.asarray(xp, dtype=jnp.float_)
    if len(np.shape(xp)) == 0:
        xp = jnp.reshape(xp, (-1,))
    if np.shape(xp)[0] == 0:
        raise ValueError("xp must be non-empty")
    if np.shape(xp)[0] == 1:
        return (jnp.zeros_like(x, dtype=jnp.int32), jnp.ones_like(x)), (
            jnp.zeros_like(x, dtype=jnp.int32), jnp.zeros_like(x))

    # Find xp[i1-1] < x <= xp[i1]
    if regular_grid:
        # Use faster index determination
        delta_x = xp[1] - xp[0]
        i1 = jnp.clip((jnp.ceil((x - xp[0]) / delta_x)).astype(jnp.int64), 1, len(xp) - 1)
        i0 = i1 - 1
    else:
        i1 = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
        i0 = i1 - 1

    dx = xp[i1] - xp[i0]
    delta = x - xp[i0]

    epsilon = np.spacing(np.finfo(xp.dtype).eps)
    dx0 = jnp.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    dx = jnp.where(dx0, 1, dx)
    alpha = delta / dx
    return (i0, (1. - alpha)), (i1, alpha)


@dataclasses.dataclass(eq=False)
class InterpolatedArray:
    x: jax.Array  # [N]
    values: jax.Array  # [..., N, ...] `axis` has N elements

    axis: int = 0
    regular_grid: bool = False

    def __post_init__(self):

        if len(np.shape(self.x)) != 1:
            raise ValueError(f"Times must be 1D, got {np.shape(self.x)}.")

        def _assert_shape(x):
            if np.shape(x)[self.axis] != np.size(self.x):
                raise ValueError(f"Input values must have time length on `axis` dimension, got {np.shape(x)}.")

        jax.tree.map(_assert_shape, self.values)

        self.x, self.values = jax.tree.map(jnp.asarray, (self.x, self.values))

    @property
    def shape(self):
        return jax.tree.map(lambda x: np.shape(x)[:self.axis] + np.shape(x)[self.axis + 1:], self.values)

    def __call__(self, time: jax.Array) -> jax.Array:
        """
        Interpolate at time based on input times.

        Args:
            time: time to evaluate at.

        Returns:
            value at given time
        """
        (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(time, self.x, regular_grid=self.regular_grid)
        return jax.tree.map(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=self.axis), self.values)
