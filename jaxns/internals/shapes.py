import numpy as np
from jax import numpy as jnp


def broadcast_dtypes(*dtypes):
    """
    Returns the dtype with highest precision.

    Args:
        *dtypes: list of JAX dtypes.

    Returns: dtype
    """
    levels = [jnp.bool_, jnp.int32, jnp.int64, jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    output = -1
    for dtype in dtypes:
        if dtype not in levels:
            raise ValueError("dtype {dtype} not in list {levels}.")
        output = max(output, levels.index(dtype))
    return levels[output]


def convert_to_array(v):
    """
    If necessary convert v to a jnp.ndarray.
    Passes through Prior.

    Args:
        v: array-like or scalar

    Returns: jnp.ndarray
    """
    if isinstance(v, (list, tuple, np.ndarray, float, int, bool, complex)):
        return jnp.asarray(v)
    return v


def tuple_prod(t):
    """
    Product of shape tuple

    Args:
        t: tuple

    Returns:
        int
    """
    if len(t) == 0:
        return 1
    res = t[0]
    for a in t[1:]:
        res *= a
    return res


def broadcast_shapes(shape1, shape2):
    """
    Broadcasts two shapes together.

    Args:
        shape1: tuple of int
        shape2: tuple of int

    Returns: tuple of int with resulting shape.
    """
    if isinstance(shape1, int):
        shape1 = (shape1,)
    if isinstance(shape2, int):
        shape2 = (shape2,)

    def left_pad_shape(shape, l):
        return tuple([1] * l + list(shape))

    l = max(len(shape1), len(shape2))
    shape1 = left_pad_shape(shape1, l - len(shape1))
    shape2 = left_pad_shape(shape2, l - len(shape2))
    out_shape = []
    for s1, s2 in zip(shape1, shape2):
        m = max(s1, s2)
        if ((s1 != m) and (s1 != 1)) or ((s2 != m) and (s2 != 1)):
            raise ValueError("Trying to broadcast {} with {}".format(shape1, shape2))
        out_shape.append(m)
    return tuple(out_shape)
