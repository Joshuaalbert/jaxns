from jax import numpy as jnp
from jax.lax import broadcast_shapes
import numpy as np
from jaxns.prior_transforms.prior_chain import Prior
import logging

logger = logging.getLogger(__name__)

def check_broadbast_shapes(to_shape, *shapes):
    result_shape = broadcast_shapes(*shapes)
    if len(result_shape) == len(to_shape):
        if tuple(result_shape) == tuple(to_shape):
            return True
    return False

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

def prior_docstring(f):
    """
    Puts the Prior docstring below each prior init.
    Args:
        f: callable
    """
    if f.__doc__ is None:
        logger.warning("{} has no docstring".format(f.__name__))
        f.__doc__ = ""
    f.__doc__ = f.__doc__+"\n\nGeneral Prior documentation:\n\n"+Prior.__init__.__doc__
    return f

def get_shape(v):
    """
    Gets shape from a value regardless of what it might be.

    Args:
        v: Prior, array, list, tuple, scalar

    Returns: tuple of shape
    """
    if isinstance(v, Prior):
        return v.shape
    if isinstance(v, (jnp.ndarray, np.ndarray)):
        return v.shape
    if isinstance(v, (list, tuple)):
        return np.asarray(v).shape
    return ()
