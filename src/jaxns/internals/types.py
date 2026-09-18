from typing import NamedTuple, Union, Any, Callable, Tuple, Dict, TypeVar

import jax
import numpy as np

__all__ = [
    'PRNGKey',
    'IntArray',
    'FloatArray',
    'BoolArray',
    'LikelihoodType',
    'UType',
    'XType',
    'LikelihoodInputType',
    'RandomVariableType',
    'MeasureType'
]

PRNGKey = jax.Array

# Type annotation for JAX and NumPy arrays, with no scalar types.
Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
# Type annotation for JAX and NumPy arrays, including float scalars.
FloatArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
# Type annotation for JAX and NumPy arrays, including integer scalars.
IntArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
# Type annotation for JAX and NumPy arrays, including boolean scalars.
BoolArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]

LikelihoodType = Callable[..., FloatArray]
RandomVariableType = TypeVar('RandomVariableType')
MeasureType = TypeVar('MeasureType')
LikelihoodInputType = Union[Tuple[Any, ...], Any]  # Likelihood conditional variables
UType = jax.Array  # Sample space type
WType = Tuple[jax.Array, ...]
XType = Dict[str, RandomVariableType]  # Prior variable type


class SignedLog(NamedTuple):
    """
    Represents a signed value in log-space
    """
    log_abs_val: jax.Array
    sign: Union[jax.Array, Any]


def isinstance_namedtuple(obj) -> bool:
    """
    Check if object is a namedtuple.

    Args:
        obj: object

    Returns:
        bool
    """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )
