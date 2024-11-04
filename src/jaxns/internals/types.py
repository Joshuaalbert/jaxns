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

Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
FloatArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
IntArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
BoolArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]

Array.__doc__ = "Type annotation for JAX array-like objects, with no scalar types."

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."

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
