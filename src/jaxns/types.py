from typing import Union

import jax
import numpy as np
from jaxctx.context import CtxParams


def _strip_dotted_prefix(name: str) -> str:
    return name[1:] if name.startswith('.') else name


if not hasattr(CtxParams, 'iter_items'):
    def _iter_items(self):
        for name, value in self.items():
            yield _strip_dotted_prefix(name), value

    CtxParams.iter_items = _iter_items


if not hasattr(CtxParams, 'get_dotted'):
    def _get_dotted(self, name: str):
        if name in self._dict:
            return self._dict[name]
        dotted_name = name if name.startswith('.') else f".{name}"
        return self._dict[dotted_name]

    CtxParams.get_dotted = _get_dotted

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

UType = CtxParams  # Sample space type
XType = CtxParams  # Prior variable type

Array.__doc__ = "Type annotation for JAX array-like objects, with no scalar types."

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."
