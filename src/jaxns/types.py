import jax
import numpy as np
from jaxctx.context import CtxParams

PRNGKey = jax.Array
# Type annotation for JAX and NumPy arrays, with no scalar types.
Array = jax.Array | np.ndarray
# Type annotation for JAX and NumPy arrays, including float scalars.
FloatArray = jax.Array | np.ndarray | float
# Type annotation for JAX and NumPy arrays, including integer scalars.
IntArray = jax.Array | np.ndarray | int
# Type annotation for JAX and NumPy arrays, including boolean scalars.
BoolArray = jax.Array | np.ndarray | np.bool_ | bool

UType = CtxParams  # Sample space type
XType = CtxParams  # Prior variable type
