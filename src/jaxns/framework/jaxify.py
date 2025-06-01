import warnings
from typing import Callable

import jax
import numpy as np

from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import LikelihoodType

__all__ = [
    'jaxify_likelihood'
]


def jaxify_likelihood(log_likelihood: Callable[..., np.ndarray], vectorised: bool = False) -> LikelihoodType:
    """
    Wraps a non-JAX log likelihood function.

    Args:
        log_likelihood: a non-JAX log-likelihood function, which accepts a number of arguments and returns a scalar
            log-likelihood.
        vectorised: if True then the `log_likelihood` must handle batched inputs, i.e. each input will receive a common
            set of batched dimensions which the function must handle.

    Returns:
        A JAX-compatible log-likelihood function.
    """
    warnings.warn(
        "You're using a non-JAX log-likelihood function. This may be slower than a JAX log-likelihood function. "
        "Also, you are responsible for ensuring that the function is deterministic. "
        "Also, you cannot use learnable parameters in the likelihood call."
    )

    def _cond_cast(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return np.asarray(x)
        return x

    def _casted_log_likelihood(*args) -> np.ndarray:
        args = jax.tree.map(_cond_cast, args)  # Convert all arguments to numpy arrays, as they now pass jax.Array
        return mp_policy.cast_to_measure(log_likelihood(*args))

    def _log_likelihood(*args) -> jax.Array:
        # Define the expected shape & dtype of output.
        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(),
            dtype=mp_policy.measure_dtype
        )
        if vectorised:
            return jax.pure_callback(_casted_log_likelihood, result_shape_dtype, *args, vmap_method='broadcast_all')
        return jax.pure_callback(_casted_log_likelihood, result_shape_dtype, *args, vmap_method=None)

    return _log_likelihood
