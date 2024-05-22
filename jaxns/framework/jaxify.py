import warnings
from typing import Callable

import jax
import numpy as np

from jaxns.internals.types import float_type, LikelihoodType

__all__ = [
    'jaxify_likelihood'
]

def jaxify_likelihood(log_likelihood: Callable[..., np.ndarray], vectorised: bool = False) -> LikelihoodType:
    """
    Wraps a non-JAX log likelihood function.

    Args:
        log_likelihood: a non-JAX log-likelihood function, which accepts a number of arguments and returns a scalar
            log-likelihood.
        vectorised: if True then the `log_likelihood` performs a vectorised computation for leading batch dimensions,
            i.e. if a leading batch dimension is added to all input arguments, then it returns a vector of
            log-likelihoods with the same leading batch dimension.

    Returns:
        A JAX-compatible log-likelihood function.
    """
    warnings.warn(
        "You're using a non-JAX log-likelihood function. This may be slower than a JAX log-likelihood function. "
        "Also, you are responsible for ensuring that the function is deterministic. "
        "Also, you cannot use learnable parameters in the likelihood call."
    )

    def _casted_log_likelihood(*args) -> np.ndarray:
        return np.asarray(log_likelihood(*args), dtype=float_type)

    def _log_likelihood(*args) -> jax.Array:
        # Define the expected shape & dtype of output.
        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(),
            dtype=float_type
        )
        return jax.pure_callback(_casted_log_likelihood, result_shape_dtype, *args, vectorized=vectorised)

    return _log_likelihood
