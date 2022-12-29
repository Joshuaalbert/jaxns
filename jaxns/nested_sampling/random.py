from typing import Optional

from etils.array_types import FloatArray, PRNGKey
from jax import random, numpy as jnp
from jax._src.scipy.special import logsumexp

from jaxns.nested_sampling.internals.log_semiring import cumulative_logsumexp

__all__ = ['random_ortho_matrix',
           'resample_indicies']


def random_ortho_matrix(key, n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.
    From https://stackoverflow.com/a/38430739

    Args:
        key: PRNG seed
        n: Size of matrix, draws from O(num_options) group.

    Returns: random [num_options,num_options] matrix with determinant = +-1
    """
    H = random.normal(key, shape=(n, n))
    Q, R = jnp.linalg.qr(H)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q


def resample_indicies(key: PRNGKey, log_weights: Optional[FloatArray] = None, S: Optional[int] = None,
                      replace: bool = False, num_total: Optional[int] = None):
    """
    resample the samples with weights which are interpreted as log_probabilities.
    Args:
        samples:
        weights:

    Returns: S samples of equal weight

    """
    if S is None:
        if log_weights is None:
            raise ValueError("Need log_weights if S is not given.")
        # ESS = (sum w)^2 / sum w^2
        S = int(jnp.exp(2. * logsumexp(log_weights) - logsumexp(2. * log_weights)))

    if replace:
        if log_weights is not None:
            # use cumulative_logsumexp because some log_weights could be really small
            log_p_cuml = cumulative_logsumexp(log_weights)
            log_r = log_p_cuml[-1] + jnp.log(1. - random.uniform(key, (S,)))
            idx = jnp.searchsorted(log_p_cuml, log_r)
        else:
            if num_total is None:
                raise ValueError("Need num_total if log_weights is None.")
            log_p_cuml = jnp.log(jnp.arange(num_total))
            log_r = log_p_cuml[-1] + jnp.log(1. - random.uniform(key, (S,)))
            idx = jnp.searchsorted(log_p_cuml, log_r)
    else:
        if log_weights is not None:
            g = -random.gumbel(key, shape=log_weights.shape) - log_weights
        else:
            if num_total is None:
                raise ValueError("Need num_total if log_weights is None.")
            g = -random.gumbel(key, shape=(num_total,))
        idx = jnp.argsort(g)[:S]
    return idx
