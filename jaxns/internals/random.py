from typing import Optional

from jax import random, numpy as jnp
from jax.scipy import special

from jaxns.internals.log_semiring import cumulative_logsumexp
from jaxns.internals.types import FloatArray, IntArray, PRNGKey

__all__ = ['random_ortho_matrix',
           'resample_indicies']


def random_ortho_matrix(key, n, special_orthogonal: bool = False):
    """
    Samples a random orthonormal n by n matrix from Stiefels manifold.
    From https://stackoverflow.com/a/38430739

    Args:
        key: PRNG seed
        n: Size of matrix, draws from O(num_options) group.

    Returns: random [num_options,num_options] matrix with determinant = +-1
    """
    H = random.normal(key, shape=(n, n))
    Q, R = jnp.linalg.qr(H)
    if special_orthogonal:
        R *= jnp.sign(R)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q


def resample_indicies(key: PRNGKey, log_weights: Optional[FloatArray] = None, S: Optional[int] = None,
                      replace: bool = False, num_total: Optional[int] = None) -> IntArray:
    """
    Get resample indicies according to a given weighting, with or without replacement.

    Args:
        key: PRNGKey
        log_weights: Optional log weights
        S: Optional number of samples. Computes effective sample size from log weights if not given.
        replace: whether to use replacement or not.
        num_total: Optional total sample size to use, must be given if `replace=False` and `log_weights=None`

    Returns:
        index array given the take indicies to resample at.
    """
    if S is None:
        if log_weights is None:
            raise ValueError("Need log_weights if S is not given.")
        # ESS = (sum w)^2 / sum w^2
        S = int(jnp.exp(2. * special.logsumexp(log_weights) - special.logsumexp(2. * log_weights)))

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
