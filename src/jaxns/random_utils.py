
import jax
from jax import numpy as jnp
from jax import random
from jax.scipy import special

from jaxns.log_semiring import cumulative_logsumexp
from jaxns.mixed_precision import mp_policy
from jaxns.types import FloatArray, IntArray, PRNGKey

__all__ = [
    'random_ortho_matrix',
    'resample',
    'resample_indicies',
]


def random_ortho_matrix(key, n, special_orthogonal: bool = False):
    """
    Sample a uniformly distributed orthogonal matrix.

    Args:
        key: PRNG seed
        n: Matrix dimension.
        special_orthogonal: Restrict the draw to determinant +1.

    Returns:
        An [n, n] orthogonal matrix, with determinant +1 when requested.
    """
    H = random.normal(key, shape=(n, n), dtype=mp_policy.measure_dtype)
    Q, R = jnp.linalg.qr(H)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    if special_orthogonal:
        # Reflect one column only for negative orientation. This maps both
        # halves of O(n) uniformly onto SO(n), including even dimensions.
        Q = Q.at[:, -1].multiply(jnp.sign(jnp.linalg.det(Q)))
    return Q


def resample_indicies(key: PRNGKey, log_weights: FloatArray | None = None, S: int | None = None,
                      replace: bool = True, num_total: int | None = None) -> IntArray:
    """
    Get resample indicies according to a given weighting, with or without replacement.

    Args:
        key: PRNGKey
        log_weights: Optional log weights
        S: Optional number of samples. Computes effective sample size from log weights if not given.
        replace: whether to use replacement or not.
        num_total: Total population size, required when log_weights is absent.

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
            # The inclusive CDF starts with one unit of mass for index zero.
            log_p_cuml = jnp.log(jnp.arange(1, num_total + 1))
            log_r = log_p_cuml[-1] + jnp.log(1. - random.uniform(key, (S,)))
            idx = jnp.searchsorted(log_p_cuml, log_r)
    else:
        if log_weights is not None:
            g = -random.gumbel(key, shape=log_weights.shape) - log_weights
        else:
            if num_total is None:
                raise ValueError("Need num_total if log_weights is None.")
            g = -random.gumbel(key, shape=(num_total,))
        # idx = jnp.argsort(g)[:S]
        _, idx = jax.lax.top_k(-g, k=S)
    return idx


def resample(
        key: PRNGKey,
        samples,
        log_weights: FloatArray,
        S: int | None = None,
        replace: bool = True,
):
    """Resample aligned pytree leaves according to log weights."""
    if S is None:
        S = int(jnp.size(log_weights))
    indices = resample_indicies(
        key=key,
        log_weights=log_weights,
        S=S,
        replace=replace,
    )
    return jax.tree.map(lambda sample: sample[indices, ...], samples)
