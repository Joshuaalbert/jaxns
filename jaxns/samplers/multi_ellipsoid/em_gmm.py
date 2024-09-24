from typing import Union

import jax
import jax.numpy as jnp
from jax import random, vmap, lax
from jax._src.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal

from jaxns.internals.mixed_precision import mp_policy


def initialize_params(key, data, n_components: int):
    """
    Initialize the parameters of a Gaussian Mixture Model.

    Args:
        key: the random key
        data: [n, d] array of data
        n_components: number of components

    Returns:
        means: [num_clusters, d] array of means
    """
    n, d = data.shape

    # Initialize means by selecting random data points
    assign_idx = random.choice(key, n, shape=(n_components,), replace=False)
    means = data[assign_idx]

    # Initialize covariances as the empirical covariance of the data
    # cov = jnp.cov(data, rowvar=False)
    cov = jnp.diag(jnp.var(data, axis=0))
    covariances = jnp.repeat(cov[None, ...], n_components, axis=0)

    # Initialize mixture weights uniformly
    log_weights = jnp.full((n_components,), -jnp.log(n_components), mp_policy.measure_dtype)

    return means, covariances, log_weights


def e_step(data, means, covariances, log_weights, mask):
    """
    Compute the responsibilities of each Gaussian for each data point.

    Args:
        data: [n, d] array of data
        means: [num_clusters, d] array of means
        covariances: [num_clusters, d, d] array of covariances
        log_weights: [num_clusters] array of log weights
        mask: [n] boolean array indicating which data points to use

    Returns:
        log_responsibilities: [num_clusters, n] array of log responsibilities
    """
    n, d = data.shape
    n_components = means.shape[0]

    # Compute the probabilities of each data point belonging to each Gaussian
    logpdf = vmap(lambda m, c: multivariate_normal.logpdf(data, m, c))(means, covariances)  # num_clusters, num_data
    if mask is not None:
        logpdf = jnp.where(mask[None, :], logpdf, mp_policy.cast_to_measure(-jnp.inf))
    logpdf_weighted = logpdf + log_weights[:, None]
    # Normalize probabilities
    log_responsibilities = logpdf_weighted - logsumexp(logpdf_weighted, axis=0)
    return log_responsibilities


def m_step(data, log_responsibilities):
    """
    Update the parameters of the Gaussian Mixture Model.

    Args:
        data: [n, d] array of data
        log_responsibilities: [num_clusters, n] array of log responsibilities

    Returns:
        means: [num_clusters, d] array of means
    """
    n_components, num_data = log_responsibilities.shape
    _, d = data.shape

    # Update means, covariances, and weights
    log_weights = logsumexp(log_responsibilities, axis=1) - jnp.log(num_data)  # num_components

    # num_components, num_data X num_data, D -> num_components, D
    weighted_responsibilities = jnp.exp(log_responsibilities - log_weights[:, None] - jnp.log(num_data))
    means = jnp.matmul(weighted_responsibilities, data)

    centered_data = data[None, :, :] - means[:, None, :]  # num_components, num_data, D

    covariances = jnp.einsum("cn,cnd,cne->cde", weighted_responsibilities, centered_data, centered_data)
    covariances = covariances + 1e-4 * jnp.eye(d)
    return (
        mp_policy.cast_to_measure(means), mp_policy.cast_to_measure(covariances), mp_policy.cast_to_measure(log_weights)
    )


# No invariance under jit...
def em_gmm(key, data, n_components, mask: Union[jax.Array, None] = None, n_iters=10, tol=1e-6):
    """
    Fit a Gaussian Mixture Model to the data using the Expectation-Maximization algorithm.

    Args:
        key: the random key
        data: [n, d] array of data
        n_components: number of components
        mask: [n] boolean array indicating which data points to use
        n_iters: maximum number of iterations
        tol: convergence tolerance

    Returns:
        cluster_id: [n] array of cluster assignments
        params: tuple of (means, covariances, log_weights)
        total_iters: total number of iterations use
    """
    means, covariances, log_weights = initialize_params(key, data, n_components)
    params = (means, covariances, log_weights)

    def body(state):
        _, i, params = state
        log_responsibilities = e_step(data, *params, mask=mask)
        new_params = m_step(data, log_responsibilities)
        done = False
        for param, new_param in zip(params, new_params):
            done = done | (jnp.all(jnp.abs(jnp.asarray(param) - jnp.asarray(new_param)) < tol)) | (i >= n_iters)

        return done, i + 1, new_params

    def cond(state):
        done, _, params = state
        return jnp.bitwise_not(done)

    _, total_iters, params = lax.while_loop(
        cond,
        body,
        (jnp.asarray(False, jnp.bool_), jnp.asarray(0, jnp.int32), params)
    )

    cluster_id = jnp.argmax(e_step(data, *params, mask=mask), axis=0)
    return cluster_id, params, total_iters
