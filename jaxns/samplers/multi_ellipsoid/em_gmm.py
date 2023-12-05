from typing import Union

import jax.numpy as jnp
from jax import random, vmap, lax
from jax._src.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal


def initialize_params(key, data, n_components: int):
    n, d = data.shape

    # Initialize means by selecting random data points
    assign_idx = random.choice(key, n, shape=(n_components,), replace=False)
    means = data[assign_idx]

    # Initialize covariances as the empirical covariance of the data
    # cov = jnp.cov(data, rowvar=False)
    cov = jnp.diag(jnp.var(data, axis=0))
    covariances = jnp.repeat(cov[None, ...], n_components, axis=0)

    # Initialize mixture weights uniformly
    log_weights = jnp.full((n_components,), -jnp.log(n_components))

    return means, covariances, log_weights


def e_step(data, means, covariances, log_weights, mask):
    n, d = data.shape
    n_components = means.shape[0]

    # Compute the probabilities of each data point belonging to each Gaussian
    logpdf = vmap(lambda m, c: multivariate_normal.logpdf(data, m, c))(means, covariances)  # num_clusters, num_data
    if mask is not None:
        logpdf = jnp.where(mask[None, :], logpdf, -jnp.inf)
    logpdf_weighted = logpdf + log_weights[:, None]
    # Normalize probabilities
    log_responsibilities = logpdf_weighted - logsumexp(logpdf_weighted, axis=0)
    return log_responsibilities


def m_step(data, log_responsibilities):
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
    return means, covariances, log_weights


# No invariance under jit...
def em_gmm(key, data, n_components, mask: Union[jnp.ndarray, None] = None, n_iters=10, tol=1e-6):
    means, covariances, log_weights = initialize_params(key, data, n_components)
    params = (means, covariances, log_weights)

    def body(state):
        _, i, params = state
        log_responsibilities = e_step(data, *params, mask=mask)
        new_params = m_step(data, log_responsibilities)
        done = False
        for param, new_param in zip(params, new_params):
            done = done | (jnp.all(jnp.abs(jnp.array(param) - jnp.array(new_param)) < tol)) | (i >= n_iters)

        return done, i + 1, new_params

    def cond(state):
        done, _, params = state
        return jnp.bitwise_not(done)

    _, total_iters, params = lax.while_loop(
        cond,
        body,
        (jnp.asarray(False), jnp.asarray(0), params)
    )

    cluster_id = jnp.argmax(e_step(data, *params, mask=mask), axis=0)
    return cluster_id, params, total_iters
