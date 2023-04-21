from functools import partial

import jax.numpy as jnp
from jax import random, vmap, jit
from jax._src.lax.control_flow import while_loop
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


def e_step(data, means, covariances, log_weights):
    n, d = data.shape
    n_components = means.shape[0]

    # Compute the probabilities of each data point belonging to each Gaussian
    logpdf = vmap(lambda m, c: multivariate_normal.logpdf(data, m, c))(means, covariances)  # num_clusters, num_data
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
@partial(jit, static_argnames=['n_components', 'n_iters', 'tol'])
def em_gmm(key, data, n_components, n_iters=10, tol=1e-6):
    means, covariances, log_weights = initialize_params(key, data, n_components)
    params = (means, covariances, log_weights)

    def body(state):
        _, i, params = state
        log_responsibilities = e_step(data, *params)
        new_params = m_step(data, log_responsibilities)
        done = False
        for param, new_param in zip(params, new_params):
            done = done | (jnp.all(jnp.abs(jnp.array(param) - jnp.array(new_param)) < tol)) | (i >= n_iters)

        return done, i + 1, new_params

    def cond(state):
        done, _, params = state
        return jnp.bitwise_not(done)

    _, total_iters, params = while_loop(
        cond,
        body,
        (jnp.asarray(False), jnp.asarray(0), params)
    )

    cluster_id = jnp.argmax(e_step(data, *params), axis=0)
    return cluster_id, params, total_iters


# def test_em_gmm():
#     data = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [4.0, 4.0], [4.1, 4.1], [4.2, 4.2]])
#     n_components = 2
#     key = random.PRNGKey(42)
#
#     cluster_id, (means, covariances, log_weights), _ = em_gmm(key, data, n_components)
#     assert jnp.all(cluster_id == jnp.asarray([1, 1, 1, 0, 0, 0]))

def recursive_gmm(key, data, n_components, n_iters=10, tol=1e-6):
    pass


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    key = random.PRNGKey(42)

    d = 100
    n_components = 10
    n_data = 10000

    X, y_true = make_blobs(n_samples=n_data, centers=n_components, n_features=d, cluster_std=1, random_state=42)
    import pylab as plt

    cluster_id, (means, covariances, log_weights), total_iters = em_gmm(key, X, n_components=n_components, n_iters=5)
    print("Used", total_iters)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_id, cmap='jet')
    plt.show()

    accuracy = adjusted_rand_score(y_true, cluster_id)
    print(accuracy)
