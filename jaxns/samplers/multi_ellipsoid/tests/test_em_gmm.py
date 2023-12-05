import pylab as plt
from jax import numpy as jnp, random, jit
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from jaxns.samplers.multi_ellipsoid.em_gmm import em_gmm


def test_em_gmm():
    data = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [4.0, 4.0], [4.1, 4.1], [4.2, 4.2]])
    n_components = 2
    key = random.PRNGKey(42)

    cluster_id, (means, covariances, log_weights), _ = em_gmm(key, data, n_components)
    assert jnp.all(cluster_id == jnp.asarray([1, 1, 1, 0, 0, 0]))


def test_blob_decomp():
    key = random.PRNGKey(42)

    d = 2
    n_data = 500
    jit_em_gmm = jit(em_gmm, static_argnames=['n_components', 'n_iters', 'tol'])

    for n_components in [2, 3]:
        X, y_true = make_blobs(n_samples=n_data, centers=n_components, n_features=d, cluster_std=1, random_state=42)

        cluster_id, (means, covariances, log_weights), total_iters = jit_em_gmm(key, X, n_components=n_components,
                                                                                n_iters=100)
        print("Used", total_iters)
        plt.scatter(X[:, 0], X[:, 1], c=cluster_id, cmap='jet')
        # plt.show()
        plt.close('all')
        accuracy = adjusted_rand_score(y_true, cluster_id)
        print(accuracy)
        assert accuracy == 1.
