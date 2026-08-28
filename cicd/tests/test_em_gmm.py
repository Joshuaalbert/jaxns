import numpy as np
import pylab as plt
from jax import jit, random
from jax import numpy as jnp
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from jaxns.sampling.gmm import (
    em_gmm,
    em_gmm_reference,
    fit_gmm,
    initialise_gmm,
)


def test_em_gmm():
    data = jnp.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [4.0, 4.0], [4.1, 4.1], [4.2, 4.2]])
    n_components = 2
    key = random.PRNGKey(42)

    cluster_id, _, _ = em_gmm(key, data, n_components)
    assert jnp.all(cluster_id == jnp.asarray([1, 1, 1, 0, 0, 0]))


def test_blob_decomp():
    key = random.PRNGKey(42)

    d = 2
    n_data = 500
    jit_em_gmm = jit(em_gmm, static_argnames=['n_components', 'n_iters', 'tol'])

    for n_components in [2, 3]:
        X, y_true = make_blobs(n_samples=n_data, centers=n_components, n_features=d, cluster_std=1, random_state=42)

        cluster_id, _, total_iters = jit_em_gmm(
            key,
            X,
            n_components=n_components,
            n_iters=100,
        )
        print("Used", total_iters)
        plt.scatter(X[:, 0], X[:, 1], c=cluster_id, cmap='jet')
        # plt.show()
        plt.close('all')
        accuracy = adjusted_rand_score(y_true, cluster_id)
        print(accuracy)
        assert accuracy == 1.


def test_warm_weighted_fit_matches_numpy_reference():
    data = jnp.asarray([
        [-2.1, -1.8],
        [-1.9, -2.2],
        [-2.0, -1.9],
        [1.8, 2.0],
        [2.2, 1.9],
        [2.0, 2.1],
        [99.0, 99.0],
    ])
    mask = jnp.asarray([True, True, True, True, True, True, False])
    sample_weights = jnp.asarray([0.08, 0.12, 0.10, 0.20, 0.25, 0.25, 0.0])
    initial = initialise_gmm(
        random.PRNGKey(8),
        data,
        2,
        mask=mask,
        log_sample_weights=jnp.where(mask, jnp.log(sample_weights), -jnp.inf),
    )
    fitted, cluster_id, normalized_weights = fit_gmm(
        random.PRNGKey(9),
        data,
        2,
        mask=mask,
        log_sample_weights=jnp.where(mask, jnp.log(sample_weights), -jnp.inf),
        initial=initial,
        n_iters=4,
    )
    reference = em_gmm_reference(
        np.asarray(data),
        initial,
        np.asarray(sample_weights),
        np.asarray(mask),
        n_iters=4,
    )

    np.testing.assert_allclose(fitted.centres, reference.centres, rtol=2e-6)
    np.testing.assert_allclose(
        fitted.covariances,
        reference.covariances,
        rtol=2e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(fitted.log_masses, reference.log_masses, rtol=2e-6)
    assert jnp.all(jnp.isfinite(fitted.centres))
    assert jnp.all(jnp.isfinite(fitted.covariances))
    assert cluster_id.shape == (7,)
    assert normalized_weights[-1] == 0.0


def test_masked_non_finite_storage_cannot_poison_fit():
    data = jnp.asarray([
        [-1.0, -1.1],
        [-0.9, -1.0],
        [1.0, 0.9],
        [1.1, 1.0],
        [jnp.nan, jnp.inf],
    ])
    mask = jnp.asarray([True, True, True, True, False])
    log_weights = jnp.where(mask, -jnp.log(4.0), -jnp.inf)
    fitted, _, _ = fit_gmm(
        random.PRNGKey(10),
        data,
        2,
        mask=mask,
        log_sample_weights=log_weights,
        n_iters=3,
    )

    assert jnp.all(jnp.isfinite(fitted.centres))
    assert jnp.all(jnp.isfinite(fitted.covariances))
