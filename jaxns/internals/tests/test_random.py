import numpy as np
from jax import random, numpy as jnp

import jaxns.internals.maps
from jaxns.internals.random import random_ortho_matrix, resample_indicies


def test_random_ortho_matrix():
    M = random_ortho_matrix(random.PRNGKey(42), 5)
    assert jnp.isclose(jnp.linalg.det(M), 1.)
    assert jnp.allclose(M.T @ M, M @ M.T, atol=1e-6)
    assert jnp.allclose(M.T @ M, jnp.eye(5), atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(M, axis=0), jnp.linalg.norm(M, axis=1))

    for i in range(100):
        M = random_ortho_matrix(random.PRNGKey(i), 5)
        np.testing.assert_allclose(jnp.abs(jnp.linalg.det(M)), 1, atol=1e-6)

    for i in range(100):
        M = random_ortho_matrix(random.PRNGKey(i), 5, special_orthogonal=True)
        np.testing.assert_allclose(jnp.linalg.det(M), 1, atol=1e-6)


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        assert jnp.all(jnp.isclose(H @ H.T, jnp.eye(3), atol=1e-6))


def test_resample_indicies():
    n = 100
    sample_key = random.PRNGKey(42)
    log_weights = jnp.zeros(n)
    indices = resample_indicies(key=sample_key,
                                log_weights=log_weights,
                                S=n,
                                replace=False)
    assert np.unique(indices).size == n
