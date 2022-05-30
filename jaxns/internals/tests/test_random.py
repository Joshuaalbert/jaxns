from jax import random, numpy as jnp

from jaxns.internals.random import random_ortho_matrix


def test_random_ortho_matrix():
    M = random_ortho_matrix(random.PRNGKey(42), 5)
    assert jnp.isclose(jnp.linalg.det(M), 1.)
    assert jnp.allclose(M.T @ M, M @ M.T, atol=1e-6)
    assert jnp.allclose(M.T @ M, jnp.eye(5), atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(M, axis=0), jnp.linalg.norm(M, axis=1))


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-6))
