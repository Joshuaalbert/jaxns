import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from jaxns.random_utils import random_ortho_matrix, resample_indicies


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_random_ortho_matrix(dimension):
    draw = jax.jit(
        random_ortho_matrix,
        static_argnames=("n", "special_orthogonal"),
    )
    M = draw(random.PRNGKey(42), dimension, special_orthogonal=True)
    np.testing.assert_allclose(jnp.linalg.det(M), 1.)
    np.testing.assert_allclose(M.T @ M, M @ M.T, atol=1e-6)
    np.testing.assert_allclose(M.T @ M, jnp.eye(dimension), atol=1e-6)
    np.testing.assert_allclose(
        jnp.linalg.norm(M, axis=0), jnp.linalg.norm(M, axis=1),
    )

    for i in range(100):
        M = draw(random.PRNGKey(i), dimension)
        np.testing.assert_allclose(jnp.abs(jnp.linalg.det(M)), 1, atol=1e-6)

    for i in range(100):
        M = draw(
            random.PRNGKey(i), dimension, special_orthogonal=True,
        )
        np.testing.assert_allclose(jnp.linalg.det(M), 1, atol=1e-6)


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        assert jnp.all(jnp.isclose(H @ H.T, jnp.eye(3), atol=1e-6))


def test_resample_indicies():
    n = 100
    sample_key = random.PRNGKey(42)
    log_weights = jnp.zeros(n)
    indices = resample_indicies(
        key=sample_key,
        log_weights=log_weights,
        S=n,
        replace=False,
    )
    assert np.unique(indices).size == n


@pytest.mark.parametrize("num_total", [1, 2, 3, 7, 11])
def test_resample_indices_uniform_with_replacement(num_total):
    num_draws = 20_000
    draw = jax.jit(
        resample_indicies,
        static_argnames=("S", "replace", "num_total"),
    )
    indices = np.asarray(draw(
        random.PRNGKey(42),
        S=num_draws,
        replace=True,
        num_total=num_total,
    ))
    assert np.all((0 <= indices) & (indices < num_total))
    counts = np.bincount(indices, minlength=num_total)
    expected_count = num_draws / num_total
    # Every index, including zero, must carry the same probability mass.
    count_std = np.sqrt(
        num_draws * (1.0 / num_total) * (1.0 - 1.0 / num_total),
    )
    np.testing.assert_allclose(
        counts, expected_count, rtol=0.0, atol=6 * count_std,
    )
