"""Independent checks of responsibility-selected Gaussian directions."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from scipy.special import logsumexp
from scipy.stats import kstest, multivariate_normal

from benchmarks.issue_312.local_gmm_sampler import (
    LocalGMMSliceSampler,
    component_log_angular_density,
    factor_mixture,
    log_direction_density,
    log_responsibilities,
)
from jaxns.sampling.gmm import GaussianMixture


def test_local_gmm_density_matches_independent_reference():
    means = np.array([[0.2, 0.4], [0.8, 0.6]])
    covariance = np.array([[[0.01, 0.008], [0.008, 0.08]], [[0.1, -0.015], [-0.015, 0.01]]])
    masses = np.array([0.3, 0.7])
    geometry = factor_mixture(GaussianMixture(
        jnp.asarray(means), jnp.asarray(covariance), jnp.log(masses), jnp.ones(2, dtype=bool),
    ))
    point = np.array([0.45, 0.55])
    logits = np.array([multivariate_normal.logpdf(point, mu, cov)
                       for mu, cov in zip(means, covariance)]) + np.log(masses)
    expected_selection = logits - logsumexp(logits)
    selection = log_responsibilities(geometry, jnp.asarray(point))
    np.testing.assert_allclose(selection, expected_selection, rtol=1e-12, atol=1e-12)
    angles = np.linspace(0, 2 * np.pi, 16384, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    inverse = np.linalg.inv(covariance)
    reference = -0.5 * np.linalg.slogdet(covariance)[1][None, :] - np.log(
        np.einsum("ni,kij,nj->nk", directions, inverse, directions),
    )
    observed = jax.vmap(component_log_angular_density, in_axes=(None, 0))(
        geometry, jnp.asarray(directions),
    )
    np.testing.assert_allclose(observed, reference, rtol=1e-12, atol=1e-12)
    density = jax.vmap(lambda angular: log_direction_density(selection, angular, 0.01))(observed)
    expected = np.logaddexp(np.log(0.01), np.log(0.99) + logsumexp(
        reference + expected_selection[None, :], axis=1,
    ))
    np.testing.assert_allclose(density, expected, rtol=1e-12, atol=1e-12)
    # This quadrature also checks the omitted sphere-area convention and the
    # sign symmetry needed to reverse a slice along an unoriented line.
    np.testing.assert_allclose(np.exp(density).mean(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(density[:8192], density[8192:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("local_selection", [True, False])
def test_local_gmm_mh_preserves_uniform_target_and_self_transitions(local_selection):
    geometry = factor_mixture(GaussianMixture(
        centres=jnp.array([[0.25, 0.5], [0.75, 0.5]]),
        covariances=jnp.array([[[0.002, 0.], [0., 0.1]], [[0.1, 0.], [0., 0.002]]]),
        log_masses=jnp.log(jnp.array([0.5, 0.5])), valid=jnp.ones(2, dtype=bool),
    ))
    sampler = LocalGMMSliceSampler(geometry, num_slices=30, local_selection=local_selection)
    rng = np.random.default_rng(31225)
    seeds = jnp.asarray(rng.uniform(size=(4096, 2)))
    keys = jax.random.split(jax.random.PRNGKey(31226), len(seeds))

    def run(active):
        @jax.jit
        @jax.vmap
        def chain(key, point):
            return active.run_chain(
                key, point, jnp.asarray(0.), jnp.asarray(-1.), lambda x: jnp.asarray(0.),
            )
        return chain(keys, seeds)

    final, (states, values) = run(sampler)
    moved = np.any(np.diff(np.concatenate([seeds[:, None], states], axis=1), axis=1) != 0, axis=2)
    np.testing.assert_array_equal(moved.sum(axis=1), final.accepted)
    np.testing.assert_array_equal(np.sum(final.component_counts, axis=1), 30)
    np.testing.assert_array_equal(final.evaluations, 30)
    assert np.all(np.asarray(values) == 0)
    assert np.all((np.asarray(states) >= 0) & (np.asarray(states) <= 1))
    assert max(kstest(axis, "uniform").statistic for axis in np.asarray(final.point).T) < 0.06
    if local_selection:
        assert 0 < int(jnp.sum(final.accepted)) < 30 * len(seeds)
        # Deliberately omit the correction as a negative control. This should
        # reveal why using a nearby Gaussian is not stationary by itself.
        uncorrected, _ = run(replace(sampler, correct_mh=False))
        assert max(kstest(axis, "uniform").statistic
                   for axis in np.asarray(uncorrected.point).T) > 0.06
    else:
        np.testing.assert_array_equal(final.accepted, 30)


def test_local_gmm_rejects_invalid_geometry():
    mixture = GaussianMixture(
        jnp.zeros((1, 2)), jnp.array([[[1., 2.], [2., 1.]]]),
        jnp.zeros(1), jnp.ones(1, dtype=bool),
    )
    with pytest.raises(np.linalg.LinAlgError):
        factor_mixture(mixture)
    with pytest.raises(ValueError, match="finite component mass"):
        factor_mixture(replace(mixture, log_masses=jnp.array([jnp.nan])))
