"""Independent numerical and scheduling checks for the offline GP experiment."""

from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfpd

from benchmarks.issue_312.gp_geometry import (
    GP,
    fit_gp,
    negative_predictive_hessian,
    predictive_mean,
    representative_indices,
)
from benchmarks.issue_312.gp_hessian_sampler import GPHessianSliceSampler
from benchmarks.issue_312.gp_refresh import run_with_refresh
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.priors import Prior


def test_gp_predictive_hessian_matches_autodiff_with_affine_inputs():
    rng = np.random.default_rng(31203)
    transform = np.array([[2., 0.3, -0.4], [0.2, 0.7, 0.1], [0., -0.2, 1.1]])
    gp = GP(*map(jnp.asarray, (
        rng.normal(size=3), np.eye(3), transform,
        rng.normal(size=(11, 3)), rng.normal(size=11), 7.4,
        np.eye(11), np.zeros(4),
    )))
    for point in rng.normal(size=(5, 3)):
        expected = -jax.hessian(partial(predictive_mean, gp))(jnp.asarray(point))
        actual = negative_predictive_hessian(gp, jnp.asarray(point))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gp_fit_recovers_quadratic_curvature_and_padding():
    rng = np.random.default_rng(31204)
    points = rng.uniform(-1, 1, (48, 2))
    precision = np.array([[10., 2.], [2., 1.]])
    values = 100.0 - 0.5 * np.einsum("ni,ij,nj->n", points, precision, points)
    gp, record, selected = fit_gp(points, values, max_points=64, max_iterations=100)
    assert len(np.unique(selected)) == 48
    assert record["training_points"] == 48
    assert record["held_out_rmse"] is None
    assert record["train_rmse"] < 1e-3
    assert np.all(np.asarray(gp.coefficients)[48:] == 0)
    actual = negative_predictive_hessian(gp, jnp.array([0.2, -0.1]))
    np.testing.assert_allclose(actual, precision, rtol=0.03, atol=0.03)
    np.testing.assert_allclose(gp.input_mean, points.mean(axis=0))
    covariance = np.cov(points.T)
    covariance += 1e-8 * np.trace(covariance) / 2 * np.eye(2)
    np.testing.assert_allclose(gp.input_cholesky @ gp.input_cholesky.T, covariance)
    _, refit, _ = fit_gp(points, values, max_points=64, previous=gp)
    assert len(refit["restart_objectives"]) == 2
    assert refit["negative_log_marginal_likelihood"] == min(refit["restart_objectives"])


def test_gp_representatives_remain_distinct_for_duplicate_points():
    points = np.zeros((20, 3))
    selected = representative_indices(points, 12)
    assert len(np.unique(selected)) == 12


def _refresh_replay_model():
    point = Prior(tfpd.Uniform(jnp.zeros(2), jnp.ones(2)), name="x").realise()
    return -jnp.sum(((point - 0.5) / 0.1) ** 2) / 2


def test_gp_batch_hook_replays_frozen_sampler_exactly():
    sampler = UniDimSliceSampler(num_slices=10)
    runner = NestedSampler(
        model=Model(prior_model=_refresh_replay_model), sampler=sampler,
        root_allocation_degree=20, replacement_width=10,
    )
    initial = runner.initialise(jax.random.PRNGKey(31205))
    reference = runner.resume_until_goal(initial, lambda s: int(s.goal_loop_iter) > 0)
    refreshes = [0.0]

    def refresh(state, contour, log_volume):
        assert np.isfinite(contour)
        assert int(state.num_samples) > 20
        assert abs(log_volume - refreshes[-1]) >= 1.0
        refreshes.append(log_volume)
        return sampler

    actual = run_with_refresh(runner, initial, sampler, refresh)
    assert len(refreshes) > 2
    assert jax.tree.structure(actual) == jax.tree.structure(reference)
    for observed, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
        np.testing.assert_array_equal(observed, expected)


def test_gp_mh_retains_rejected_states_and_uniform_target():
    # The true contour is the whole cube while the artificial GP produces a
    # strongly position-dependent direction law. It must not bias that cube.
    rng = np.random.default_rng(31206)
    points = rng.uniform(size=(64, 2))
    values = -80 * (points[:, 0] - points[:, 1] ** 2) ** 2
    gp, _, _ = fit_gp(points, values, max_points=64)
    sampler = GPHessianSliceSampler(
        gp, log_likelihood_peak=1.0, quadratic_dimension=2, num_slices=20,
    )
    seeds = jnp.asarray(rng.uniform(size=(2048, 2)))

    def chain(key, point):
        return sampler.run_chain(
            key, point, jnp.asarray(0.), jnp.asarray(-1.), lambda x: jnp.asarray(0.),
        )

    final, (states, _) = jax.jit(jax.vmap(chain))(
        jax.random.split(jax.random.PRNGKey(31207), len(seeds)), seeds,
    )
    differences = np.diff(np.concatenate([seeds[:, None], states], axis=1), axis=1)
    moved = np.any(differences != 0, axis=2)
    np.testing.assert_array_equal(moved.sum(axis=1), final.accepted)
    assert 0 < int(jnp.sum(final.accepted)) < 20 * len(seeds)
    assert np.all((np.asarray(states) >= 0) & (np.asarray(states) <= 1))
    # Predetermined loose DKW bound; this probes stationarity, not convergence.
    from scipy.stats import kstest

    assert max(kstest(axis, "uniform").statistic for axis in np.asarray(final.point).T) < 0.06
