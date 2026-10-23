"""Independent references for the incremental residual-network experiment."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.nn_geometry import (
    NN,
    fit_nn,
    negative_predictive_hessian,
    predictive_mean,
    rebase,
)
from benchmarks.issue_312.nn_hessian_sampler import NNHessianSliceSampler


def test_nn_representation_linear_quadratic_reference():
    transform = np.array([[2., 0.], [0.4, 3.]])
    linear = np.array([[1.3, 0.2], [-0.5, 2.1]])
    network = NN(*map(jnp.asarray, (
        [0.4, 0.2], np.linalg.inv(transform), transform, 4.0, 7.0,
        linear, [0.1, -0.3], np.zeros((2, 0)), np.zeros(0), np.zeros((0, 2)),
    )))
    point = jnp.array([0.6, 0.8])
    features = ((point - network.input_mean) @ transform.T) @ linear + network.bias
    np.testing.assert_allclose(predictive_mean(network, point), 7 - 2 * features @ features)
    expected = 4 * transform.T @ linear @ linear.T @ transform
    np.testing.assert_allclose(negative_predictive_hessian(network, point), expected)


def test_nn_rebase_preserves_values_and_curvature():
    rng = np.random.default_rng(78)
    network = NN(*map(jnp.asarray, (
        [0.4, 0.2], np.array([[2., 0.], [0.1, 0.3]]),
        np.linalg.inv(np.array([[2., 0.], [0.1, 0.3]])), 2.0, 4.0,
        rng.normal(size=(2, 2)), rng.normal(size=2), rng.normal(size=(2, 3)),
        rng.normal(size=3), rng.normal(size=(3, 2)),
    )))
    rebased = rebase(network, np.array([0.2, 0.7]), np.array([[0.2, 0.], [0.4, 1.1]]), 0.003)
    points = jnp.asarray(rng.normal(size=(30, 2)))
    for function in (predictive_mean, negative_predictive_hessian):
        evaluate = jax.vmap(function, in_axes=(None, 0))
        np.testing.assert_allclose(evaluate(rebased, points), evaluate(network, points),
                                   rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("relative_loss", [False, True])
def test_nn_fit_recovers_known_quadratic(relative_loss):
    rng = np.random.default_rng(90)
    points = rng.uniform(size=(128, 2))
    precision = np.array([[30., 12.], [12., 10.]])
    delta = points - [0.4, 0.55]
    values = 10 - 0.5 * np.einsum("ni,ij,nj->n", delta, precision, delta)
    network, record, indices = fit_nn(
        points, values, peak=10, hidden_features=0, max_points=64, max_iterations=500,
        relative_loss=relative_loss,
    )
    assert len(set(indices)) == 64
    assert record["train_rmse"] < 1e-5
    assert record["held_out_rmse"] < 1e-5
    np.testing.assert_allclose(negative_predictive_hessian(network, jnp.array([0.3, 0.6])),
                               precision, rtol=1e-5, atol=1e-5)


def test_nn_mh_retains_rejections_and_preserves_uniform_target():
    # The network invents nonconstant geometry on a truly uniform cube.
    # The endpoint direction-density correction must undo that preference.
    rng = np.random.default_rng(31216)
    points = rng.uniform(size=(64, 2))
    values = -80 * (points[:, 0] - points[:, 1] ** 2) ** 2
    network, _, _ = fit_nn(
        points, values, peak=1.0, max_points=64, hidden_features=8, max_iterations=300,
    )
    sampler = NNHessianSliceSampler(
        network, log_likelihood_peak=1.0, quadratic_dimension=2, num_slices=20,
    )
    seeds = jnp.asarray(rng.uniform(size=(2048, 2)))

    @jax.jit
    @jax.vmap
    def chain(key, point):
        return sampler.run_chain(
            key, point, jnp.asarray(0.), jnp.asarray(-1.), lambda x: jnp.asarray(0.),
        )

    final, (states, values) = chain(
        jax.random.split(jax.random.PRNGKey(31217), len(seeds)), seeds,
    )
    differences = np.diff(np.concatenate([seeds[:, None], states], axis=1), axis=1)
    moved = np.any(differences != 0, axis=2)
    np.testing.assert_array_equal(moved.sum(axis=1), final.accepted)
    assert 0 < int(jnp.sum(final.accepted)) < 20 * len(seeds)
    assert np.all((np.asarray(states) >= 0) & (np.asarray(states) <= 1))
    assert np.all(np.asarray(values) == 0)
    from scipy.stats import kstest

    assert max(kstest(axis, "uniform").statistic for axis in np.asarray(final.point).T) < 0.06
