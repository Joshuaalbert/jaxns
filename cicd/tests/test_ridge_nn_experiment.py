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
