"""Independent checks of the issue-312 analytic direction experiment."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.hessian_sampler import (
    ToyHessianSliceSampler,
    draw_direction,
    local_geometry,
    log_direction_density,
    negative_hessian_block,
)


def toy_log_likelihood(point, curvature=0.3):
    residual = point[0] - 0.5 - curvature * jnp.sin(2 * jnp.pi * point[1])
    return -0.5 * (
        (residual / 1e-5) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2)
    )


@pytest.mark.parametrize("coordinate,residual", [(0.12, 2e-5), (0.7, -3e-5)])
def test_ridge_hessian_matches_autodiff(coordinate, residual):
    point = jnp.full((18,), 0.5).at[1].set(coordinate)
    point = point.at[0].set(0.5 + 0.3 * jnp.sin(2 * jnp.pi * coordinate) + residual)
    reference = -jax.hessian(toy_log_likelihood)(point)
    np.testing.assert_allclose(
        negative_hessian_block(point, 1e-5, 0.3), reference[:2, :2],
        rtol=1e-13, atol=2e-5,
    )
    eigenvalues, rotation = np.linalg.eigh(np.asarray(reference))
    expected = (rotation * np.maximum(eigenvalues, 12.0)) @ rotation.T
    geometry = local_geometry(point, 1e-5, 0.3, 12.0)
    basis = np.eye(18)
    basis[:2, :2] = geometry.rotation
    actual = (basis * geometry.precision) @ basis.T
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-5)


def test_ridge_angular_density_matches_dense_reference():
    point = jnp.full((18,), 0.5).at[1].set(0.1)
    point = point.at[0].set(0.5 + 0.3 * jnp.sin(0.2 * jnp.pi) + 2e-5)
    geometry = local_geometry(point, 1e-5, 0.3, 12.0)
    direction = draw_direction(jax.random.PRNGKey(12), geometry)
    basis = np.eye(18)
    basis[:2, :2] = geometry.rotation
    precision = (basis * geometry.precision) @ basis.T
    expected = (
        0.5 * np.linalg.slogdet(precision)[1]
        - 9 * np.log(direction @ precision @ direction)
    )
    np.testing.assert_allclose(log_direction_density(direction, geometry), expected)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    np.testing.assert_equal(
        log_direction_density(direction, geometry),
        log_direction_density(-direction, geometry),
    )


@pytest.mark.parametrize("gap", [15.0, 40.0])
def test_ridge_scaled_metric_matches_straight_contour_covariance(gap):
    # A uniform 15-ball of radius sqrt(2*gap) has coordinate variance
    # 2*gap/17. Its three independent unit intervals have variance 1/12.
    point = jnp.full((18,), 0.5)
    geometry = local_geometry(point, 1e-5, 0.0, 12.0, 2 * gap / 17)
    basis = np.eye(18)
    basis[:2, :2] = geometry.rotation
    covariance = (basis / geometry.precision) @ basis.T
    expected = np.full(18, 1 / 12)
    expected[0] = 1e-10 * 2 * gap / 17
    expected[2:16] = 0.05 ** 2 * 2 * gap / 17
    np.testing.assert_allclose(covariance, np.diag(expected), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("curvature", [0.0, 0.3])
def test_ridge_mh_counts_rejections_as_transitions(curvature):
    point = jnp.full((18,), 0.5).at[1].set(0.125)
    point = point.at[0].set(0.5 + curvature * jnp.sin(0.25 * jnp.pi) - 1e-5)

    def likelihood(value):
        return toy_log_likelihood(value, curvature)
    sampler = ToyHessianSliceSampler(num_slices=90, curvature=curvature)
    final, (points, values) = jax.jit(sampler.run_chain, static_argnums=4)(
        jax.random.PRNGKey(17), point, likelihood(point), jnp.asarray(-15.0),
        likelihood,
    )
    chain = np.concatenate([point[None], points], axis=0)
    moved = np.any(np.diff(chain, axis=0) != 0, axis=1)
    assert len(points) == 90
    assert int(final.accepted) == np.count_nonzero(moved)
    assert np.all(np.asarray(values) > -15.0)
    assert np.all((np.asarray(points) >= 0) & (np.asarray(points) <= 1))
    if curvature == 0:
        assert int(final.accepted) == 90
    else:
        assert 0 < int(final.accepted) < 90
