"""Scientific and continuation checks for the local whitening experiment."""

import dataclasses
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.test_constrained_sampler import QuadraticModel, _request
from jaxns.algorithm.depth import CoreWorkBatch, _refresh_covariance_for_work
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.pytree import TreeField
from jaxns.samples import PhantomSamples, Samples
from jaxns.sampling.batching import sample_complete_chains
from jaxns.sampling.covariance import (
    empty_covariance_directions,
    refresh_covariance_directions,
)
from jaxns.sampling.slice import _sample_direction


def _samples(points, likelihoods):
    size = len(likelihoods)
    return Samples(
        U_samples=jnp.asarray(points),
        log_likelihoods=jnp.asarray(likelihoods),
        log_L_constraints=jnp.full(size, -jnp.inf),
        out_degree=jnp.zeros(size, dtype=jnp.int64),
        num_likelihood_evaluations=jnp.ones(size, dtype=jnp.int64),
        phantom_samples=PhantomSamples(
            U_samples=None,
            log_L=jnp.empty((size, 0)),
            valid_mask=jnp.zeros(size, dtype=bool),
        ),
    )


def test_covariance_uses_strict_contour_and_excludes_padding():
    points = np.array([
        [0.2, 0.3], [0.3, 0.7], [0.4, 0.4],
        [0.5, 0.8], [0.9, 0.1], [np.nan, np.nan],
    ])
    samples = _samples(points, [1.0, 2.0, 3.0, 4.0, 0.0, 10.0])
    data = empty_covariance_directions(2, 1e-4, 1.0)
    fitted = jax.jit(refresh_covariance_directions)(
        data, samples, jnp.asarray(5), jnp.asarray(1.0), jnp.asarray(-2.0),
    )
    covariance = np.cov(points[1:4], rowvar=False)
    covariance += 1e-4 * np.trace(covariance) / 2 * np.eye(2)
    np.testing.assert_allclose(
        fitted.cholesky @ fitted.cholesky.T, covariance, rtol=1e-6,
    )
    assert int(fitted.num_points) == 3
    assert int(fitted.num_updates) == 1


def test_covariance_refreshes_after_one_efold_in_either_direction():
    samples = _samples([[0.2, 0.3], [0.4, 0.5], [0.8, 0.7]], [1.0, 2.0, 3.0])
    refresh = jax.jit(refresh_covariance_directions)
    first = refresh(
        empty_covariance_directions(2, 1e-6, 1.0),
        samples, jnp.asarray(3), jnp.asarray(0.0), jnp.asarray(-2.0),
    )
    for log_X, updates in [(-2.99, 1), (-3.0, 2), (-1.0, 2)]:
        fitted = refresh(
            first, samples, jnp.asarray(3), jnp.asarray(0.0), jnp.asarray(log_X),
        )
        assert int(fitted.num_updates) == updates
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(restored.cholesky, first.cholesky)
    assert float(restored.log_X) == -2.0


@pytest.mark.parametrize("collapsed", [False, True])
def test_isotropic_core_keeps_singular_covariance_positive(collapsed):
    points = np.full((4, 2), 0.5)
    if not collapsed:
        points[:, 0] = [0.2, 0.4, 0.6, 0.8]
    data = refresh_covariance_directions(
        empty_covariance_directions(2, 1e-8, 1.0),
        _samples(points, [1.0, 2.0, 3.0, 4.0]),
        jnp.asarray(4), jnp.asarray(0.0), jnp.asarray(-2.0),
    )
    covariance = np.asarray(data.cholesky @ data.cholesky.T)
    assert np.all(np.isfinite(covariance))
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_cholesky_directions_use_covariance_not_its_inverse():
    key = jax.random.PRNGKey(312)
    point = TreeField({"a": jnp.ones(2), "b": jnp.ones(1)})
    cholesky = jnp.asarray([[0.1, 0, 0], [0.05, 0.5, 0], [0, 0.1, 0.8]])
    normal = point.random_normal_like(key)
    flat = jnp.concatenate([normal.tree["a"], normal.tree["b"]])
    expected = cholesky @ flat
    expected /= jnp.linalg.norm(expected)
    actual = _sample_direction(key, point, cholesky=cholesky)
    np.testing.assert_allclose(
        jnp.concatenate([actual.tree["a"], actual.tree["b"]]), expected,
        rtol=1e-6,
    )


def test_covariance_directions_preserve_model_unit_dtype():
    point = TreeField(jnp.ones(3, dtype=jnp.float32))
    actual = _sample_direction(
        jax.random.PRNGKey(313),
        point,
        cholesky=jnp.eye(3, dtype=jnp.float64),
    )
    assert actual.tree.dtype == jnp.float32


def test_covariance_scalar_and_continued_chains_agree():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(num_slices=32, num_phantom_samples=3)
    sampler = sampler._with_direction_cholesky(
        jnp.asarray([[0.08, 0.0], [0.04, 0.3]]),
    )
    request = _request(width=8)
    reference = jax.jit(
        lambda value: sample_complete_chains(sampler, value, model=model),
    )(request)
    continued = jax.jit(lambda value: sampler.get_samples(value, model=model))(request)
    for expected, actual in zip(
        jax.tree.leaves(reference), jax.tree.leaves(continued), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        reference.num_likelihood_evaluations, continued.num_likelihood_evaluations,
    )


def test_covariance_scalar_sampler_retains_frozen_geometry():
    sampler = UniDimSliceSampler(num_slices=4)
    cholesky = jnp.asarray([[0.08, 0.0], [0.04, 0.3]])
    fitted = sampler._with_direction_cholesky(cholesky)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_array_equal(restored._direction_cholesky, cholesky)
    assert sampler._direction_cholesky is None
    with pytest.raises(ValueError, match="non-periodic"):
        dataclasses.replace(sampler, _periodic=(True, False))._with_direction_cholesky(
            cholesky,
        )


def test_refresh_uses_classic_volume_and_state_retains_geometry():
    state = make_state(
        root_out_degree=4,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0),
        out_degree=(0, 0, 0, 0),
    ).covariance_directions()
    work = CoreWorkBatch(
        valid=jnp.asarray([True]),
        parent_idx=jnp.asarray([1]),
        log_L_constraint=jnp.asarray([2.0]),
        seed_idx=jnp.asarray([2]),
    )
    fitted = jax.jit(_refresh_covariance_for_work)(state, work)
    # Two singleton races have mean volume (4/5)*(3/4), independent of
    # physical padding and of the number of points supplied to the fit.
    np.testing.assert_allclose(fitted.covariance_data.log_X, np.log(3 / 5))
    assert int(fitted.covariance_data.num_points) == 2
    restored = pickle.loads(pickle.dumps(fitted.resize(16)))
    for expected, actual in zip(
        jax.tree.leaves(fitted.covariance_data),
        jax.tree.leaves(restored.covariance_data),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert restored.iso_directions().covariance_data is None
    with pytest.raises(ValueError, match="before fitting a GMM"):
        restored.fit_gmm_directions()
    with pytest.raises(ValueError, match="before merging"):
        restored.merge(restored)


@pytest.mark.parametrize("coefficient", [0.0, -1.0, np.nan, np.inf])
def test_covariance_configuration_rejects_invalid_scales(coefficient):
    state = make_state(
        root_out_degree=2, log_likelihoods=(1.0, 2.0), out_degree=(0, 0),
    )
    with pytest.raises(ValueError, match="regularisation"):
        state.covariance_directions(relative_regularisation=coefficient)
    with pytest.raises(ValueError, match="refresh_efolds"):
        state.covariance_directions(refresh_efolds=coefficient)
