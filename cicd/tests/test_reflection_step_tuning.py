"""Scientific gates for the frozen benchmark schedule, with fixed random streams."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.allocation_observer import run_with_observer
from benchmarks.issue_312.reflection_sampler import ReflectionSampler, reflection_trajectory
from benchmarks.issue_312.tuning_common import exact_draws, make_model, numpy_log_likelihood
from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
from jaxns.core import NestedSampler
from jaxns.pytree import pytree_ravel
from jaxns.samples import PhantomSamples


def test_frozen_step_lookup_uses_parent_contour_and_declared_tails():
    sampler = FrozenContourReflectionSampler(gaps=(1., 4., 49.), steps=(.001, .002, .004))
    contours = jnp.asarray([-jnp.inf, -100., -49., -4., -2., -1., 0., 1.])
    actual = jax.jit(jax.vmap(sampler.step_for_contour))(contours)
    np.testing.assert_allclose(
        actual, [.003, .003, .004, .002, np.sqrt(2)*.001, .001, .001, .001], rtol=1e-14,
    )


def test_constant_schedule_exact_rng_and_phantom_equivalence():
    model = make_model()
    point, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(118)))
    assert point.dtype == jnp.float64 and jax.default_backend() == 'cpu'

    def likelihood(u):
        return model.log_likelihood(unravel(u))

    points, _ = exact_draws(np.random.default_rng(229), 32, 15.)
    keys = jax.random.split(jax.random.PRNGKey(230), 32)
    peak = likelihood(jnp.full(18, .5))
    results = []
    for cls in (ReflectionSampler, FrozenContourReflectionSampler):
        sampler = cls(num_cycles=9, slices_per_cycle=1, num_bounces=10,
                      step_size=.003, num_phantom_samples=8)

        @jax.jit
        @jax.vmap
        def run(key, u):
            return sampler.run_chain(key, u, likelihood(u), peak - 15., likelihood)

        results.append(run(keys, jnp.asarray(points)))
    for a, b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        np.testing.assert_array_equal(a, b)
    final, (path, values) = results[1]
    assert np.all(np.asarray(values) > float(peak) - 15.)
    assert np.all((np.asarray(path) >= 0) & (np.asarray(path) <= 1))
    np.testing.assert_array_equal(final.gradients, 90)
    assert np.all(np.asarray(final.evaluations) >= 108)


def test_scheduled_reflection_reversal_rejections_and_call_counts():
    def likelihood(point):
        residual = point[0] - .5 - .3 * jnp.sin(2 * jnp.pi * point[1])
        return -.5*((residual/1e-5)**2 + jnp.sum(((point[2:16]-.5)/.05)**2))

    sampler = FrozenContourReflectionSampler(
        num_cycles=9, slices_per_cycle=0, num_bounces=10,
        gaps=(1., 49.), steps=(.003, .005),
    )
    points, _ = exact_draws(np.random.default_rng(231), 256, 15.)
    velocity = np.random.default_rng(232).normal(size=points.shape)
    h = sampler.step_for_contour(jnp.asarray(-15.))

    @jax.jit
    @jax.vmap
    def advance(point, direction):
        return reflection_trajectory(point, direction, h, 10, jnp.asarray(-15.),
                                     jax.value_and_grad(likelihood))

    forward, directions = advance(jnp.asarray(points), jnp.asarray(velocity))
    reverse, reverse_velocity = advance(forward, -directions)
    np.testing.assert_allclose(reverse, points, rtol=0, atol=2e-9)
    np.testing.assert_allclose(reverse_velocity, -velocity, rtol=0, atol=2e-8)

    @jax.jit
    @jax.vmap
    def run(key, point):
        return sampler.run_chain(key, point, likelihood(point), jnp.asarray(-15.), likelihood)

    final, (paths, values) = run(jax.random.split(jax.random.PRNGKey(233), 256), points)
    differences = np.diff(np.concatenate([points[:, None], paths], axis=1), axis=1)
    changes = np.any(differences != 0, axis=2)
    np.testing.assert_array_equal(final.accepted, changes.sum(axis=1))
    assert 0 < np.sum(changes) < changes.size
    assert np.all(np.asarray(values) > -15.)
    np.testing.assert_array_equal(final.evaluations, 99)
    np.testing.assert_array_equal(final.gradients, 90)


@pytest.mark.parametrize('allocation', ['uniform', 'evidence_improving'])
def test_scheduled_phantom_retention_preserves_entire_classic_continuation(allocation):
    model = make_model()
    sampler = FrozenContourReflectionSampler(
        num_cycles=9, slices_per_cycle=1, num_bounces=10,
        gaps=(1., 49.), steps=(.001, .002), peak=39.,
    )
    states = []
    for retain in (False, True):
        runner = NestedSampler(
            model=model, root_allocation_degree=12, replacement_width=4, delta_K=12,
            initial_capacity=64, unlimited_samples=True, allocation_target=allocation,
            sampler=sampler, collect_phantom_samples=retain,
        )
        state = run_with_observer(
            runner, runner.initialise(jax.random.PRNGKey(3122798)),
            lambda before, after, work: int(after.depth_loop_iter) >= 8,
            lambda current: False,
        )
        assert state.samples.phantom_samples.U_samples is None
        if retain:
            assert state.samples.phantom_samples.log_L.shape[1] == 8
        # Only the requested payload differs; every remaining scientific and
        # continuation leaf, including scheduling and RNG, must be bit identical.
        empty = PhantomSamples(None, jnp.zeros(0, bool), jnp.empty((0, 0)))
        states.append(dataclasses.replace(
            state, samples=dataclasses.replace(state.samples, phantom_samples=empty),
        ))
    assert jax.tree.structure(states[0]) == jax.tree.structure(states[1])
    for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(a, b)


def test_exact_reference_support_and_model_precision():
    model = make_model()
    point, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(234)))
    assert point.dtype == jnp.float64 and point.shape == (18,)
    likelihood = jax.jit(jax.vmap(lambda u: model.log_likelihood(unravel(u))))
    for gap in (5., 15., 40., 100., 1000., np.inf):
        points, _ = exact_draws(np.random.default_rng(235), 1024, gap)
        values = np.asarray(likelihood(jnp.asarray(points)))
        np.testing.assert_allclose(values, numpy_log_likelihood(points), rtol=0, atol=5e-5)
        assert np.all((points >= 0) & (points <= 1))
        peak = numpy_log_likelihood(np.full((1, 18), .5))[0]
        assert np.all(values > peak - gap)
