"""Frozen-duration correctness: heterogeneous work, reversibility and continuation."""

import dataclasses
import json
import pickle
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.allocation_observer import run_with_observer
from benchmarks.issue_312.duration_common import make_arm, numpy_parameters
from benchmarks.issue_312.duration_schedule import DurationContourReflectionSampler
from benchmarks.issue_312.reflection_sampler import ReflectionSampler, reflection_trajectory
from benchmarks.issue_312.tuning_common import exact_draws, make_model
from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
from jaxns.core import NestedSampler
from jaxns.pytree import pytree_ravel
from jaxns.samples import PhantomSamples


@pytest.fixture
def frozen_schedule():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/issue_312/reflection_tuning'
    # The publication artifact is the same inherited schedule, not a new pilot.
    return json.loads((path / 'SCHEDULE.json').read_text())


def test_duration_rounding_nodes_interiors_and_tails(frozen_schedule):
    _, parameters = make_arm('duration', frozen_schedule)
    positive = np.geomspace(frozen_schedule['gaps'][0] / 100, 49, 10001)
    gaps = np.concatenate((positive, frozen_schedule['gaps'], [0, -1, 100, np.inf]))
    contours = frozen_schedule['peak'] - gaps
    count, step = jax.jit(jax.vmap(parameters))(jnp.asarray(contours))
    expected_count, expected_step = numpy_parameters('duration', frozen_schedule, contours)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_allclose(step, expected_step, rtol=2e-15, atol=0)
    np.testing.assert_allclose(count * step, .03, rtol=2e-15, atol=0)
    assert int(count.min()) == 10 and int(count.max()) == 152
    _, desired = numpy_parameters('short', frozen_schedule, contours)
    assert np.all(np.asarray(step) <= desired * (1 + 2e-15))
    assert np.all(np.asarray(step)[gaps > 49] == .003)


def test_constant_duration_and_old_short_are_exact_controls():
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122715)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    points, _ = exact_draws(np.random.default_rng(3122715), 32, 15.)
    keys = jax.random.split(jax.random.PRNGKey(3122716), len(points))
    peak = likelihood(jnp.full(18, .5))
    results = []
    for cls in (ReflectionSampler, FrozenContourReflectionSampler,
                DurationContourReflectionSampler):
        sampler = cls(num_cycles=9, slices_per_cycle=1, num_bounces=10,
                      step_size=.003, num_phantom_samples=8)

        @jax.jit
        @jax.vmap
        def run(key, point):
            return sampler.run_chain(key, point, likelihood(point), peak - 15, likelihood)

        results.append(run(keys, jnp.asarray(points)))
    for result in results[1:]:
        for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(results[0]),
                                    strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert jax.default_backend() == 'cpu' and results[-1][0].point.dtype == jnp.float64


def test_heterogeneous_counts_support_and_rejection_repeats(frozen_schedule):
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122715)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    sampler, parameters = make_arm('duration', frozen_schedule)
    sampler = dataclasses.replace(sampler, slices_per_cycle=0)
    gaps = np.tile([.1, 5., 15., 40., 100.], 16)
    points = np.concatenate([
        exact_draws(np.random.default_rng([3122715, i]), 1, gap)[0]
        for i, gap in enumerate(gaps)
    ])
    contours = jnp.asarray(frozen_schedule['peak'] - gaps)
    keys = jax.random.split(jax.random.PRNGKey(3122717), len(points))

    @jax.jit
    @jax.vmap
    def run(key, point, contour):
        return sampler.run_chain(key, point, likelihood(point), contour, likelihood)

    final, (paths, values) = run(keys, jnp.asarray(points), contours)
    counts, _ = jax.vmap(parameters)(contours)
    np.testing.assert_array_equal(final.gradients, 9 * counts)
    np.testing.assert_array_equal(final.evaluations, 9 * (counts + 1))
    changes = np.any(np.diff(np.concatenate([points[:, None], paths], axis=1), axis=1) != 0,
                     axis=2)
    np.testing.assert_array_equal(final.accepted, changes.sum(axis=1))
    assert changes.any() and not changes.all()
    assert np.all(np.asarray(values) > np.asarray(contours)[:, None])
    assert np.all((np.asarray(paths) >= 0) & (np.asarray(paths) <= 1))
    # A smaller heterogeneous batch has the same lane results and no cross-lane state.
    smaller = run(keys[:5], jnp.asarray(points[:5]), contours[:5])
    for a, b in zip(jax.tree.leaves(smaller), jax.tree.leaves((final, (paths, values))),
                    strict=True):
        np.testing.assert_array_equal(a, b[:5])


@pytest.mark.parametrize('gap', [.1, 5., 100.])
def test_duration_phase_space_reversal_including_maximum_count(frozen_schedule, gap):
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122715)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    _, parameters = make_arm('duration', frozen_schedule)
    contour = jnp.asarray(frozen_schedule['peak'] - gap)
    count, nominal_step = parameters(contour)
    points, _ = exact_draws(np.random.default_rng(3122715), 64, gap)
    rng = np.random.default_rng(3122716)
    velocities = rng.normal(size=points.shape)
    steps = np.asarray(nominal_step) * rng.uniform(.8, 1.2, len(points))

    @jax.jit
    @jax.vmap
    def advance(point, direction, step):
        return reflection_trajectory(point, direction, step, count, contour,
                                     jax.value_and_grad(likelihood))

    forward, direction = advance(jnp.asarray(points), jnp.asarray(velocities), steps)
    backward, reversed_direction = advance(forward, -direction, steps)
    np.testing.assert_allclose(backward, points, rtol=0, atol=2e-9)
    np.testing.assert_allclose(reversed_direction, -velocities, rtol=0, atol=2e-8)


@pytest.mark.parametrize('allocation', ['uniform', 'evidence_improving'])
def test_variable_count_phantoms_checkpoint_growth_continuation(allocation):
    model = make_model()
    # Unit-test contours cover prior-scale values so actual core batches have
    # heterogeneous counts; this synthetic schedule is never used in evaluation.
    sampler = DurationContourReflectionSampler(
        num_cycles=9, slices_per_cycle=1, num_bounces=10,
        gaps=(1e6, 1e10), steps=(.0003, .003), peak=0.,
    )
    states, observed_counts = [], []
    for retain in (False, True):
        runner = NestedSampler(
            model=model, root_allocation_degree=12, replacement_width=4, delta_K=12,
            initial_capacity=32, unlimited_samples=True, allocation_target=allocation,
            sampler=sampler, collect_phantom_samples=retain,
        )

        def observe(before, after, work):
            valid = int(work.num_valid)
            counts, _ = jax.vmap(sampler.parameters_for_contour)(work.log_L_constraint[:valid])
            observed_counts.extend(np.asarray(counts).tolist())
            return int(after.depth_loop_iter) >= 16

        state = runner.initialise(jax.random.PRNGKey(3122717))
        if retain:
            state = run_with_observer(
                runner, state, lambda before, after, work: int(after.depth_loop_iter) >= 3,
                lambda current: False,
            )
            state = pickle.loads(pickle.dumps(jax.device_get(state)))
        state = run_with_observer(runner, state, observe, lambda current: False)
        assert state.samples.phantom_samples.U_samples is None
        assert state.samples.log_likelihoods.shape[0] > 32
        empty = PhantomSamples(None, jnp.zeros(0, bool), jnp.empty((0, 0)))
        states.append(dataclasses.replace(
            state, samples=dataclasses.replace(state.samples, phantom_samples=empty),
        ))
    assert len(set(observed_counts)) > 1
    assert jax.tree.structure(states[0]) == jax.tree.structure(states[1])
    for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(a, b)
