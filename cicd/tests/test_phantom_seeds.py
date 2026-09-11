"""Independent dense population checks for the all-phantom experiment."""

import dataclasses
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.test_core import _allocation_plan
from jaxns.algorithm import depth
from jaxns.algorithm.race_tree import build_block_state
from jaxns.samples import PhantomSamples
from jaxns.sampling.phantom_seeds import (
    gather_seed_points,
    phantom_rank_to_identity,
    phantom_seed_cumulative,
    phantom_seed_cumulative_batch,
)


def phantom_state():
    state = make_state(
        root_out_degree=2, log_likelihoods=(1., 2., 4., 5.),
        log_L_constraints=(-np.inf, -np.inf, 1., 2.),
        out_degree=(1, 1, 0, 0), max_samples=5, num_phantom=3,
    )
    log_L = np.array([
        [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf],
        [3., 2., 5.], [4., 5., 3.], [10., 10., 10.],
    ])
    valid = np.array([
        [False] * 3, [False] * 3, [True, False, True],
        [True] * 3, [True] * 3,
    ])
    phantoms = PhantomSamples(
        U_samples=jnp.arange(15., dtype=float).reshape(5, 3) / 15,
        log_L=jnp.asarray(log_L), valid_mask=jnp.asarray(valid),
        seed_log_L_sorted=jnp.asarray(np.sort(
            np.where(valid, log_L, -np.inf), axis=1,
        )),
    )
    return dataclasses.replace(
        state, samples=dataclasses.replace(
            state.samples, phantom_samples=phantoms,
        ),
    )


@pytest.mark.parametrize("constraint", [-np.inf, 0., 1., 2., 3., 4., 5.])
def test_hierarchical_selection_matches_dense_population(constraint):
    state = phantom_state()
    samples = state.samples
    phantoms = samples.phantom_samples
    expected = [
        row * 4 + slot + 1
        for row in range(int(state.num_samples)) for slot in range(3)
        if samples.log_L_constraints[row] <= constraint
        and phantoms.valid_mask[row, slot]
        and phantoms.log_L[row, slot] > constraint
    ]
    cumulative = jax.jit(phantom_seed_cumulative)(
        samples, state.num_samples, jnp.float64(constraint),
    )
    assert int(cumulative[-1]) == len(expected)
    if expected:
        actual = jax.jit(phantom_rank_to_identity)(
            samples, cumulative, jnp.float64(constraint),
            jnp.arange(len(expected)),
        )
        np.testing.assert_array_equal(actual, expected)


def test_identity_coordinates_survive_growth_trim_and_checkpoint():
    state = phantom_state()
    identities = jnp.array([0, 4, 9, 11, 13, 15])
    expected = gather_seed_points(state.samples, identities)
    for restored in (state.resize(13), pickle.loads(pickle.dumps(state))):
        actual = gather_seed_points(restored.samples, identities)
        np.testing.assert_array_equal(actual.U0, expected.U0)
        np.testing.assert_array_equal(actual.log_L0, expected.log_L0)
        trimmed = restored.trim()
        assert trimmed.samples.phantom_samples.U_samples.shape == (4, 3)
        assert (
            trimmed.samples.phantom_samples.seed_log_L_sorted.shape == (4, 3)
        )
    np.testing.assert_array_equal(
        expected.log_L0, np.array([1., 2., 3., 5., 4., 3.]),
    )


def test_original_no_replacement_rule_includes_all_phantoms():
    state = phantom_state()
    blocks = build_block_state(
        state.samples, state.root_out_degree, state.num_samples,
    )
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (0, 0, 7, 0, 0)),
        blocks.valid, shell_size=7, tail_K=jnp.int32(0),
    )
    selected = jax.jit(depth._sample_stationary_seeds)(
        jax.random.PRNGKey(51), state, schedule,
        jnp.full(7, 2.), jnp.zeros(7, dtype=bool),
        jnp.ones(7, dtype=bool), jnp.full(7, -1),
        jnp.full(7, -jnp.inf), jnp.zeros(7, dtype=bool),
    )
    # Two classics and all five eligible phantoms, each exactly once.
    np.testing.assert_array_equal(
        np.sort(selected), [8, 9, 11, 12, 13, 14, 15],
    )
    points = gather_seed_points(state.samples, selected)
    assert np.all(points.log_L0 > 2.)


def test_batched_contours_exactly_match_independent_queries():
    state = phantom_state().resize(19)
    constraints = jnp.array([-jnp.inf, 0., 1., 2., 2.5, 3., 4., 5., jnp.inf])
    expected = jnp.stack([
        phantom_seed_cumulative(state.samples, state.num_samples, value)
        for value in constraints
    ])
    actual = jax.jit(phantom_seed_cumulative_batch)(
        state.samples, state.num_samples, constraints,
    )
    np.testing.assert_array_equal(actual, expected)


def test_phantom_can_make_an_unpublished_contour_seedable():
    state = phantom_state()
    phantoms = state.samples.phantom_samples
    log_L = phantoms.log_L.at[3, 1].set(6.)
    state = dataclasses.replace(state, samples=dataclasses.replace(
        state.samples, phantom_samples=dataclasses.replace(
            phantoms, log_L=log_L,
            seed_log_L_sorted=jnp.sort(jnp.where(
                phantoms.valid_mask, log_L, -jnp.inf,
            ), axis=1),
        ),
    ))
    blocks = build_block_state(
        state.samples, state.root_out_degree, state.num_samples,
    )
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (0, 0, 1, 0, 0)),
        blocks.valid, shell_size=1, tail_K=jnp.int32(0),
    )
    has_seed, _, constraint = depth._effective_parent_contour(
        schedule, jnp.float64(5.), state,
    )
    assert has_seed and float(constraint) == 5.
    selected = jax.jit(depth._sample_stationary_seeds)(
        jax.random.PRNGKey(52), state, schedule,
        jnp.array([5.]), jnp.array([False]), jnp.array([True]),
        jnp.array([-1]), jnp.array([-jnp.inf]), jnp.array([False]),
    )
    np.testing.assert_array_equal(selected, [14])
