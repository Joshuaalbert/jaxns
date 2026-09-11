"""Exact population and frozen-selector checks for block-indexed phantom A."""

import dataclasses
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.phantom_seeding.reference import reference_stationary_seeds
from cicd.tests.core_fixtures import make_state
from cicd.tests.test_core import _allocation_plan
from cicd.tests.test_phantom_seeds import phantom_state
from jaxns.algorithm import depth
from jaxns.algorithm.race_tree import build_block_state
from jaxns.samples import PhantomSamples
from jaxns.sampling.phantom_index import (
    build_phantom_seed_index,
    phantom_block_cumulative,
    phantom_block_rank_to_identity,
    update_phantom_seed_index,
)
from jaxns.sampling.phantom_seeds import gather_seed_points


def indexed_state(state, block_size):
    index = build_phantom_seed_index(
        state.samples, state.num_samples, block_size=block_size,
    )
    return dataclasses.replace(
        state,
        samples=dataclasses.replace(
            state.samples,
            phantom_samples=dataclasses.replace(
                state.samples.phantom_samples, seed_log_L_sorted=None,
            ),
        ),
        phantom_seed_index=index,
    )


@pytest.mark.parametrize("block_size", [1, 2, 3, 8])
def test_all_ranks_match_enumeration_with_ties_infinities_and_partial_tails(block_size):
    state = phantom_state().resize(53)
    rng = np.random.default_rng(247)
    birth = rng.integers(-3, 8, 53).astype(float)
    birth[:2] = -np.inf
    likelihood = rng.integers(-3, 13, (53, 3)).astype(float)
    likelihood[4, 1] = np.inf
    likelihood[5, :] = birth[5]
    valid = rng.random((53, 3)) > 0.25
    valid[:2] = False
    samples = dataclasses.replace(
        state.samples,
        log_L_constraints=jnp.asarray(birth),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((53, 3)), log_L=jnp.asarray(likelihood),
            valid_mask=jnp.asarray(valid),
        ),
    )
    constraints = jnp.asarray([-np.inf, -3., 0., 2., 4., 7., 10., 12., np.inf])
    for accepted in (0, 1, 5, 16, 52, 53):
        num_samples = jnp.asarray(accepted)
        index = build_phantom_seed_index(samples, num_samples, block_size=block_size)
        cumulative = phantom_block_cumulative(samples, num_samples, constraints, index)
        for lane, constraint in enumerate(np.asarray(constraints)):
            expected = [
                row * 4 + slot + 1
                for row in range(accepted) for slot in range(3)
                if valid[row, slot] and birth[row] <= constraint < likelihood[row, slot]
            ]
            assert int(cumulative[lane, -1]) == len(expected)
            if expected:
                actual = phantom_block_rank_to_identity(
                    samples, num_samples, cumulative[lane], jnp.asarray(constraint),
                    jnp.arange(len(expected)), index,
                )
                np.testing.assert_array_equal(actual, expected)


def test_partial_tail_is_visible_before_sealing_and_survives_growth_resume():
    original = phantom_state()
    state = indexed_state(original, 3)  # three sealed rows, one accepted tail row
    constraints = jnp.asarray([1., 2., 3., 4., 5., np.inf])
    baseline = phantom_block_cumulative(
        state.samples, state.num_samples, constraints, state.phantom_seed_index,
    )[:, -1]
    assert int(baseline[1]) == 5  # includes all three phantoms in the unfinished tail
    for restored in (state.resize(19), state.trim(), pickle.loads(pickle.dumps(state))):
        assert restored.samples.phantom_samples.seed_log_L_sorted is None
        np.testing.assert_array_equal(
            phantom_block_cumulative(
                restored.samples, restored.num_samples, constraints, restored.phantom_seed_index,
            )[:, -1],
            baseline,
        )
    # Rows beyond the accepted prefix contain deliberately eligible junk.
    # Accepting that row changes only tail visibility; no full block is sealed.
    grown = state.resize(19)
    next_count = jnp.asarray(5)
    index = update_phantom_seed_index(grown.phantom_seed_index, grown.samples, next_count)
    rebuilt = build_phantom_seed_index(grown.samples, next_count, block_size=3)
    for actual, expected in zip(jax.tree.leaves(index), jax.tree.leaves(rebuilt)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("block_size", [2, 3, 8])
@pytest.mark.parametrize("shell_size", [1, 7, 17])
def test_selected_identities_match_frozen_a_with_collisions_and_exhaustion(block_size, shell_size):
    state = phantom_state()
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (0, 0, shell_size, 0, 0)),
        blocks.valid, shell_size=shell_size, tail_K=jnp.int32(0),
    )
    candidate = indexed_state(state, block_size)
    reference = jax.jit(reference_stationary_seeds)
    actual = jax.jit(depth._sample_stationary_seeds)
    for heterogeneous in (False, True):
        constraints = jnp.asarray(
            np.resize([-np.inf, 1., 2., 3., 4.], shell_size)
            if heterogeneous else np.full(shell_size, 2.),
        )
        valid = jnp.arange(shell_size) < max(shell_size - 2, 1)
        reserved_ids = jnp.full(shell_size, -1).at[0].set(9)
        reserved_contours = jnp.full(shell_size, 2.)
        reserved_valid = jnp.arange(shell_size) == 0
        args = (
            schedule, constraints, jnp.isneginf(constraints), valid,
            reserved_ids, reserved_contours, reserved_valid,
        )
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            expected_ids = reference(key, state, *args)
            np.testing.assert_array_equal(actual(key, state, *args), expected_ids)
            selected = actual(key, candidate, *args)
            np.testing.assert_array_equal(selected, expected_ids)
            expected_points = gather_seed_points(state.samples, expected_ids)
            actual_points = gather_seed_points(candidate.samples, selected)
            for left, right in zip(
                jax.tree.leaves(actual_points), jax.tree.leaves(expected_points),
            ):
                np.testing.assert_array_equal(left, right)


def test_block_index_preserves_unpublished_contour_and_fallback():
    state = phantom_state()
    phantoms = state.samples.phantom_samples
    log_L = phantoms.log_L.at[3, 1].set(6.)
    state = dataclasses.replace(state, samples=dataclasses.replace(
        state.samples, phantom_samples=dataclasses.replace(
            phantoms, log_L=log_L,
            seed_log_L_sorted=jnp.sort(jnp.where(phantoms.valid_mask, log_L, -jnp.inf)),
        ),
    ))
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (0, 0, 1, 0, 0)),
        blocks.valid, shell_size=1, tail_K=jnp.int32(0),
    )
    candidate = indexed_state(state, 3)
    for constraint in (-np.inf, 2., 5., 6., np.inf):
        reference = depth._effective_parent_contour(schedule, jnp.asarray(constraint), state)
        actual = depth._effective_parent_contour(schedule, constraint, candidate)
        for left, right in zip(actual, reference):
            np.testing.assert_array_equal(left, right)


def test_completed_block_update_matches_rebuild():
    state = phantom_state().resize(17)
    samples = dataclasses.replace(
        state.samples, log_L_constraints=jnp.full((17,), -1.),
    )
    index = build_phantom_seed_index(samples, jnp.asarray(0), block_size=3)
    for accepted in (1, 2, 3, 7, 8, 15, 17):
        index = update_phantom_seed_index(index, samples, jnp.asarray(accepted))
        expected = build_phantom_seed_index(samples, jnp.asarray(accepted), block_size=3)
        for left, right in zip(jax.tree.leaves(index), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(left, right)


def test_merging_reference_and_indexed_states_rebuilds_canonical_counts():
    left = phantom_state().trim()
    right = indexed_state(phantom_state().trim(), 3)
    for merged in (left.merge(right), right.merge(left)):
        assert merged.samples.phantom_samples.seed_log_L_sorted is None
        expected = build_phantom_seed_index(
            merged.samples, merged.num_samples, block_size=3,
        )
        for actual, reference in zip(
                jax.tree.leaves(merged.phantom_seed_index), jax.tree.leaves(expected),
        ):
            np.testing.assert_array_equal(actual, reference)


def test_retained_start_group_reservations_survive_index_growth():
    state = phantom_state()
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (0, 0, 8, 0, 0)),
        blocks.valid, shell_size=3, tail_K=jnp.int32(0),
    )
    schedule = dataclasses.replace(
        schedule, new_start=jnp.ones(3, dtype=bool),
        start_seed_log_L_constraint=jnp.asarray(2.),
        num_start_seeds=jnp.asarray(2), num_published_start_seeds=jnp.asarray(2),
    )
    for identity in (8, 9):  # one classic and one phantom from a preceding batch
        slots, groups = depth._insert_seed_reservation(
            jnp.asarray(identity), schedule.current_start_group,
            schedule.start_seed_reservation_idx, schedule.start_seed_reservation_group,
        )
        schedule = dataclasses.replace(
            schedule, start_seed_reservation_idx=slots, start_seed_reservation_group=groups,
        )
    candidate = indexed_state(state, 3).resize(19)
    reference = jax.jit(reference_stationary_seeds)
    actual = jax.jit(depth._sample_stationary_seeds)
    constraints = jnp.full(3, 2.)
    args = (
        constraints, jnp.zeros(3, dtype=bool), jnp.ones(3, dtype=bool),
        jnp.full(3, -1), jnp.full(3, -jnp.inf), jnp.zeros(3, dtype=bool),
    )
    for seed in range(10):
        current = schedule
        for batch in range(3):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), batch)
            expected = reference(key, state, current, *args)
            np.testing.assert_array_equal(actual(key, candidate, current, *args), expected)
            current = depth._retain_start_seed_reservations(
                current, expected, constraints, jnp.ones(3, dtype=bool), state,
            )


def test_multiple_random_proposal_batches_match_frozen_reference():
    state = make_state(
        root_out_degree=100, log_likelihoods=tuple(float(i + 1) for i in range(100)),
        out_degree=(0,) * 100, max_samples=100, num_phantom=3,
    )
    likelihood = jnp.ones((100, 3))
    state = dataclasses.replace(state, samples=dataclasses.replace(
        state.samples, U_samples=jnp.linspace(0.1, 0.9, 100),
        phantom_samples=PhantomSamples(
            U_samples=jnp.linspace(0.1, 0.9, 300).reshape(100, 3),
            log_L=likelihood, valid_mask=jnp.ones((100, 3), dtype=bool),
            seed_log_L_sorted=likelihood,
        ),
    ))
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    schedule = depth._new_thread_schedule(
        state, blocks, _allocation_plan(blocks, (400,) + (0,) * 99),
        blocks.valid, shell_size=1, tail_K=jnp.int32(0),
    )
    schedule = schedule.resize_start_seed_reservations(1024)

    def reserve(identity, entries):
        return depth._insert_seed_reservation(
            identity, schedule.current_start_group, entries[0], entries[1],
        )

    slots, groups = jax.lax.fori_loop(
        0, 397, reserve,
        (schedule.start_seed_reservation_idx, schedule.start_seed_reservation_group),
    )
    schedule = dataclasses.replace(
        schedule, new_start=jnp.ones(1, dtype=bool),
        start_seed_log_L_constraint=jnp.asarray(-jnp.inf),
        start_seed_reservation_idx=slots, start_seed_reservation_group=groups,
        num_start_seeds=jnp.asarray(397), num_published_start_seeds=jnp.asarray(397),
    )
    candidate = indexed_state(state, 64)
    reference = jax.jit(reference_stationary_seeds)
    actual = jax.jit(depth._sample_stationary_seeds)
    args = (
        schedule, jnp.asarray([-jnp.inf]), jnp.asarray([True]), jnp.asarray([True]),
        jnp.asarray([-1]), jnp.asarray([-jnp.inf]), jnp.asarray([False]),
    )
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        expected = reference(key, state, *args)
        assert int(expected[0]) in (397, 398, 399)
        np.testing.assert_array_equal(actual(key, candidate, *args), expected)


@pytest.mark.parametrize("block_size", [0, -1])
def test_invalid_block_size_fails_during_tracing(block_size):
    state = phantom_state()
    with pytest.raises(ValueError, match="block size must be positive"):
        build_phantom_seed_index.lower(
            state.samples, state.num_samples, block_size=block_size,
        )
