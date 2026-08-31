"""Focused contracts for the compiled scheduling core."""

import dataclasses
import inspect
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns import constrained_sampler, core
from jaxns.algorithm import depth
from jaxns.algorithm.allocation import (
    AllocationPlan,
    VolumePath,
)
from jaxns.algorithm.race_tree import (
    build_block_state,
    initialise_likelihood_order,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
    EllipsoidalDirection,
    UniDimSliceSampler,
    _take_phantom_prefix,
)
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.ellipsoid import empty_sampler_data


@dataclasses.dataclass(slots=True, frozen=True)
class DeterministicSampler(PureDataclassPytree, AbstractSampler):
    """Traceable sampler used to isolate scheduling from MCMC behavior."""

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
    ):
        del key, args, params
        finite_constraint = jnp.where(
            jnp.isfinite(log_L_constraint),
            log_L_constraint,
            0.0,
        )
        return (
            seed_point.U0,
            finite_constraint + 1.0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=jnp.zeros(
                    (0,) + jnp.shape(seed_point.U0),
                    dtype=jnp.asarray(seed_point.U0).dtype,
                ),
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros((0,), dtype=jnp.asarray(log_L_constraint).dtype),
            ),
        )


DeterministicSampler.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class TwoDimensionalModel(PureDataclassPytree):
    """Small vector model for exercising direction geometry in the core."""

    def _periodic_coordinates(self, args=(), params=None) -> tuple[bool, ...]:
        """Match the internal model geometry contract used by runners."""
        del args, params
        return False, False

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 2

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, shape=(2,))

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(self, U, args=(), params=None, *, allow_nan=True):
        del args, params, allow_nan
        return -jnp.sum(jnp.square(U - jnp.asarray([0.3, 0.7])))

    def log_prior(self, U, args=(), params=None):
        del args, params
        return jnp.where(jnp.all((U >= 0.0) & (U <= 1.0)), 0.0, -jnp.inf)


TwoDimensionalModel.register_pytree()


def _allocation_plan(
        block_state,
        gap: tuple[int, ...],
) -> AllocationPlan:
    """Build a minimal exact-gap plan for scheduler contract tests."""
    gap_array = jnp.asarray(gap, dtype=jnp.int32)  # [G]
    return AllocationPlan(
        target_K=block_state.incoming_K + gap_array,
        current_K=block_state.incoming_K,
        unit_peak_utility=block_state.valid.astype(jnp.float64),
        log_L_blocks=block_state.log_L_blocks,
        valid=block_state.valid,
        volume_path=VolumePath(
            X_prev=jnp.ones(gap_array.shape),
            X=jnp.ones(gap_array.shape),
            shell_mass=jnp.zeros(gap_array.shape),
        ),
    )


def test_continuations_wait_until_each_frozen_thread_has_started():
    """A narrow sampler window rotates breadth-first over a wider gap."""
    state = make_state(
        root_out_degree=6,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        log_L_constraints=(-np.inf,) * 6,
        out_degree=(0,) * 6,
        max_samples=6,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        state,
        block_state,
        _allocation_plan(block_state, (4, 0, 0, 0, 0, 0)),
        block_state.valid,
        shell_size=2,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )

    first = depth._fill_thread_heads(
        jax.random.PRNGKey(30),
        state,
        schedule,
    )
    np.testing.assert_array_equal(first.thread_id, np.asarray([0, 1]))
    first = depth._enqueue_thread_continuations(
        first,
        jnp.asarray([2, 3], dtype=jnp.int32),
        first.thread_id,
        first.terminal_log_L,
        first.valid,
    )
    first = depth._release_thread_heads(first, first.valid)

    second = depth._fill_thread_heads(
        jax.random.PRNGKey(31),
        state,
        first,
    )
    np.testing.assert_array_equal(second.thread_id, np.asarray([2, 3]))
    second = depth._release_thread_heads(second, second.valid)

    resumed = depth._fill_thread_heads(
        jax.random.PRNGKey(32),
        state,
        second,
    )
    np.testing.assert_array_equal(resumed.thread_id, np.asarray([0, 1]))
    np.testing.assert_array_equal(resumed.parent_idx, np.asarray([2, 3]))
    assert int(resumed.continuation_count) == 0


def test_same_contour_parallel_threads_use_distinct_stationary_seeds():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=4,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        state,
        block_state,
        _allocation_plan(block_state, (2, 0, 0, 0)),
        block_state.valid,
        shell_size=4,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(7),
        state,
        schedule,
        jnp.asarray(4, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(
        np.asarray(work.valid),
        np.asarray([True, True, False, False]),
    )
    assert int(work.seed_idx[0]) != int(work.seed_idx[1])


def test_mixed_contour_seed_groups_remain_distinct_after_rejection():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0),
        log_L_constraints=(-np.inf, -np.inf, 1.0, 1.0),
        out_degree=(2, 0, 0, 0),
        max_samples=5,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        state,
        block_state,
        _allocation_plan(block_state, (0, 0, 0, 0, 0)),
        block_state.valid,
        shell_size=5,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    constraints = jnp.asarray(
        [-jnp.inf, -jnp.inf, 1.0, 1.0, 1.0],
    )
    selected = depth._sample_stationary_seeds(
        jax.random.PRNGKey(0),
        state,
        schedule,
        constraints,
        jnp.isneginf(constraints),
        jnp.ones((5,), dtype=bool),
        jnp.full((5,), -1, dtype=jnp.int32),
        jnp.full((5,), -jnp.inf),
        jnp.zeros((5,), dtype=bool),
    )
    selected = np.asarray(selected)

    assert len(set(selected[:2].tolist())) == 2
    assert set(selected[2:].tolist()) == {1, 2, 3}
    for seed_idx, constraint in zip(selected, constraints, strict=True):
        assert state.samples.log_likelihoods[seed_idx] > constraint
        assert state.samples.log_L_constraints[seed_idx] <= constraint


def test_pending_same_contour_seeds_are_reserved_across_refills():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0),
        log_L_constraints=(-np.inf, -np.inf, 1.0, 1.0),
        out_degree=(2, 0, 0, 0),
        max_samples=4,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        state,
        block_state,
        _allocation_plan(block_state, (0, 0, 0, 0)),
        block_state.valid,
        shell_size=2,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    selected = depth._sample_stationary_seeds(
        jax.random.PRNGKey(1),
        state,
        schedule,
        jnp.asarray([1.0, 1.0]),
        jnp.zeros((2,), dtype=bool),
        jnp.ones((2,), dtype=bool),
        jnp.asarray([1, -1]),
        jnp.asarray([1.0, -jnp.inf]),
        jnp.asarray([True, False]),
    )

    assert set(np.asarray(selected).tolist()) == {2, 3}


def test_seed_pool_includes_completed_rows_accepted_after_freeze():
    source = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 2),
        max_samples=4,
    )
    block_state = build_block_state(
        source.samples,
        source.root_out_degree,
        source.num_samples,
        likelihood_order=source.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        source,
        block_state,
        _allocation_plan(block_state, (0, 0, 0, 0)),
        block_state.valid,
        shell_size=2,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    current = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0),
        log_L_constraints=(-np.inf, -np.inf, 2.0, 2.0),
        out_degree=(0, 2, 0, 0),
        max_samples=4,
    )
    schedule = depth._update_seed_reservoir(
        schedule,
        jnp.asarray([2, 3], dtype=jnp.int32),
        jnp.asarray([True, True]),
    )
    keys = jax.random.split(jax.random.PRNGKey(19), 64)

    def draw(one_key):
        return depth._sample_stationary_seeds(
            one_key,
            current,
            schedule,
            jnp.asarray([2.0]),
            jnp.asarray([False]),
            jnp.asarray([True]),
            jnp.asarray([-1], dtype=jnp.int32),
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
        )[0]

    # Both appended rows are stationary at L=2. A cache of ongoing heads could
    # expose only one because thread survival censors it below its terminal.
    assert set(np.asarray(jax.vmap(draw)(keys)).tolist()) == {2, 3}


def test_appended_seed_population_remains_distinct_when_large_enough():
    """Eligible reservoir rows extend the exact no-replacement pool."""
    source = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=6,
    )
    block_state = build_block_state(
        source.samples,
        source.root_out_degree,
        source.num_samples,
        likelihood_order=source.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        source,
        block_state,
        _allocation_plan(block_state, (0, 0, 0, 0, 0, 0)),
        block_state.valid,
        shell_size=3,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    current = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        log_L_constraints=(-np.inf, -np.inf, 1.0, 1.0, 2.0, 2.0),
        out_degree=(2, 2, 0, 0, 0, 0),
        max_samples=6,
    )
    # Fix the bounded reservoir to two eligible rows and one ineligible row.
    # Without its exact eligibility count the three lanes can incorrectly
    # reuse a seed even though identities {1, 2, 3} are all available.
    schedule = dataclasses.replace(
        schedule,
        seed_reservoir_idx=jnp.asarray([2, 3, 4], dtype=jnp.int32),
        seed_reservoir_priority=jnp.asarray([0.9, 0.8, 0.7]),
        seed_reservoir_valid=jnp.asarray([True, True, True]),
    )
    constraints = jnp.full((3,), 1.0)

    selected = depth._sample_stationary_seeds(
        jax.random.PRNGKey(29),
        current,
        schedule,
        constraints,
        jnp.zeros((3,), dtype=bool),
        jnp.ones((3,), dtype=bool),
        jnp.full((3,), -1, dtype=jnp.int32),
        jnp.full((3,), -jnp.inf),
        jnp.zeros((3,), dtype=bool),
    )

    assert set(np.asarray(selected).tolist()) == {1, 2, 3}


def test_seed_source_refresh_is_bounded_by_planning_width():
    source = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=4,
    )
    block_state = build_block_state(
        source.samples,
        source.root_out_degree,
        source.num_samples,
        likelihood_order=source.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        source,
        block_state,
        _allocation_plan(block_state, (1, 0, 0, 0)),
        block_state.valid,
        shell_size=2,
        tail_K=jnp.asarray(0, dtype=jnp.int32),
    )
    refresh_rows = (
        depth.SEED_SOURCE_REFRESH_BATCHES
        * schedule.seed_reservoir_idx.shape[0]
    )
    assert schedule.continuation_parent_idx.shape[0] == (
        refresh_rows + schedule.seed_reservoir_idx.shape[0]
    )

    assert not bool(depth._seed_source_refresh_due(
        dataclasses.replace(
            source,
            num_samples=source.num_samples + refresh_rows - 1,
        ),
        schedule,
    ))
    assert bool(depth._seed_source_refresh_due(
        dataclasses.replace(
            source,
            num_samples=source.num_samples + refresh_rows,
        ),
        schedule,
    ))


def test_frozen_target_projects_by_successor_contour():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 3.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=5,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    schedule = depth._new_thread_schedule(
        state,
        block_state,
        _allocation_plan(block_state, (0, 0, 0, 0, 0)),
        block_state.valid,
        shell_size=2,
        tail_K=jnp.asarray(9, dtype=jnp.int32),
    )
    schedule = dataclasses.replace(
        schedule,
        target_K=jnp.asarray([10, 20, 0, 0, 0], dtype=jnp.int32),
    )
    refined = dataclasses.replace(
        block_state,
        log_L_blocks=jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0]),
        valid=jnp.ones((5,), dtype=bool),
    )

    np.testing.assert_array_equal(
        np.asarray(depth._project_allocation_target(schedule, refined)),
        np.asarray([10, 10, 20, 20, 9]),
    )


def test_depth_epoch_appends_without_reordering_coordinate_payload():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=3,
        initial_capacity=3,
        sampler=DeterministicSampler(),
        depth_condition=DepthCondition(),
    )
    state = ns.initialise(jax.random.PRNGKey(2))
    root_coordinates = np.asarray(state.samples.U_samples[:2]).copy()
    next_state = ns.run_single_iteration(
        state,
        depth_cond=DepthCondition(),
        key=jax.random.PRNGKey(3),
    )

    np.testing.assert_array_equal(
        np.asarray(next_state.samples.U_samples[:2]),
        root_coordinates,
    )
    assert int(next_state.num_samples) == 3
    sample_fields = {field.name for field in dataclasses.fields(next_state.samples)}
    for field_name in (
        "parent_idx",
        "requested_parent_idx",
        "requested_log_L_constraint",
        "seed_idx",
    ):
        assert field_name not in sample_fields
    assert (
        int(next_state.root_out_degree)
        + int(jnp.sum(next_state.samples.out_degree[:3]))
        == 3
    )


def test_depth_relevance_is_a_complete_prefix_when_remaining_mass_rises():
    valid = jnp.asarray([True, True, True, False])
    plan = AllocationPlan(
        target_K=jnp.zeros((4,), dtype=jnp.int32),
        current_K=jnp.zeros((4,), dtype=jnp.int32),
        unit_peak_utility=valid.astype(jnp.float64),
        log_L_blocks=jnp.log(jnp.asarray([1.0, 1.1, 100.0, 1.0])),
        valid=valid,
        volume_path=VolumePath(
            X_prev=jnp.asarray([1.0, 0.9, 0.8, 0.0]),
            X=jnp.asarray([0.9, 0.8, 0.7, 0.0]),
            shell_mass=jnp.asarray([0.1, 0.1, 0.1, 0.0]),
        ),
    )

    # The middle contour falls below the threshold, but the sharp likelihood
    # rise at the third contour makes the tail important again. Skipping the
    # middle block would leave an impossible hole in lineage coverage.
    relevant = depth._depth_relevant_blocks(
        plan,
        DepthCondition(dlogZ=jnp.asarray(0.85)),
    )
    np.testing.assert_array_equal(np.asarray(relevant), np.asarray(valid))


def test_likelihood_order_merges_only_integer_identities():
    log_likelihoods = jnp.asarray([2.0, 4.0, 3.0, 1.0, 5.0])
    order = initialise_likelihood_order(
        log_likelihoods,
        jnp.asarray(2, dtype=jnp.int32),
    )
    merged = order.insert(
        log_likelihoods,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(merged.sample_indices[:4]),
        np.asarray([3, 0, 2, 1]),
    )


def test_likelihood_order_rank_merge_matches_stable_reference():
    rng = np.random.default_rng(17)
    for old_count, batch_size, valid_count in (
            (1, 4, 4),
            (7, 5, 3),
            (19, 8, 7),
    ):
        capacity = old_count + batch_size + 3
        # Rounded values deliberately exercise equality and stable ordering.
        values = np.round(rng.normal(size=capacity), decimals=1)
        log_likelihoods = jnp.asarray(values)
        order = initialise_likelihood_order(
            log_likelihoods,
            jnp.asarray(old_count, dtype=jnp.int32),
        )
        valid = jnp.arange(batch_size) < valid_count
        merged = order.insert(
            log_likelihoods,
            jnp.asarray(old_count, dtype=jnp.int32),
            valid,
        )

        identities = np.arange(old_count + valid_count)
        expected = identities[
            np.argsort(values[identities], kind="stable")
        ]
        np.testing.assert_array_equal(
            np.asarray(merged.sample_indices[:expected.size]),
            expected,
        )
        np.testing.assert_array_equal(
            np.asarray(merged.sample_indices[expected.size:]),
            -1,
        )


def test_partial_batch_respects_non_multiple_max_samples():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=2,
        max_samples=3,
        initial_capacity=4,
        sampler=DeterministicSampler(),
        depth_condition=DepthCondition(),
    )
    state = dataclasses.replace(
        ns.initialise(jax.random.PRNGKey(8)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
    )
    next_state = ns.run_single_iteration(state)

    assert int(next_state.num_samples) == 3
    assert (
        int(next_state.root_out_degree)
        + int(jnp.sum(next_state.samples.out_degree[:3]))
        == 3
    )


def test_outer_target_uses_fixed_initial_degree_not_mutable_root_degree():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        delta_K=1,
        max_samples=20,
        initial_capacity=20,
        sampler=DeterministicSampler(),
        depth_condition=DepthCondition(),
    )
    state = dataclasses.replace(
        ns.initialise(jax.random.PRNGKey(81)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
    )

    next_state = ns.run_single_iteration(state)

    # Depth iteration two targets d_0 * Delta K * iteration = 4 at the root.
    # Once those threads are accepted, the mutable sentinel out-degree is
    # also four, but
    # it must not be fed back into the target and start an unbounded chase.
    assert int(next_state.root_out_degree) == 4


def test_resume_uses_stored_key_and_matches_uninterrupted_run():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=8,
        initial_capacity=4,
        sampler=DeterministicSampler(),
        depth_condition=DepthCondition(),
    )

    def goal_one(state):
        return int(state.goal_loop_iter) >= 1

    def goal_three(state):
        return int(state.goal_loop_iter) >= 3

    key = jax.random.PRNGKey(9)

    uninterrupted = ns.run_until_goal(goal_three, key=key)
    checkpoint = ns.run_until_goal(goal_one, key=key)
    resumed = ns.resume_until_goal(checkpoint, goal_three)

    assert jax.tree.structure(uninterrupted) == jax.tree.structure(resumed)
    leaves_a = jax.tree.leaves(uninterrupted)
    leaves_b = jax.tree.leaves(resumed)
    for left, right in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_python_goal_loop_reports_terminal_depth_budget_without_iteration():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=2,
        initial_capacity=2,
        sampler=DeterministicSampler(),
    )

    state = ns.run_until_goal(
        lambda state: int(state.goal_loop_iter) >= 10,
        depth_cond=DepthCondition(),
        key=jax.random.PRNGKey(19),
    )

    assert int(state.num_samples) == 2
    assert int(state.goal_loop_iter) == 0
    assert int(state.termination_reason) == depth.MAX_SAMPLES_REACHED
    assert not bool(state.needs_growth)
    assert not bool(state.depth_reached)


def test_filled_target_advances_allocation_without_exposing_user_goal():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        delta_K=1,
        max_samples=4,
        initial_capacity=4,
        sampler=DeterministicSampler(),
    )
    initial = ns.initialise(jax.random.PRNGKey(290))
    plateau_likelihood = initial.samples.log_likelihoods.at[:2].set(0.0)
    initial = dataclasses.replace(
        initial,
        samples=dataclasses.replace(
            initial.samples,
            log_likelihoods=plateau_likelihood,
        ),
        likelihood_order=initialise_likelihood_order(
            plateau_likelihood,
            initial.num_samples,
        ),
        log_L_supremum=jnp.asarray(0.0),
    )

    observed = []

    def goal(state):
        observed.append(int(state.allocation_loop_iter))
        return False

    # K=2 already fills the first uniform target. The terminal plateau still
    # fails the expected-depth cut, so progress requires the next allocation
    # target without pretending that a user-visible depth was completed.
    returned = ns.resume_until_goal(
        initial,
        goal,
        depth_cond=DepthCondition(dlogZ=jnp.asarray(0.0)),
    )

    assert observed == [0]
    assert int(returned.goal_loop_iter) == 0
    assert int(returned.allocation_loop_iter) == 1
    assert int(returned.termination_reason) == depth.MAX_SAMPLES_REACHED
    assert not bool(returned.depth_reached)


def test_sample_storage_modes_are_explicit_and_inspectable():
    finite_default = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        sampler=DeterministicSampler(),
    )
    assert finite_default.max_samples == 2 * core.SAMPLES_PER_ROOT
    assert finite_default.initial_capacity == 2 + core.INITIAL_BATCHES
    assert not finite_default.unlimited_samples

    finite_large = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=5000,
        sampler=DeterministicSampler(),
    )
    assert finite_large.max_samples == 5000
    assert finite_large.initial_capacity == 2 + core.INITIAL_BATCHES

    unlimited = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        unlimited_samples=True,
        sampler=DeterministicSampler(),
    )
    assert unlimited.max_samples is None
    assert unlimited.initial_capacity == 2 + core.INITIAL_BATCHES

    with pytest.raises(ValueError, match="conflicts"):
        NestedSampler(
            model=make_toy_model(),
            max_samples=100,
            unlimited_samples=True,
        )


def _assert_single_depth_outcome(state):
    outcomes = (
        int(state.termination_reason) != 0,
        bool(state.needs_growth),
        bool(state.depth_reached),
    )
    assert sum(outcomes) == 1


def test_compiled_depth_classifies_normal_growth_and_terminal_returns():
    common = {
        "model": make_toy_model(),
        "target_num_live_points": 2,
        "shell_size": 1,
        "delta_K": 1,
        "sampler": DeterministicSampler(),
    }
    normal_sampler = NestedSampler(
        max_samples=4,
        initial_capacity=4,
        **common,
    )
    normal_state = normal_sampler.initialise(jax.random.PRNGKey(31))
    _, normal_goal_key = jax.random.split(normal_state.random_key)
    normal = normal_sampler.run_single_iteration(
        normal_state,
        depth_cond=DepthCondition(dlogZ=jnp.asarray(1.1)),
    )
    _assert_single_depth_outcome(normal)
    assert bool(normal.depth_reached)
    np.testing.assert_array_equal(normal.random_key, normal_goal_key)

    growth_sampler = NestedSampler(
        unlimited_samples=True,
        initial_capacity=2,
        depth_condition=DepthCondition(),
        **common,
    )
    growth_state = dataclasses.replace(
        growth_sampler.initialise(jax.random.PRNGKey(32)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
    )
    growth_depth_key, growth_goal_key = jax.random.split(
        growth_state.random_key
    )
    growth = growth_sampler.run_single_iteration(
        growth_state,
        depth_cond=DepthCondition(dlogZ=jnp.asarray(0.5)),
    )
    _assert_single_depth_outcome(growth)
    assert bool(growth.needs_growth)
    assert int(growth.goal_loop_iter) == 1
    np.testing.assert_array_equal(growth.random_key, growth_depth_key)
    np.testing.assert_array_equal(growth.goal_key, growth_goal_key)

    terminal_sampler = NestedSampler(
        max_samples=2,
        initial_capacity=2,
        **common,
    )
    terminal_state = dataclasses.replace(
        terminal_sampler.initialise(jax.random.PRNGKey(33)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
    )
    terminal = terminal_sampler.run_single_iteration(terminal_state)
    _assert_single_depth_outcome(terminal)
    assert int(terminal.termination_reason) == depth.MAX_SAMPLES_REACHED
    assert int(terminal.goal_loop_iter) == 1


def test_unlimited_growth_matches_preallocated_scientific_continuation():
    common = {
        "model": make_toy_model(),
        "target_num_live_points": 2,
        "shell_size": 1,
        "delta_K": 1,
        "unlimited_samples": True,
        "sampler": DeterministicSampler(),
        "depth_condition": DepthCondition(),
    }
    tiny = NestedSampler(initial_capacity=2, **common)
    preallocated = NestedSampler(initial_capacity=32, **common)

    observed_goal_boundaries = []

    def goal(state):
        observed_goal_boundaries.append(bool(state.depth_reached))
        return int(state.goal_loop_iter) >= 2

    key = jax.random.PRNGKey(34)

    depth_cond = DepthCondition(dlogZ=jnp.asarray(0.5))
    grown = tiny.run_until_goal(goal, depth_cond=depth_cond, key=key)
    reference = preallocated.run_until_goal(
        goal,
        depth_cond=depth_cond,
        key=key,
    )

    # The tiny run crosses a physical growth boundary. The valid scientific
    # prefix and continuation state must remain independent of that
    # implementation-only pause.
    assert grown.samples.log_likelihoods.shape[0] > 2
    assert reference.samples.log_likelihoods.shape[0] == 32
    # Resizing is an implementation detail inside one depth epoch. A custom
    # scientific goal must only observe initial or completed depth boundaries.
    assert all(observed_goal_boundaries)
    assert int(grown.goal_loop_iter) == int(reference.goal_loop_iter) == 2
    assert int(grown.num_samples) == int(reference.num_samples)
    grown = grown.trim()
    reference = reference.trim()
    for left, right in zip(
        jax.tree.leaves(grown),
        jax.tree.leaves(reference),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_finite_capacity_terminates_below_and_exactly_at_hard_maximum():
    common = {
        "model": make_toy_model(),
        "target_num_live_points": 2,
        "shell_size": 2,
        "delta_K": 2,
        "sampler": DeterministicSampler(),
    }
    below = NestedSampler(
        initial_capacity=3,
        max_samples=3,
        **common,
    ).run_until_goal(
        lambda state: False,
        depth_cond=DepthCondition(),
        key=jax.random.PRNGKey(35),
    )
    assert int(below.num_samples) == 3
    assert below.samples.log_likelihoods.shape[0] == 3
    assert int(below.termination_reason) == depth.MAX_SAMPLES_REACHED
    _assert_single_depth_outcome(below)

    exact = NestedSampler(
        initial_capacity=3,
        max_samples=5,
        **common,
    ).run_until_goal(
        lambda state: False,
        depth_cond=DepthCondition(),
        key=jax.random.PRNGKey(36),
    )
    assert int(exact.num_samples) == 5
    assert exact.samples.log_likelihoods.shape[0] == 5
    assert int(exact.termination_reason) == depth.MAX_SAMPLES_REACHED
    _assert_single_depth_outcome(exact)
    assert (
        int(exact.to_result().termination_reason)
        == depth.MAX_SAMPLES_REACHED
    )


def test_state_checkpoint_round_trip_preserves_resume_key_and_order():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=4,
        sampler=DeterministicSampler(),
    )
    state = ns.initialise(jax.random.PRNGKey(10))
    restored = pickle.loads(pickle.dumps(state))

    np.testing.assert_array_equal(
        np.asarray(restored.random_key),
        np.asarray(state.random_key),
    )
    np.testing.assert_array_equal(
        np.asarray(restored.likelihood_order.sample_indices),
        np.asarray(state.likelihood_order.sample_indices),
    )


def test_ellipsoidal_state_survives_checkpoint_growth_and_resume():
    model = TwoDimensionalModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        direction=EllipsoidalDirection(
            num_components=1,
            min_effective_samples=3,
            num_iterations=3,
            population_size=12,
        ),
    )
    common = {
        "model": model,
        "root_allocation_degree": 6,
        "shell_size": 2,
        "delta_K": 2,
        "unlimited_samples": True,
        "sampler": sampler,
        "depth_condition": DepthCondition(),
    }
    small = NestedSampler(initial_capacity=6, **common)
    large = NestedSampler(initial_capacity=48, **common)

    def goal(state):
        return int(state.goal_loop_iter) >= 2

    depth_cond = DepthCondition()
    key = jax.random.PRNGKey(246)

    grown = small.run_until_goal(goal, depth_cond=depth_cond, key=key)
    reference = large.run_until_goal(goal, depth_cond=depth_cond, key=key)
    assert grown.samples.log_likelihoods.shape[0] > 6
    assert int(grown.sampler_data.num_updates) > 0
    assert bool(jnp.any(grown.sampler_data.valid))
    assert int(grown.sampler_data.num_directions) == (
        int(grown.num_samples) - 6
    ) * sampler.num_slices
    assert 0 <= int(grown.sampler_data.num_isotropic) <= int(
        grown.sampler_data.num_directions
    )

    grown = grown.trim()
    reference = reference.trim()
    for left, right in zip(
        jax.tree.leaves(grown.samples),
        jax.tree.leaves(reference.samples),
        strict=True,
    ):
        left = np.asarray(left)
        right = np.asarray(right)
        if np.issubdtype(left.dtype, np.inexact):
            # Reductions over different padded capacities may differ by the
            # final floating-point bit while representing the same immutable
            # fit and scientific continuation.
            np.testing.assert_allclose(left, right, rtol=2e-15, atol=0.0)
        else:
            np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(grown.random_key, reference.random_key)
    np.testing.assert_array_equal(grown.goal_key, reference.goal_key)
    np.testing.assert_allclose(
        np.sort(np.asarray(grown.samples.log_likelihoods)),
        np.sort(np.asarray(reference.samples.log_likelihoods)),
        rtol=2e-15,
        atol=0.0,
    )
    for left, right in zip(
        jax.tree.leaves(grown.sampler_data),
        jax.tree.leaves(reference.sampler_data),
        strict=True,
    ):
        left = np.asarray(left)
        right = np.asarray(right)
        if np.issubdtype(left.dtype, np.inexact):
            np.testing.assert_allclose(left, right, rtol=2e-15, atol=0.0)
        else:
            np.testing.assert_array_equal(left, right)

    restored = pickle.loads(pickle.dumps(grown))
    for left, right in zip(
        jax.tree.leaves(restored.sampler_data),
        jax.tree.leaves(grown.sampler_data),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def next_goal(state):
        return int(state.goal_loop_iter) >= 3

    # A checkpoint is useful only if the restored geometry drives the same
    # subsequent chains, not merely if its arrays survive serialization.
    continued = small.resume_until_goal(
        grown,
        next_goal,
        depth_cond=depth_cond,
    ).trim()
    resumed = small.resume_until_goal(
        restored,
        next_goal,
        depth_cond=depth_cond,
    ).trim()
    for left, right in zip(
        jax.tree.leaves(continued),
        jax.tree.leaves(resumed),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_public_scientific_data_objects_are_frozen_and_slotted():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=3,
        initial_capacity=3,
        sampler=DeterministicSampler(),
    )
    state = ns.initialise(jax.random.PRNGKey(4))
    direction = EllipsoidalDirection()
    sampler_data = empty_sampler_data(num_components=1, dimension=1)

    # NestedSampler is mutable configuration whose dependent defaults are
    # normalised with ordinary typed assignments during construction. The
    # scientific state it produces remains immutable.
    assert hasattr(type(ns), "__slots__")
    ns.store_phantom_samples = True
    assert ns.store_phantom_samples

    for value, field_name in (
        (state, "num_samples"),
        (state.samples, "out_degree"),
        (direction, "num_components"),
        (sampler_data, "centres"),
    ):
        assert hasattr(type(value), "__slots__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field_name, None)


def test_parallel_replacement_delegates_batching_without_sequential_map():
    depth_source = inspect.getsource(depth)
    sampler_source = inspect.getsource(constrained_sampler)
    assert "sample_request(" in depth_source
    assert "_continue_slice_chains(" in sampler_source
    assert "jax.lax.map(" not in depth_source
    assert "jax.lax.map(" not in sampler_source


def test_retained_phantoms_are_generated_chain_prefix():
    chain = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    retained = _take_phantom_prefix(chain, 2)
    np.testing.assert_array_equal(np.asarray(retained), [10.0, 20.0])


def test_nested_sampler_resolves_and_preserves_phantom_capacity():
    model = TwoDimensionalModel()
    default = NestedSampler(
        model=model,
        collect_phantom_samples=True,
    )
    bounded = NestedSampler(
        model=model,
        collect_phantom_samples=True,
        max_phantom_samples=5,
    )

    assert default.max_phantom_samples == 2
    assert default.sampler.num_phantom() == 2
    assert bounded.max_phantom_samples == 5
    assert bounded.sampler.num_phantom() == 5

    # NestedSampler owns the high-level D-sized default even when the caller
    # supplies an otherwise unbounded built-in slice sampler. Direct low-level
    # use remains capable of retaining every eligible transition.
    custom_unbounded = UniDimSliceSampler(
        model=model,
        num_slices=10,
        collect_phantom_samples=True,
    )
    assert custom_unbounded.num_phantom() == 9
    custom_default = NestedSampler(model=model, sampler=custom_unbounded)
    assert custom_default.max_phantom_samples == 2
    assert custom_default.sampler.num_phantom() == 2

    custom_explicit = NestedSampler(
        model=model,
        sampler=custom_unbounded,
        max_phantom_samples=np.int64(4),
    )
    assert custom_explicit.max_phantom_samples is int(
        custom_explicit.max_phantom_samples
    )
    assert custom_explicit.max_phantom_samples == 4
    assert custom_explicit.sampler.num_phantom() == 4

    sampler_capacity = dataclasses.replace(
        custom_unbounded,
        max_phantom_samples=5,
    )
    sampler_precedence = NestedSampler(model=model, sampler=sampler_capacity)
    assert sampler_precedence.max_phantom_samples == 5
    with pytest.raises(ValueError, match="disagrees"):
        NestedSampler(
            model=model,
            sampler=sampler_capacity,
            max_phantom_samples=4,
        )

    restored_sampler = pickle.loads(pickle.dumps(bounded))
    assert restored_sampler.max_phantom_samples == 5
    assert restored_sampler.sampler.num_phantom() == 5

    state = bounded.initialise(jax.random.PRNGKey(284))
    restored_state = pickle.loads(pickle.dumps(state))
    assert restored_state.samples.phantom_samples.log_L.shape[1] == 5

    with pytest.raises(ValueError, match="collect_phantom_samples"):
        NestedSampler(model=model, max_phantom_samples=1)
    with pytest.raises(ValueError, match="num_slices - 1"):
        NestedSampler(
            model=model,
            collect_phantom_samples=True,
            max_phantom_samples=10,
        )


def test_additional_retained_phantoms_leave_classic_run_invariant():
    model = TwoDimensionalModel()
    common = {
        "model": model,
        "collect_phantom_samples": True,
        "root_allocation_degree": 4,
        "shell_size": 2,
        "max_samples": 6,
        "initial_capacity": 6,
        "depth_condition": DepthCondition(),
    }
    short = NestedSampler(max_phantom_samples=1, **common)
    long = NestedSampler(max_phantom_samples=9, **common)
    key = jax.random.PRNGKey(1284)

    short_state = short.run(key)
    long_state = long.run(key)

    for short_value, long_value in (
        (short_state.num_samples, long_state.num_samples),
        (short_state.termination_reason, long_state.termination_reason),
        (
            short_state.samples.log_L_constraints,
            long_state.samples.log_L_constraints,
        ),
        (
            short_state.samples.log_likelihoods,
            long_state.samples.log_likelihoods,
        ),
        (short_state.samples.U_samples, long_state.samples.U_samples),
        (short_state.samples.out_degree, long_state.samples.out_degree),
        (
            short_state.samples.num_likelihood_evaluations,
            long_state.samples.num_likelihood_evaluations,
        ),
    ):
        np.testing.assert_array_equal(short_value, long_value)

    short_result = short_state.to_result().trim()
    long_result = long_state.to_result().trim()
    evidence_key = jax.random.PRNGKey(2284)
    short_evidence = short_result.sample_evidence_mc(
        num_samples=16,
        conditioning="classic",
        key=evidence_key,
    )
    long_evidence = long_result.sample_evidence_mc(
        num_samples=16,
        conditioning="classic",
        key=evidence_key,
    )
    np.testing.assert_array_equal(
        short_evidence.log_Z_samples,
        long_evidence.log_Z_samples,
    )
