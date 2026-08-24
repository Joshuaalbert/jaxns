"""Focused contracts for the compiled scheduling core."""

import dataclasses
import inspect
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns import core
from jaxns.allocation import (
    AllocationPlan,
    VolumePath,
    closest_seedable_parent_block_python,
    stationary_seed_indices_python,
)
from jaxns.constrained_sampler import AbstractSampler, _take_phantom_prefix
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import (
    build_block_state,
    initialise_likelihood_order,
)
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.termination_condition import TerminationCondition
from tests.core_fixtures import make_state
from tests.distributed_support import make_toy_model


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


def test_stationary_seed_mask_uses_generation_interval_not_suffix():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 3.0, 4.0, 0.5),
        log_L_constraints=(-np.inf, 0.0, 2.0, -np.inf),
        out_degree=(1, 1, 0, 0),
        max_samples=4,
    )
    stationary = core._stationary_seed_mask(
        state,
        jnp.asarray(1.0),
        jnp.asarray(False),
    )
    np.testing.assert_array_equal(
        np.asarray(stationary),
        np.asarray([False, True, False, False]),
    )
    np.testing.assert_array_equal(
        stationary_seed_indices_python(
            state.samples,
            int(state.num_samples),
            1.0,
            from_root=False,
        ),
        np.asarray([1]),
    )

    root_stationary = core._stationary_seed_mask(
        state,
        jnp.asarray(-jnp.inf),
        jnp.asarray(True),
    )
    np.testing.assert_array_equal(
        np.asarray(root_stationary),
        np.asarray([True, False, False, True]),
    )


def test_missing_stationary_seed_reparents_to_closest_shallower_contour():
    state = make_state(
        root_out_degree=1,
        log_likelihoods=(1.0, 2.0, 4.0),
        log_L_constraints=(-np.inf, 1.0, 3.0),
        out_degree=(1, 1, 0),
        max_samples=3,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
    )

    # lambda=2 has no sample whose generation interval contains it. The
    # closest shallower lambda=1 contour has sample 1 as an exact stationary
    # seed and must be the effective parent block.
    effective = core._closest_seedable_parent_block(
        state,
        block_state,
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert int(effective) == 0
    assert (
        closest_seedable_parent_block_python(state, block_state, 1)
        == int(effective)
    )


def test_depth_epoch_appends_without_reordering_coordinate_payload():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=3,
        initial_capacity=3,
        sampler=DeterministicSampler(),
        termination_condition=TerminationCondition(max_samples=3),
    )
    state = ns.initialise(jax.random.PRNGKey(2))
    root_coordinates = np.asarray(state.samples.U_samples[:2]).copy()
    next_state = ns.run_single_iteration(
        state,
        depth_cond=TerminationCondition(max_samples=3),
        key=jax.random.PRNGKey(3),
    )

    np.testing.assert_array_equal(
        np.asarray(next_state.samples.U_samples[:2]),
        root_coordinates,
    )
    assert int(next_state.num_samples) == 3
    for field_name in (
        "parent_idx",
        "requested_parent_idx",
        "requested_log_L_constraint",
        "seed_idx",
    ):
        assert not hasattr(next_state.samples, field_name)
    assert (
        int(next_state.root_out_degree)
        + int(jnp.sum(next_state.samples.out_degree[:3]))
        == 3
    )


def test_scheduler_marks_maximal_thread_prefix_and_stratifies_seeds():
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
    )
    plan = AllocationPlan(
        target_K=jnp.asarray([2, 0, 0, 0], dtype=jnp.int32),
        current_K=jnp.zeros((4,), dtype=jnp.int32),
        unit_peak_utility=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        log_L_blocks=jnp.asarray([1.0, jnp.inf, jnp.inf, jnp.inf]),
        valid=jnp.asarray([True, False, False, False]),
        volume_path=VolumePath(
            X_prev=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            X=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            shell_mass=jnp.zeros((4,)),
        ),
    )
    work = core._plan_work_batch(
        jax.random.PRNGKey(7),
        state,
        block_state,
        plan,
        relevant=plan.valid,
        shell_size=4,
    )

    np.testing.assert_array_equal(
        np.asarray(work.valid),
        np.asarray([True, True, False, False]),
    )
    assert int(work.seed_idx[0]) != int(work.seed_idx[1])


def test_stratified_seed_rank_is_uniform_for_each_stationary_set():
    # Different contour lanes generally have different eligible samples. A
    # shared rotation may couple their choices, but each marginal must remain
    # uniform or the scheduler would distort relative mode weights.
    unit_draws = (jnp.arange(600, dtype=jnp.float64) + 0.5) / 600.0
    for mask in (
        jnp.asarray([True, False, True, True]),
        jnp.asarray([False, True, False, True]),
    ):
        selected = jax.vmap(
            lambda draw, stationary=mask: core._uniform_ranked_masked(
                draw,
                stationary,
            )
        )(unit_draws)
        counts = np.bincount(np.asarray(selected), minlength=mask.shape[0])
        expected = 600 // int(jnp.sum(mask))
        np.testing.assert_array_equal(
            counts[np.asarray(mask)],
            np.full(int(jnp.sum(mask)), expected),
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
    relevant = core._depth_relevant_blocks(
        plan,
        TerminationCondition(dlogZ=jnp.asarray(0.85)),
    )
    np.testing.assert_array_equal(np.asarray(relevant), np.asarray(valid))


def test_scheduler_starts_one_thread_for_one_gap_rise():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0, 3.0),
        log_L_constraints=(-np.inf, 1.0, 2.0),
        out_degree=(1, 1, 0),
        max_samples=6,
    )
    block_state = build_block_state(
        state.samples,
        state.root_out_degree,
        state.num_samples,
    )
    target = jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.int32)
    plan = AllocationPlan(
        target_K=target,
        current_K=jnp.zeros_like(target),
        unit_peak_utility=block_state.valid.astype(jnp.float64),
        log_L_blocks=block_state.log_L_blocks,
        valid=block_state.valid,
        volume_path=VolumePath(
            X_prev=jnp.ones(target.shape),
            X=jnp.ones(target.shape),
            shell_mass=jnp.zeros(target.shape),
        ),
    )

    work = core._plan_work_batch(
        jax.random.PRNGKey(11),
        state,
        block_state,
        plan,
        relevant=block_state.valid,
        shell_size=4,
    )

    np.testing.assert_array_equal(
        np.asarray(work.valid),
        np.asarray([True, False, False, False]),
    )
    assert float(work.log_L_constraint[0]) < 3.0


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
        termination_condition=TerminationCondition(max_samples=3),
    )
    state = dataclasses.replace(
        ns.initialise(jax.random.PRNGKey(8)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
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
        termination_condition=TerminationCondition(max_samples=20),
    )
    state = dataclasses.replace(
        ns.initialise(jax.random.PRNGKey(81)),
        goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
    )

    next_state = ns.run_single_iteration(state)

    # Iteration one targets d_0 + Delta K = 3 at the root. Once that one
    # thread is accepted, the mutable sentinel out-degree is also three, but
    # it must not be fed back into the target and start an unbounded chase.
    assert int(next_state.root_out_degree) == 3


def test_resume_uses_stored_key_and_matches_uninterrupted_run():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=8,
        initial_capacity=4,
        sampler=DeterministicSampler(),
        termination_condition=TerminationCondition(max_samples=8),
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


def test_python_goal_loop_returns_after_depth_cannot_make_progress():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=8,
        initial_capacity=4,
        sampler=DeterministicSampler(),
    )

    state = ns.run_until_goal(
        lambda state: int(state.goal_loop_iter) >= 10,
        depth_cond=TerminationCondition(max_samples=2),
        key=jax.random.PRNGKey(19),
    )

    assert int(state.num_samples) == 2
    assert int(state.goal_loop_iter) == 2


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


def test_public_data_objects_are_frozen_and_slotted():
    ns = NestedSampler(
        model=make_toy_model(),
        target_num_live_points=2,
        shell_size=1,
        max_samples=3,
        initial_capacity=3,
        sampler=DeterministicSampler(),
    )
    state = ns.initialise(jax.random.PRNGKey(4))

    for value, field_name in (
        (ns, "max_samples"),
        (state, "num_samples"),
        (state.samples, "out_degree"),
    ):
        assert hasattr(type(value), "__slots__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field_name, None)


def test_parallel_replacement_uses_full_vmap_not_sequential_lax_map():
    source = inspect.getsource(core)
    assert "jax.vmap(sample_one)" in source
    assert "jax.lax.map(" not in source


def test_retained_phantoms_are_generated_chain_prefix():
    chain = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    retained = _take_phantom_prefix(chain, 2)
    np.testing.assert_array_equal(np.asarray(retained), [10.0, 20.0])
