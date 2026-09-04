"""Scientific and lifecycle contracts for opt-in phantom seed selection."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.algorithm import depth
from jaxns.checkpoint import CheckpointManager
from jaxns.constrained_sampler import AbstractSampler, ConstrainedSampleBatch
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.distributed_core import (
    DistributedNestedSampler,
    ReservationState,
    _prepare_task,
)
from jaxns.mixed_precision import mp_policy
from jaxns.samples import PhantomSamples
from jaxns.sampling.seeding import (
    PhantomSeedPool,
    phantom_seed_is_eligible,
)


def _assert_same_tree(left, right) -> None:
    """Compare complete immutable Pytree contents without host coercion gaps."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
        )


def _pool_with_active_and_staging() -> PhantomSeedPool:
    """Build both pool generations through the immutable public data API."""
    pool = PhantomSeedPool.empty(
        capacity=3,
        U_template=jnp.zeros((2,), dtype=jnp.float64),
    )
    pool = pool.stage(
        U_samples=jnp.asarray([[0.1, 0.2], [0.3, 0.4]]),
        log_L=jnp.asarray([2.0, 3.0]),
        log_L_birth=jnp.asarray([-jnp.inf, 0.0]),
        cluster_idx=jnp.asarray([4, 5], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.2, 0.8]),
        valid=jnp.asarray([True, True]),
    ).promote()
    return pool.stage(
        U_samples=jnp.asarray([[0.5, 0.6]]),
        log_L=jnp.asarray([4.0]),
        log_L_birth=jnp.asarray([1.0]),
        cluster_idx=jnp.asarray([6], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.6]),
        valid=jnp.asarray([True]),
    )


def _state_with_active_phantom(
        *,
        cluster_idx: int,
        U_sample: float,
        log_L: float,
        log_L_birth: float,
        num_phantom: int = 0,
):
    """Attach one active representative without exposing pool internals."""
    state = make_state(
        root_out_degree=3,
        log_likelihoods=(1.0, 2.0, 3.0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf),
        out_degree=(0, 0, 0),
        max_samples=12,
        num_phantom=num_phantom,
    )
    pool = PhantomSeedPool.empty(
        capacity=3,
        U_template=state.U_supremum,
    )
    pool = pool.stage(
        U_samples=jnp.asarray([U_sample]),
        log_L=jnp.asarray([log_L]),
        log_L_birth=jnp.asarray([log_L_birth]),
        cluster_idx=jnp.asarray([cluster_idx], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.75]),
        valid=jnp.asarray([True]),
    ).promote()
    return dataclasses.replace(state, phantom_seed_pool=pool)


def _schedule_root_increment(state, *, shell_size: int):
    """Materialise one ordinary additive root-population increment."""
    root_degree = int(state.root_out_degree)
    state = dataclasses.replace(
        state,
        allocation_loop_iter=jnp.asarray(1, dtype=mp_policy.count_dtype),
    )
    return depth._start_schedule_round(
        state,
        DepthCondition(),
        shell_size=shell_size,
        allocation_target="uniform",
        root_degree=root_degree,
        delta_K=root_degree,
    )


@pytest.mark.parametrize(
    "birth, likelihood, constraint, expected",
    [
        (-np.inf, 1.0, -np.inf, True),
        (0.0, 2.0, 0.0, True),
        (0.0, 2.0, 1.0, True),
        (1.0, 2.0, 0.5, False),
        (0.0, 1.0, 1.0, False),
        (0.0, 0.5, 1.0, False),
    ],
)
def test_phantom_seed_eligibility_matches_strict_interval_reference(
        birth,
        likelihood,
        constraint,
        expected,
):
    """Birth is inclusive while the requested death contour is strict."""
    actual = phantom_seed_is_eligible(
        jnp.asarray(birth),
        jnp.asarray(likelihood),
        jnp.asarray(constraint),
    )
    assert bool(actual) is expected


def test_phantom_seed_eligibility_vector_matches_numpy_reference():
    """The accelerated predicate is exactly the direct interval oracle."""
    birth = np.asarray([-np.inf, 0.0, 1.0, 2.0])
    likelihood = np.asarray([0.0, 2.0, 2.0, 4.0])
    constraint = np.asarray([-np.inf, 1.0, 2.0, 1.5])
    expected = (birth <= constraint) & (constraint < likelihood)

    actual = jax.jit(phantom_seed_is_eligible)(
        jnp.asarray(birth),
        jnp.asarray(likelihood),
        jnp.asarray(constraint),
    )

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_phantom_seeding_is_explicit_and_requires_retained_states():
    """Default scheduling stays classic-only and allocates no hidden pool."""
    model = make_toy_model()
    disabled = NestedSampler(
        model=model,
        root_allocation_degree=2,
        initial_capacity=2,
        collect_phantom_samples=True,
        max_phantom_samples=1,
    )
    assert disabled.phantom_seeding is False
    assert disabled.initialise(jax.random.PRNGKey(1)).phantom_seed_pool is None

    with pytest.raises(ValueError, match="fixed, fully valid phantom"):
        NestedSampler(
            model=model,
            root_allocation_degree=2,
            initial_capacity=2,
            phantom_seeding=True,
        )

    enabled = NestedSampler(
        model=model,
        root_allocation_degree=2,
        initial_capacity=2,
        collect_phantom_samples=True,
        max_phantom_samples=1,
        phantom_seeding=True,
    )
    state = enabled.initialise(jax.random.PRNGKey(1))
    assert enabled.phantom_seeding is True
    assert state.phantom_seed_pool is not None
    assert not bool(jnp.any(state.phantom_seed_pool.active.valid))
    assert not bool(jnp.any(state.phantom_seed_pool.staging.valid))


def test_empty_phantom_pool_preserves_classic_work_plan_exactly():
    """Opting in cannot perturb scheduling before a phantom is published."""
    classic = make_state(
        root_out_degree=3,
        log_likelihoods=(1.0, 2.0, 3.0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf),
        out_degree=(0, 0, 0),
        max_samples=12,
        num_phantom=1,
    )
    classic = _schedule_root_increment(classic, shell_size=3)
    enabled = dataclasses.replace(
        classic,
        phantom_seed_pool=PhantomSeedPool.empty(
            capacity=3,
            U_template=classic.U_supremum,
        ),
    )
    key = jax.random.PRNGKey(292)
    classic_schedule, classic_work = depth._plan_scheduled_work_batch(
        key,
        classic,
        classic.scheduler_data,
        jnp.asarray(3, dtype=mp_policy.index_dtype),
    )
    enabled_schedule, enabled_work = depth._plan_scheduled_work_batch(
        key,
        enabled,
        enabled.scheduler_data,
        jnp.asarray(3, dtype=mp_policy.index_dtype),
    )

    np.testing.assert_array_equal(classic_work.valid, enabled_work.valid)
    np.testing.assert_array_equal(
        classic_work.parent_idx,
        enabled_work.parent_idx,
    )
    np.testing.assert_array_equal(
        classic_work.log_L_constraint,
        enabled_work.log_L_constraint,
    )
    np.testing.assert_array_equal(
        classic_work.seed_idx,
        enabled_work.seed_idx,
    )
    np.testing.assert_array_equal(
        np.asarray(classic_schedule.thread_id),
        np.asarray(enabled_schedule.thread_id),
    )


def test_variable_valid_phantom_sampler_must_explicitly_support_seeding():
    """A static maximum alone cannot define unbiased reservoir admission."""

    class VariablePhantomSampler(AbstractSampler):
        def num_phantom(self) -> int:
            return 1

        def get_sample(
                self,
                key,
                log_L_constraint,
                seed_point,
                args=(),
                params=None,
        ):
            del key, log_L_constraint, seed_point, args, params
            raise AssertionError("Configuration validation must run first.")

    with pytest.raises(ValueError, match="fixed, fully valid phantom"):
        NestedSampler(
            model=make_toy_model(),
            root_allocation_degree=2,
            initial_capacity=2,
            sampler=VariablePhantomSampler(),
            phantom_seeding=True,
        )


def test_local_and_distributed_configuration_uses_root_width_pool():
    """Execution topology cannot change the configured reservoir capacity."""
    model = make_toy_model()
    local = NestedSampler(
        model=model,
        root_allocation_degree=2,
        shell_size=5,
        initial_capacity=5,
        collect_phantom_samples=True,
        max_phantom_samples=1,
        phantom_seeding=True,
    )
    distributed = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        initial_capacity=5,
        collect_phantom_samples=True,
        max_phantom_samples=1,
        phantom_seeding=True,
    )

    local_state = local.initialise(jax.random.PRNGKey(21))
    distributed_state = distributed._core.initialise(
        jax.random.PRNGKey(21)
    )

    assert local_state.phantom_seed_pool.active.valid.shape == (2,)
    assert distributed_state.phantom_seed_pool.active.valid.shape == (2,)


def test_phantom_pool_stage_is_value_independent_and_promotes_atomically():
    """Returned values cannot alter retained cluster identities or priority."""
    empty = PhantomSeedPool.empty(
        capacity=2,
        U_template=jnp.zeros((1,), dtype=jnp.float64),
    )
    common = {
        "U_samples": jnp.asarray([[0.1], [0.2], [0.3]]),
        "log_L_birth": jnp.asarray([-jnp.inf, 0.0, 1.0]),
        "cluster_idx": jnp.asarray(
            [10, 11, 12],
            dtype=mp_policy.index_dtype,
        ),
        "priority": jnp.asarray([0.1, 0.9, 0.5]),
        "valid": jnp.asarray([True, True, True]),
    }
    first = empty.stage(
        log_L=jnp.asarray([2.0, 20.0, 200.0]),
        **common,
    )
    permuted_values = empty.stage(
        log_L=jnp.asarray([200.0, 2.0, 20.0]),
        **common,
    )

    # Capacity keeps the same two priority-selected clusters even though the
    # returned likelihood ordering is reversed. Likelihood is payload, never
    # a reservoir replacement score.
    np.testing.assert_array_equal(
        np.sort(np.asarray(first.staging.cluster_idx[first.staging.valid])),
        np.asarray([11, 12]),
    )
    np.testing.assert_array_equal(
        first.staging.cluster_idx,
        permuted_values.staging.cluster_idx,
    )
    np.testing.assert_array_equal(
        first.staging.priority,
        permuted_values.staging.priority,
    )
    _assert_same_tree(first.active, empty.active)

    promoted = first.promote()
    np.testing.assert_array_equal(
        np.sort(np.asarray(promoted.active.cluster_idx)),
        np.sort(np.asarray(first.staging.cluster_idx)),
    )
    assert not bool(jnp.any(promoted.staging.valid))


def test_pool_membership_is_invariant_to_publication_batches():
    """The same completed clusters survive regardless of staging batches."""
    batched = PhantomSeedPool.empty(
        capacity=2,
        U_template=jnp.zeros((1,), dtype=jnp.float64),
    ).stage(
        U_samples=jnp.asarray([[0.1], [0.2]]),
        log_L=jnp.asarray([100.0, 200.0]),
        log_L_birth=jnp.asarray([-jnp.inf, -jnp.inf]),
        cluster_idx=jnp.asarray([1, 2], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.99, 0.98]),
        valid=jnp.asarray([True, True]),
    ).promote()
    batched = batched.stage(
        U_samples=jnp.asarray([[0.3]]),
        log_L=jnp.asarray([3.0]),
        log_L_birth=jnp.asarray([1.0]),
        cluster_idx=jnp.asarray([3], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.01]),
        valid=jnp.asarray([True]),
    ).promote()
    together = PhantomSeedPool.empty(
        capacity=2,
        U_template=jnp.zeros((1,), dtype=jnp.float64),
    ).stage(
        U_samples=jnp.asarray([[0.1], [0.2], [0.3]]),
        log_L=jnp.asarray([100.0, 200.0, 3.0]),
        log_L_birth=jnp.asarray([-jnp.inf, -jnp.inf, 1.0]),
        cluster_idx=jnp.asarray([1, 2, 3], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.99, 0.98, 0.01]),
        valid=jnp.asarray([True, True, True]),
    ).promote()

    np.testing.assert_array_equal(
        np.sort(np.asarray(batched.active.cluster_idx)),
        np.sort(np.asarray(together.active.cluster_idx)),
    )


def test_phantom_is_a_separate_cluster_distinct_seed_source():
    """A phantom source coexists with exact cluster-distinct classic seeds."""
    state = _state_with_active_phantom(
        cluster_idx=1,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=3)

    key = jax.random.split(jax.random.PRNGKey(7), 32)[0]
    _, work = depth._plan_scheduled_work_batch(
        key,
        state,
        state.scheduler_data,
        jnp.asarray(3, dtype=mp_policy.index_dtype),
    )
    assert len(set(np.asarray(work.seed_idx).tolist())) == 3
    pool_lane = np.flatnonzero(np.asarray(work.seed_pool_idx) >= 0)
    assert pool_lane.size == 1
    lane = pool_lane[0]
    assert int(work.seed_idx[lane]) == 1
    seed_points = depth._seed_points_for_work(state, work)
    assert float(seed_points.U0[lane]) == pytest.approx(0.77)
    assert float(seed_points.log_L0[lane]) == pytest.approx(10.0)


def test_phantom_pool_locator_uses_append_row_not_likelihood_rank():
    """A shuffled classic append order still materialises the selected pool row."""
    state = make_state(
        root_out_degree=3,
        log_likelihoods=(3.0, 1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf),
        out_degree=(0, 0, 0),
        max_samples=12,
    )
    pool = PhantomSeedPool.empty(
        capacity=3,
        U_template=state.U_supremum,
    ).stage(
        U_samples=jnp.asarray([0.77]),
        log_L=jnp.asarray([10.0]),
        log_L_birth=jnp.asarray([-jnp.inf]),
        cluster_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.75]),
        valid=jnp.asarray([True]),
    ).promote()
    state = dataclasses.replace(state, phantom_seed_pool=pool)
    state = _schedule_root_increment(state, shell_size=3)

    key = jax.random.split(jax.random.PRNGKey(71), 32)[0]
    _, work = depth._plan_scheduled_work_batch(
        key,
        state,
        state.scheduler_data,
        jnp.asarray(3, dtype=mp_policy.index_dtype),
    )
    lane = np.flatnonzero(np.asarray(work.seed_pool_idx) >= 0)
    assert lane.size == 1
    assert int(work.seed_idx[lane[0]]) == 0
    seed_points = depth._seed_points_for_work(state, work)
    assert float(seed_points.U0[lane[0]]) == pytest.approx(0.77)
    assert float(seed_points.log_L0[lane[0]]) == pytest.approx(10.0)


def test_fixed_source_choice_is_independent_of_pool_capacity():
    """Capacity changes coverage, not the conditional phantom source law."""
    uses = []
    for capacity in (1, 3):
        state = make_state(
            root_out_degree=3,
            log_likelihoods=(1.0, 2.0, 3.0),
            log_L_constraints=(-np.inf, -np.inf, -np.inf),
            out_degree=(0, 0, 0),
            max_samples=12,
        )
        pool = PhantomSeedPool.empty(
            capacity=capacity,
            U_template=state.U_supremum,
        ).stage(
            U_samples=jnp.asarray([0.77]),
            log_L=jnp.asarray([10.0]),
            log_L_birth=jnp.asarray([-jnp.inf]),
            cluster_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
            priority=jnp.asarray([0.75]),
            valid=jnp.asarray([True]),
        ).promote()
        state = dataclasses.replace(state, phantom_seed_pool=pool)
        state = _schedule_root_increment(state, shell_size=1)
        def select_one(key, current_state):
            _, _, use_phantom = depth._sample_phantom_stationary_seeds(
                key,
                current_state,
                current_state.scheduler_data,
                jnp.asarray([-jnp.inf]),
                jnp.asarray([True]),
                jnp.asarray([-1], dtype=mp_policy.index_dtype),
                jnp.asarray([-jnp.inf]),
                jnp.asarray([False]),
                jnp.asarray([-1], dtype=mp_policy.index_dtype),
            )
            return use_phantom[0]

        selected = jax.jit(jax.vmap(
            select_one,
            in_axes=(0, None),
        ))(
            jax.random.split(jax.random.PRNGKey(293), 128),
            state,
        )
        uses.append(np.asarray(selected))

    np.testing.assert_array_equal(uses[0], uses[1])
    assert 0.35 < float(np.mean(uses[0])) < 0.65


def test_distributed_request_materialises_the_same_phantom_seed():
    """Worker payloads use the coordinator's selected phantom source."""
    state = _state_with_active_phantom(
        cluster_idx=0,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=1)
    prepared = None
    for key in jax.random.split(jax.random.PRNGKey(72), 32):
        candidate = _prepare_task(
            dataclasses.replace(state, random_key=key),
            ReservationState.empty(
                state.samples.log_likelihoods.shape[0]
            ),
            None,
            jnp.asarray([-1], dtype=mp_policy.index_dtype),
            jnp.asarray([-jnp.inf]),
            jnp.asarray([False]),
            dispatch_width=1,
            max_threads=jnp.asarray(1, dtype=mp_policy.index_dtype),
            max_samples=12,
        )
        if int(candidate.work.seed_pool_idx[0]) >= 0:
            prepared = candidate
            break

    assert prepared is not None
    assert bool(prepared.has_work)
    assert int(prepared.work.seed_idx[0]) == 0
    assert int(prepared.work.seed_pool_idx[0]) >= 0
    assert float(prepared.request.seed_points.U0[0]) == pytest.approx(0.77)
    assert float(prepared.request.seed_points.log_L0[0]) == pytest.approx(10.0)


def test_same_contour_reservation_uses_cluster_across_seed_representations():
    """A pending cluster excludes both its classic and phantom identities."""
    state = _state_with_active_phantom(
        cluster_idx=1,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=2)

    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(8),
        state,
        state.scheduler_data,
        jnp.asarray(2, dtype=mp_policy.index_dtype),
        reserved_seed_idx=jnp.asarray(
            [1, -1],
            dtype=mp_policy.index_dtype,
        ),
        reserved_log_L_constraint=jnp.asarray([-jnp.inf, -jnp.inf]),
        reserved_valid=jnp.asarray([True, False]),
    )

    assert 1 not in set(np.asarray(work.seed_idx).tolist())
    assert len(set(np.asarray(work.seed_idx).tolist())) == 2


def test_parent_exclusion_overlapping_reservation_is_counted_once():
    """An overlap cannot make the last distinct cluster look exhausted."""
    state = _state_with_active_phantom(
        cluster_idx=0,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=2)
    schedule = dataclasses.replace(
        state.scheduler_data,
        parent_idx=jnp.asarray([0, 0], dtype=mp_policy.index_dtype),
        thread_id=jnp.asarray([20, 21], dtype=mp_policy.index_dtype),
        log_L_constraint=jnp.asarray([-jnp.inf, -jnp.inf]),
        terminal_log_L=jnp.asarray([3.0, 3.0]),
        new_start=jnp.asarray([False, False]),
        valid=jnp.asarray([True, True]),
        next_run=state.scheduler_data.num_runs,
        remaining_in_run=jnp.asarray(0, dtype=mp_policy.count_dtype),
        continuation_count=jnp.asarray(0, dtype=mp_policy.count_dtype),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)

    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(82),
        state,
        schedule,
        jnp.asarray(2, dtype=mp_policy.index_dtype),
        reserved_seed_idx=jnp.asarray(
            [0, -1],
            dtype=mp_policy.index_dtype,
        ),
        reserved_log_L_constraint=jnp.asarray([-jnp.inf, -jnp.inf]),
        reserved_valid=jnp.asarray([True, False]),
    )

    np.testing.assert_array_equal(
        np.sort(np.asarray(work.seed_idx)),
        np.asarray([1, 2]),
    )


def test_different_contours_do_not_globally_reserve_source_cluster():
    """One stationary cluster may seed distinct constrained-prior groups."""
    state = make_state(
        root_out_degree=1,
        log_likelihoods=(3.0,),
        log_L_constraints=(-np.inf,),
        out_degree=(0,),
        max_samples=8,
    )
    pool = PhantomSeedPool.empty(
        capacity=2,
        U_template=state.U_supremum,
    ).stage(
        U_samples=jnp.asarray([0.77]),
        log_L=jnp.asarray([4.0]),
        log_L_birth=jnp.asarray([-jnp.inf]),
        cluster_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.75]),
        valid=jnp.asarray([True]),
    ).promote()
    state = dataclasses.replace(state, phantom_seed_pool=pool)
    state = depth._start_schedule_round(
        dataclasses.replace(
            state,
            allocation_loop_iter=jnp.asarray(
                1,
                dtype=mp_policy.count_dtype,
            ),
        ),
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=1,
        delta_K=1,
    )
    schedule = state.scheduler_data
    schedule = dataclasses.replace(
        schedule,
        parent_idx=jnp.asarray([-1, -1], dtype=mp_policy.index_dtype),
        thread_id=jnp.asarray([50, 51], dtype=mp_policy.index_dtype),
        log_L_constraint=jnp.asarray([-jnp.inf, 1.0]),
        terminal_log_L=jnp.asarray([3.0, 3.0]),
        new_start=jnp.asarray([True, True]),
        valid=jnp.asarray([True, True]),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(0, dtype=mp_policy.count_dtype),
        continuation_count=jnp.asarray(0, dtype=mp_policy.count_dtype),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)

    seed_idx, pool_idx, use_phantom = (
        depth._sample_phantom_stationary_seeds(
            jax.random.PRNGKey(3),
            state,
            schedule,
            jnp.asarray([-jnp.inf, 1.0]),
            jnp.asarray([True, True]),
            jnp.asarray([-1, -1], dtype=mp_policy.index_dtype),
            jnp.asarray([-jnp.inf, -jnp.inf]),
            jnp.asarray([False, False]),
            jnp.asarray([-1, -1], dtype=mp_policy.index_dtype),
        )
    )

    np.testing.assert_array_equal(np.asarray(seed_idx), [0, 0])
    assert np.all(np.asarray(pool_idx) >= 0)
    assert np.all(np.asarray(use_phantom))


def test_current_thread_cluster_avoids_own_cluster_when_alternative_exists():
    """A continuation avoids its own cluster when another source is eligible."""
    state = _state_with_active_phantom(
        cluster_idx=1,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=1)
    schedule = state.scheduler_data
    schedule = dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx.at[0].set(1),
        thread_id=schedule.thread_id.at[0].set(40),
        log_L_constraint=schedule.log_L_constraint.at[0].set(2.0),
        terminal_log_L=schedule.terminal_log_L.at[0].set(3.0),
        new_start=schedule.new_start.at[0].set(False),
        valid=schedule.valid.at[0].set(True),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(0, dtype=mp_policy.count_dtype),
        continuation_count=jnp.asarray(0, dtype=mp_policy.count_dtype),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)

    selected = []
    for key in jax.random.split(jax.random.PRNGKey(9), 16):
        _, work = depth._plan_scheduled_work_batch(
            key,
            state,
            schedule,
            jnp.asarray(1, dtype=mp_policy.index_dtype),
        )
        selected.append(int(work.seed_idx[0]))

    # At L=2, cluster two's classic and cluster one's phantom both cross the
    # contour. The latter is forbidden because cluster one made this head.
    assert set(selected) == {2}


def test_lone_own_phantom_falls_back_to_exact_classic_seed():
    """A lane never extends its source cluster through an own phantom."""
    state = make_state(
        root_out_degree=1,
        log_likelihoods=(3.0,),
        log_L_constraints=(-np.inf,),
        out_degree=(0,),
        max_samples=8,
    )
    pool = PhantomSeedPool.empty(
        capacity=1,
        U_template=state.U_supremum,
    ).stage(
        U_samples=jnp.asarray([0.77]),
        log_L=jnp.asarray([4.0]),
        log_L_birth=jnp.asarray([-np.inf]),
        cluster_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.75]),
        valid=jnp.asarray([True]),
    ).promote()
    state = dataclasses.replace(state, phantom_seed_pool=pool)
    state = _schedule_root_increment(state, shell_size=1)
    schedule = state.scheduler_data
    schedule = dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx.at[0].set(0),
        thread_id=schedule.thread_id.at[0].set(40),
        log_L_constraint=schedule.log_L_constraint.at[0].set(3.0),
        terminal_log_L=schedule.terminal_log_L.at[0].set(5.0),
        new_start=schedule.new_start.at[0].set(False),
        valid=schedule.valid.at[0].set(True),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(
            0,
            dtype=schedule.remaining_in_run.dtype,
        ),
        continuation_count=jnp.asarray(
            0,
            dtype=schedule.continuation_count.dtype,
        ),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)

    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(91),
        state,
        schedule,
        jnp.asarray(1, dtype=mp_policy.index_dtype),
    )

    assert bool(work.valid[0])
    assert int(work.seed_idx[0]) == 0
    assert int(work.seed_pool_idx[0]) == -1
    assert int(work.parent_idx[0]) == -1
    assert np.isneginf(float(work.log_L_constraint[0]))
    seed_point = depth._seed_points_for_work(state, work)
    assert float(seed_point.log_L0[0]) > float(work.log_L_constraint[0])


def test_phantom_seed_preserves_requested_parent_out_degree():
    """The seed initializes MCMC; it never becomes the race-tree parent."""
    state = make_state(
        root_out_degree=1,
        log_likelihoods=(1.0, 2.0, 4.0),
        log_L_constraints=(-np.inf, 1.0, 3.0),
        out_degree=(1, 1, 0),
        max_samples=12,
        num_phantom=1,
    )
    pool = PhantomSeedPool.empty(
        capacity=3,
        U_template=state.U_supremum,
    ).stage(
        U_samples=jnp.asarray([0.77]),
        log_L=jnp.asarray([10.0]),
        log_L_birth=jnp.asarray([-jnp.inf]),
        cluster_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.75]),
        valid=jnp.asarray([True]),
    ).promote()
    state = dataclasses.replace(state, phantom_seed_pool=pool)
    state = _schedule_root_increment(state, shell_size=1)
    schedule = state.scheduler_data
    schedule = dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx.at[0].set(1),
        thread_id=schedule.thread_id.at[0].set(41),
        log_L_constraint=schedule.log_L_constraint.at[0].set(2.0),
        terminal_log_L=schedule.terminal_log_L.at[0].set(3.0),
        new_start=schedule.new_start.at[0].set(True),
        valid=schedule.valid.at[0].set(True),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(0, dtype=mp_policy.count_dtype),
        continuation_count=jnp.asarray(0, dtype=mp_policy.count_dtype),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)
    work = None
    for key in jax.random.split(jax.random.PRNGKey(10), 32):
        _, candidate = depth._plan_scheduled_work_batch(
            key,
            state,
            schedule,
            jnp.asarray(1, dtype=mp_policy.index_dtype),
        )
        if int(candidate.seed_pool_idx[0]) >= 0:
            work = candidate
            break
    assert work is not None
    assert int(work.seed_idx[0]) == 0
    assert int(work.seed_pool_idx[0]) >= 0
    assert int(work.parent_idx[0]) == 1
    assert float(work.log_L_constraint[0]) == 2.0

    batch = ConstrainedSampleBatch(
        U_samples=jnp.asarray([0.8]),
        log_likelihoods=jnp.asarray([3.5]),
        num_likelihood_evaluations=jnp.asarray(
            [1],
            dtype=mp_policy.count_dtype,
        ),
        phantom_samples=PhantomSamples(
            U_samples=jnp.asarray([[0.81]]),
            valid_mask=jnp.asarray([[True]]),
            log_L=jnp.asarray([[3.25]]),
        ),
        num_directions=jnp.asarray([1], dtype=mp_policy.count_dtype),
        num_isotropic=jnp.asarray([1], dtype=mp_policy.count_dtype),
    )
    accepted = depth._accept_work_batch(state, work, batch)

    assert int(accepted.samples.out_degree[0]) == 1
    assert int(accepted.samples.out_degree[1]) == 2


def test_reparented_phantom_birth_uses_effective_sampling_contour():
    """Staged provenance records the contour actually used by the chain."""
    state = make_state(
        root_out_degree=1,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, 1.0),
        out_degree=(1, 0),
        max_samples=8,
        num_phantom=1,
    )
    state = dataclasses.replace(
        state,
        phantom_seed_pool=PhantomSeedPool.empty(
            capacity=3,
            U_template=state.U_supremum,
        ),
    )
    state = _schedule_root_increment(state, shell_size=1)
    schedule = state.scheduler_data
    schedule = dataclasses.replace(
        schedule,
        parent_idx=schedule.parent_idx.at[0].set(1),
        thread_id=schedule.thread_id.at[0].set(42),
        log_L_constraint=schedule.log_L_constraint.at[0].set(2.0),
        terminal_log_L=schedule.terminal_log_L.at[0].set(4.0),
        new_start=schedule.new_start.at[0].set(True),
        valid=schedule.valid.at[0].set(True),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(
            0,
            dtype=schedule.remaining_in_run.dtype,
        ),
        continuation_count=jnp.asarray(
            0,
            dtype=schedule.continuation_count.dtype,
        ),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)

    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(12),
        state,
        schedule,
        jnp.asarray(1, dtype=mp_policy.index_dtype),
    )
    assert float(schedule.log_L_constraint[0]) == 2.0
    assert float(work.log_L_constraint[0]) == 1.0
    assert int(work.parent_idx[0]) == 0

    batch = ConstrainedSampleBatch(
        U_samples=jnp.asarray([0.8]),
        log_likelihoods=jnp.asarray([3.5]),
        num_likelihood_evaluations=jnp.asarray(
            [1],
            dtype=mp_policy.count_dtype,
        ),
        phantom_samples=PhantomSamples(
            U_samples=jnp.asarray([[0.81]]),
            valid_mask=jnp.asarray([[True]]),
            log_L=jnp.asarray([[2.5]]),
        ),
        num_directions=jnp.asarray([1], dtype=mp_policy.count_dtype),
        num_isotropic=jnp.asarray([1], dtype=mp_policy.count_dtype),
    )
    accepted = depth._accept_work_batch(state, work, batch)
    staging = accepted.phantom_seed_pool.staging
    staged = np.flatnonzero(np.asarray(staging.valid))
    assert staged.size == 1
    assert float(staging.log_L_birth[staged[0]]) == 1.0


def test_phantom_representative_offset_is_independent_of_returned_values():
    """Likelihood permutations cannot select a more favourable chain state."""
    state = _state_with_active_phantom(
        cluster_idx=0,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
        num_phantom=3,
    )
    state = _schedule_root_increment(state, shell_size=1)
    _, work = depth._plan_scheduled_work_batch(
        jax.random.PRNGKey(11),
        state,
        state.scheduler_data,
        jnp.asarray(1, dtype=mp_policy.index_dtype),
    )

    def accept(log_L):
        batch = ConstrainedSampleBatch(
            U_samples=jnp.asarray([0.8]),
            log_likelihoods=jnp.asarray([3.5]),
            num_likelihood_evaluations=jnp.asarray(
                [1],
                dtype=mp_policy.count_dtype,
            ),
            phantom_samples=PhantomSamples(
                # Coordinates encode the generated offset so the test detects
                # a value-dependent switch to another phantom in the cluster.
                U_samples=jnp.asarray([[0.11, 0.22, 0.33]]),
                valid_mask=jnp.asarray([[True, True, True]]),
                log_L=jnp.asarray([log_L]),
            ),
            num_directions=jnp.asarray([1], dtype=mp_policy.count_dtype),
            num_isotropic=jnp.asarray([1], dtype=mp_policy.count_dtype),
        )
        return depth._accept_work_batch(state, work, batch)

    first = accept([4.0, 5.0, 6.0]).phantom_seed_pool.staging
    permuted = accept([6.0, 4.0, 5.0]).phantom_seed_pool.staging
    np.testing.assert_array_equal(first.cluster_idx, permuted.cluster_idx)
    np.testing.assert_array_equal(first.priority, permuted.priority)
    np.testing.assert_array_equal(first.U_samples, permuted.U_samples)


def test_phantom_candidates_publish_after_generation_completes():
    """A staged cohort becomes active only at its publication boundary."""
    state = _state_with_active_phantom(
        cluster_idx=0,
        U_sample=0.77,
        log_L=10.0,
        log_L_birth=-np.inf,
    )
    state = _schedule_root_increment(state, shell_size=1)
    original_active = state.phantom_seed_pool.active
    pool = state.phantom_seed_pool.stage(
        U_samples=jnp.asarray([0.66]),
        log_L=jnp.asarray([9.0]),
        log_L_birth=jnp.asarray([-np.inf]),
        cluster_idx=jnp.asarray([7], dtype=mp_policy.index_dtype),
        priority=jnp.asarray([0.95]),
        valid=jnp.asarray([True]),
    )
    state = dataclasses.replace(state, phantom_seed_pool=pool)
    _assert_same_tree(state.phantom_seed_pool.active, original_active)

    published = depth._publish_seed_source(state)
    np.testing.assert_array_equal(
        np.sort(np.asarray(
            published.phantom_seed_pool.active.cluster_idx[
                published.phantom_seed_pool.active.valid
            ]
        )),
        np.asarray([0, 7]),
    )
    assert not bool(jnp.any(published.phantom_seed_pool.staging.valid))

    drained_schedule = dataclasses.replace(
        published.scheduler_data,
        active=jnp.asarray(False),
        valid=jnp.zeros_like(published.scheduler_data.valid),
        next_run=published.scheduler_data.num_runs,
        remaining_in_run=jnp.asarray(0, dtype=mp_policy.count_dtype),
        continuation_count=jnp.asarray(0, dtype=mp_policy.count_dtype),
    )
    published = dataclasses.replace(
        published,
        scheduler_data=drained_schedule,
    )
    continued = depth._continue_schedule_round(
        published,
        drained_schedule,
        DepthCondition(),
        shell_size=1,
    )

    np.testing.assert_array_equal(
        np.sort(np.asarray(
            continued.phantom_seed_pool.active.cluster_idx[
                continued.phantom_seed_pool.active.valid
            ]
        )),
        np.asarray([0, 7]),
    )
    assert not bool(jnp.any(continued.phantom_seed_pool.staging.valid))


def test_phantom_seed_pool_survives_resize_trim_and_checkpoint(tmp_path):
    """Physical sample capacity cannot resize or perturb the bounded pool."""
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0),
        out_degree=(0, 0),
        max_samples=4,
    )
    pool = _pool_with_active_and_staging()
    state = dataclasses.replace(state, phantom_seed_pool=pool)

    resized = state.resize(16)
    trimmed = resized.trim()
    _assert_same_tree(resized.phantom_seed_pool, pool)
    _assert_same_tree(trimmed.phantom_seed_pool, pool)
    assert resized.phantom_seed_pool.active.valid.shape == (3,)

    with CheckpointManager(tmp_path) as manager:
        manager.save(resized)
    with CheckpointManager(tmp_path) as manager:
        restored = manager.load()
    _assert_same_tree(restored.phantom_seed_pool, pool)


def test_state_merge_clears_phantom_seed_pool():
    """Merge does not invent cluster-ID remapping across append namespaces."""
    left = make_state(
        root_out_degree=1,
        log_likelihoods=(0.0,),
        out_degree=(0,),
        max_samples=1,
    )
    right = make_state(
        root_out_degree=1,
        log_likelihoods=(1.0,),
        out_degree=(0,),
        max_samples=1,
    )
    pool = _pool_with_active_and_staging()
    left = dataclasses.replace(left, phantom_seed_pool=pool)
    right = dataclasses.replace(right, phantom_seed_pool=pool)

    merged = left.merge(right)

    assert merged.phantom_seed_pool is None
