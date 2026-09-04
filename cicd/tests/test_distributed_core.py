"""Scientific contracts at the asynchronous constrained-sampling boundary."""

import dataclasses
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.algorithm.depth import (
    SEED_SOURCE_REFRESH_WINDOWS,
    CoreWorkBatch,
    _insert_seed_reservation,
    _seed_reservation_contains,
    _start_schedule_round,
)
from jaxns.constrained_sampler import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    LikelihoodEvaluation,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.distributed_core import (
    DistributedNestedSampler,
    DistributedRunError,
    DistributedState,
    PendingTask,
    ReservationState,
    _accept_task,
    _change_reservations,
    _depth_status,
    _planning_state,
)
from jaxns.runtime.client import RuntimeUnavailableError
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.seeding import PhantomSeedPool


def _local_checkpoint(
        runner: DistributedNestedSampler,
        key,
) -> DistributedState:
    """Build planning state without requiring the process-runtime boundary."""
    state = runner._core.initialise(key)
    return DistributedState(
        state=state,
        reservations=ReservationState.empty(
            state.samples.log_likelihoods.shape[0]
        ),
        pending=(),
        next_task_id=0,
        session_id="unit-test",
        depth_active=False,
        goal_key=state.goal_key,
    )


def _active_phantom_pool(state) -> PhantomSeedPool:
    """Build two eligible source clusters through the immutable pool API."""
    pool = PhantomSeedPool.empty(
        capacity=2,
        U_template=state.U_supremum,
    ).assign_staging_slots(
        log_L_slot=jnp.asarray([-jnp.inf, -jnp.inf]),
        valid=jnp.asarray([True, True]),
    )
    return pool.stage(
        # Distinct coordinates make the selected source observable at the
        # worker boundary; high likelihoods keep both eligible for this
        # scheduling-only fixture.
        U_samples=jnp.asarray([0.123, 0.876]),
        log_L=jnp.asarray([10.0, 10.0]),
        log_L_birth=jnp.asarray([-jnp.inf, -jnp.inf]),
        cluster_idx=jnp.asarray([0, 1], dtype=jnp.int32),
        priority=jnp.asarray([0.8, 0.7]),
        valid=jnp.asarray([True, True]),
        slot_idx=jnp.asarray([0, 1], dtype=jnp.int32),
        log_L_slot=jnp.asarray([-jnp.inf, -jnp.inf]),
    ).promote()


def test_distributed_directions_change_only_at_drained_boundaries():
    """Direction geometry is scientific state, never in-flight task state."""
    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=4,
        initial_capacity=8,
    )
    assert runner.delta_K == 4
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(246))

    fitted = checkpoint.fit_gmm_directions(
        num_iterations=3,
        iso_prob=0.01,
    )
    assert fitted.state.sampler_data.centres.shape == (1, 1)
    assert bool(fitted.state.sampler_data.enabled)
    isotropic = fitted.iso_directions()
    assert not bool(isotropic.state.sampler_data.enabled)
    restored = isotropic.gmm_directions()
    assert bool(restored.state.sampler_data.enabled)

    active = dataclasses.replace(restored, depth_active=True)
    with pytest.raises(RuntimeError, match="drained distributed state"):
        active.fit_gmm_directions()
    with pytest.raises(RuntimeError, match="drained distributed state"):
        active.iso_directions()
    with pytest.raises(RuntimeError, match="drained distributed state"):
        active.gmm_directions()


def test_distributed_initialisation_dispatches_every_likelihood():
    class Client:
        evaluations = 0
        reject_first = True

        def __init__(self):
            self.results = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback

        def register(self, session_id, session):
            del session_id, session
            return (2,)

        def evaluate_many(self, session_id, tasks):
            del session_id
            for task_id, request in tasks:
                Client.evaluations += 1
                if Client.reject_first:
                    likelihood = jnp.full((1,), -jnp.inf)
                    Client.reject_first = False
                else:
                    likelihood = -jnp.square(request.U_samples - 0.25)
                self.results.append((
                    task_id,
                    LikelihoodEvaluation(log_likelihoods=likelihood),
                ))

        def receive_group(self, session_id, timeout_s):
            del session_id, timeout_s
            results = tuple(self.results)
            self.results.clear()
            return results

        def acknowledge(self, session_id, task_id):
            del session_id, task_id

        def release(self, session_id):
            del session_id

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=3,
        initial_capacity=6,
    )

    checkpoint = runner._initialise_connected(
        Client(),
        "initialisation-test",
        jax.random.PRNGKey(41),
    )

    assert Client.evaluations == 4
    assert int(checkpoint.state.num_samples) == 3
    assert checkpoint.next_task_id == 4
    np.testing.assert_array_equal(
        np.asarray(checkpoint.state.samples.num_likelihood_evaluations[:3]),
        np.asarray([2, 1, 1], dtype=np.int32),
    )
    assert bool(jnp.all(
        checkpoint.state.samples.log_likelihoods[:3] > -jnp.inf
    ))


def test_distributed_completion_order_preserves_scientific_state():
    """Worker latency cannot change phantom seeds or the committed race."""
    model = make_toy_model()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        collect_phantom_samples=True,
        max_phantom_samples=1,
    )
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=2,
        max_samples=8,
        initial_capacity=8,
        sampler=sampler,
        phantom_seeding=True,
    )

    class ArrivalOrderClient:
        def __init__(self, reverse):
            self.reverse = reverse
            self.results = []
            self.submitted = []
            self.completion_order = []

        def capacity(self, session_id, timeout_s):
            del session_id, timeout_s
            return 2

        def submit_many(self, session_id, tasks):
            del session_id
            self.submitted.extend(tasks)
            self.results.extend(
                (task_id, sample_request(sampler, request))
                for task_id, request in tasks
            )

        def receive_group(self, session_id, timeout_s):
            del session_id, timeout_s
            # Each result models one scalar worker completion group. Selecting
            # opposite task-ID extremes creates the same work with different
            # observable arrival latency.
            select = max if self.reverse else min
            result_idx = select(
                range(len(self.results)),
                key=lambda idx: self.results[idx][0],
            )
            result = self.results.pop(result_idx)
            self.completion_order.append(result[0])
            return (result,)

        def acknowledge(self, session_id, task_id):
            del session_id, task_id

    def goal(state):
        return int(state.goal_loop_iter) >= 1

    def run(reverse):
        client = ArrivalOrderClient(reverse)
        checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(294))
        pool = _active_phantom_pool(checkpoint.state)
        # Exercise k=1 so D_g^k=d_0+Delta K k exposes two simultaneous
        # root starts whose completion order can actually differ.
        checkpoint = dataclasses.replace(
            checkpoint,
            state=dataclasses.replace(
                checkpoint.state,
                allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
                # This key makes the initial wave use an active phantom, so
                # reversed arrivals exercise the new scientific seed source.
                random_key=jnp.asarray(
                    [172216686, 3883864598],
                    dtype=jnp.uint32,
                ),
                phantom_seed_pool=pool,
            ),
        )
        completed = runner._run_connected(
            client,
            checkpoint,
            goal,
            DepthCondition(),
            checkpoint_manager=None,
        )
        return completed.state.trim(), client

    fifo_state, fifo_client = run(reverse=False)
    reversed_state, reversed_client = run(reverse=True)

    assert fifo_client.completion_order[:2] == [0, 1]
    assert reversed_client.completion_order[:2] == [1, 0]
    assert fifo_client.completion_order != reversed_client.completion_order

    # This makes the completion-order assertion exercise the phantom path,
    # rather than proving only the pre-existing classic scheduler behavior.
    assert any(
        any(
            float(request.seed_points.U0[0]) == pytest.approx(phantom_U)
            for phantom_U in (0.123, 0.876)
        )
        for _, request in fifo_client.submitted[:2]
    )

    # Both executions must have planned the same requests before their
    # completion latency can be treated as the only independent variable.
    assert [task_id for task_id, _ in fifo_client.submitted] == [
        task_id for task_id, _ in reversed_client.submitted
    ]
    for (_, fifo_request), (_, reversed_request) in zip(
        fifo_client.submitted,
        reversed_client.submitted,
        strict=True,
    ):
        assert jax.tree.structure(fifo_request) == jax.tree.structure(
            reversed_request
        )
        for fifo_leaf, reversed_leaf in zip(
            jax.tree.leaves(fifo_request),
            jax.tree.leaves(reversed_request),
            strict=True,
        ):
            np.testing.assert_array_equal(reversed_leaf, fifo_leaf)

    assert jax.tree.structure(fifo_state) == jax.tree.structure(
        reversed_state
    )
    for fifo_leaf, reversed_leaf in zip(
        jax.tree.leaves(fifo_state),
        jax.tree.leaves(reversed_state),
        strict=True,
    ):
        np.testing.assert_array_equal(reversed_leaf, fifo_leaf)


def test_worker_starvation_waits_until_results_or_capacity_recover():
    class ResultClient:
        def __init__(self):
            self.receive_calls = 0
            self.capacity_calls = 0

        def receive_group(self, session_id, timeout_s):
            del session_id, timeout_s
            self.receive_calls += 1
            if self.receive_calls == 1:
                raise RuntimeUnavailableError("no result yet")
            return ((7, "result"),)

        def capacity(self, session_id, timeout_s):
            del session_id, timeout_s
            self.capacity_calls += 1
            return 0

    class CapacityClient:
        def __init__(self):
            self.capacities = iter((0, 0, 2))

        def capacity(self, session_id, timeout_s):
            del session_id, timeout_s
            return next(self.capacities)

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=2,
        receive_timeout_s=0.01,
    )
    result_client = ResultClient()

    group = runner._receive_group_waiting_for_workers(
        result_client,
        "session",
    )

    assert group == ((7, "result"),)
    assert result_client.capacity_calls == 1
    runner._wait_for_worker_capacity(CapacityClient(), "session")


def _work() -> CoreWorkBatch:
    return CoreWorkBatch(
        valid=jnp.asarray([True, True, False]),
        parent_idx=jnp.asarray([0, -1, 0], dtype=jnp.int32),
        log_L_constraint=jnp.asarray([0.0, -jnp.inf, 0.0]),
        seed_idx=jnp.asarray([1, 0, 1], dtype=jnp.int32),
        seed_pool_idx=jnp.full((3,), -1, dtype=jnp.int32),
        phantom_idx=jnp.zeros((3,), dtype=jnp.int32),
        phantom_priority=jnp.full((3,), -jnp.inf),
        phantom_slot_idx=jnp.full((3,), -1, dtype=jnp.int32),
        phantom_log_L_slot=jnp.full((3,), jnp.inf),
    )


def _batch() -> ConstrainedSampleBatch:
    return ConstrainedSampleBatch(
        U_samples=jnp.asarray([0.4, 0.6, 0.0]),
        log_likelihoods=jnp.asarray([0.5, 0.25, -jnp.inf]),
        num_likelihood_evaluations=jnp.asarray([3, 2, 0], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((3, 0)),
            valid_mask=jnp.zeros((3, 0), dtype=bool),
            log_L=jnp.zeros((3, 0)),
        ),
        num_directions=jnp.zeros((3,), dtype=jnp.int32),
        num_isotropic=jnp.zeros((3,), dtype=jnp.int32),
    )


def _reservation_reference(capacity, parent_indices, valid, sign):
    parent_delta = np.zeros((capacity,), dtype=np.int64)
    root_delta = 0
    num_reserved = 0
    for parent, is_valid in zip(parent_indices, valid, strict=True):
        if not is_valid:
            continue
        num_reserved += sign
        if parent < 0:
            root_delta += sign
        else:
            parent_delta[parent] += sign
    return parent_delta, root_delta, num_reserved


def test_reservations_match_linear_python_reference():
    work = CoreWorkBatch(
        valid=jnp.asarray([True, False, True, True, False]),
        parent_idx=jnp.asarray([2, 3, -1, 2, 0]),
        log_L_constraint=jnp.zeros((5,)),
        seed_idx=jnp.zeros((5,), dtype=jnp.int32),
        seed_pool_idx=jnp.full((5,), -1, dtype=jnp.int32),
        phantom_idx=jnp.zeros((5,), dtype=jnp.int32),
        phantom_priority=jnp.full((5,), -jnp.inf),
        phantom_slot_idx=jnp.full((5,), -1, dtype=jnp.int32),
        phantom_log_L_slot=jnp.full((5,), jnp.inf),
    )
    reservations = ReservationState.empty(6)
    added = _change_reservations(reservations, work, 1)
    expected = _reservation_reference(
        6,
        np.asarray(work.parent_idx),
        np.asarray(work.valid),
        1,
    )
    np.testing.assert_array_equal(np.asarray(added.parent_delta), expected[0])
    assert int(added.root_delta) == expected[1]
    assert int(added.num_reserved) == expected[2]

    removed = _change_reservations(added, work, -1)
    np.testing.assert_array_equal(
        np.asarray(removed.parent_delta),
        np.zeros((6,), dtype=np.int64),
    )
    assert int(removed.root_delta) == 0
    assert int(removed.num_reserved) == 0


def test_reservations_are_planning_data_until_once_only_acceptance():
    model = make_toy_model()
    state = NestedSampler(
        model=model,
        root_allocation_degree=2,
        shell_size=1,
        max_samples=6,
        initial_capacity=6,
    ).initialise(jax.random.PRNGKey(1))
    batch = dataclasses.replace(
        _batch(),
        log_likelihoods=jnp.asarray([
            state.samples.log_likelihoods[0] + 1.0,
            state.samples.log_likelihoods[1] + 1.0,
            -jnp.inf,
        ]),
    )
    reservations = ReservationState.empty(6)
    reserved = _change_reservations(reservations, _work(), 1)

    # Pending work changes the planning lineage count, while the committed
    # state remains a valid two-sample race tree until a result exists.
    assert int(state.root_out_degree) == 2
    assert int(state.samples.out_degree[0]) == 0
    assert int(reserved.root_delta) == 1
    assert int(reserved.parent_delta[0]) == 1
    assert int(reserved.num_reserved) == 2
    provisional = _planning_state(state, reserved)
    assert int(provisional.root_out_degree) == 3
    assert int(provisional.samples.out_degree[0]) == 1

    accepted = _accept_task(
        state,
        reserved,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(2.0),
        _work(),
        batch,
    )
    assert bool(accepted.accepted)
    assert int(accepted.state.num_samples) == 4
    assert int(accepted.state.root_out_degree) == 3
    assert int(accepted.state.samples.out_degree[0]) == 1
    assert int(accepted.reservations.num_reserved) == 0
    np.testing.assert_array_equal(
        np.asarray(accepted.reservations.parent_delta),
        np.zeros((6,), dtype=np.int32),
    )


def test_distributed_continuation_returns_to_heap_not_dispatch_window():
    state = NestedSampler(
        model=make_toy_model(),
        root_allocation_degree=2,
        shell_size=1,
        max_samples=6,
        initial_capacity=6,
    ).initialise(jax.random.PRNGKey(33))
    state = _start_schedule_round(
        state,
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=2,
        delta_K=1,
    )
    work = CoreWorkBatch(
        valid=jnp.asarray([True]),
        parent_idx=jnp.asarray([0], dtype=jnp.int32),
        log_L_constraint=state.samples.log_likelihoods[0:1],
        seed_idx=jnp.asarray([1], dtype=jnp.int32),
        seed_pool_idx=jnp.asarray([-1], dtype=jnp.int32),
        phantom_idx=jnp.asarray([0], dtype=jnp.int32),
        phantom_priority=jnp.asarray([-jnp.inf]),
        phantom_slot_idx=jnp.asarray([-1], dtype=jnp.int32),
        phantom_log_L_slot=jnp.asarray([jnp.inf]),
    )
    accepted_log_L = state.samples.log_likelihoods[0] + 1.0
    batch = ConstrainedSampleBatch(
        U_samples=state.samples.U_samples[0:1],
        log_likelihoods=jnp.reshape(accepted_log_L, (1,)),
        num_likelihood_evaluations=jnp.asarray(
            [3],
            dtype=state.samples.num_likelihood_evaluations.dtype,
        ),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((1, 0)),
            valid_mask=jnp.zeros((1, 0), dtype=bool),
            log_L=jnp.zeros((1, 0)),
        ),
        num_directions=jnp.zeros((1,), dtype=jnp.int32),
        num_isotropic=jnp.zeros((1,), dtype=jnp.int32),
    )
    accepted = _accept_task(
        state,
        _change_reservations(ReservationState.empty(6), work, 1),
        jnp.asarray(7, dtype=jnp.int32),
        accepted_log_L + 1.0,
        work,
        batch,
    )

    assert bool(accepted.accepted)
    schedule = accepted.state.scheduler_data
    assert int(schedule.continuation_count) == 1
    assert int(schedule.continuation_parent_idx[0]) == 2
    assert int(schedule.continuation_thread_id[0]) == 7
    assert not bool(jnp.any(schedule.valid))


def test_distributed_checkpoint_preserves_unreturned_task_and_keys():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0),
        out_degree=(0, 0),
        max_samples=6,
    )
    work = _work()
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(2), 3),
        valid=work.valid,
        log_L_constraints=work.log_L_constraint,
        seed_points=SeedPoint(
            U0=jnp.asarray([0.3, 0.7, 0.3]),
            log_L0=jnp.asarray([0.0, 1.0, 0.0]),
        ),
        sampler_data=None,
    )
    reservations = _change_reservations(
        ReservationState.empty(6),
        work,
        1,
    )
    checkpoint = DistributedState(
        state=state,
        reservations=reservations,
        pending=(PendingTask(
            task_id=7,
            thread_id=jnp.asarray(3, dtype=jnp.int32),
            terminal_log_L=jnp.asarray(2.0),
            work=work,
            request=request,
        ),),
        next_task_id=8,
        session_id="session",
        depth_active=True,
        goal_key=jax.random.PRNGKey(3),
    )
    restored = pickle.loads(pickle.dumps(checkpoint))

    assert restored.pending[0].task_id == 7
    assert restored.next_task_id == 8
    np.testing.assert_array_equal(
        np.asarray(restored.pending[0].request.keys),
        np.asarray(request.keys),
    )
    assert int(restored.state.num_samples) == 2
    assert int(restored.reservations.num_reserved) == 2
    try:
        restored.to_result()
    except RuntimeError as exc:
        assert "pending" in str(exc)
    else:  # pragma: no cover - defensive assertion form
        raise AssertionError("Pending work was exposed as a complete result.")


def test_growth_preserves_pending_payload_and_logical_depth():
    model = make_toy_model()
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=1,
        unlimited_samples=True,
        initial_capacity=2,
    )
    state = _local_checkpoint(runner, jax.random.PRNGKey(8)).state
    work = CoreWorkBatch(
        valid=jnp.asarray([True]),
        parent_idx=jnp.asarray([0], dtype=jnp.int32),
        log_L_constraint=state.samples.log_likelihoods[:1],
        seed_idx=jnp.asarray([1], dtype=jnp.int32),
        seed_pool_idx=jnp.asarray([-1], dtype=jnp.int32),
        phantom_idx=jnp.asarray([0], dtype=jnp.int32),
        phantom_priority=jnp.asarray([-jnp.inf]),
        phantom_slot_idx=jnp.asarray([-1], dtype=jnp.int32),
        phantom_log_L_slot=jnp.asarray([jnp.inf]),
    )
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(9), 1),
        valid=work.valid,
        log_L_constraints=work.log_L_constraint,
        seed_points=SeedPoint(
            U0=state.samples.U_samples[1:2],
            log_L0=state.samples.log_likelihoods[1:2],
        ),
        sampler_data=None,
    )
    checkpoint = DistributedState(
        state=state,
        reservations=_change_reservations(
            ReservationState.empty(2),
            work,
            1,
        ),
        pending=(PendingTask(
            task_id=0,
            thread_id=jnp.asarray(0, dtype=jnp.int32),
            terminal_log_L=jnp.asarray(2.0),
            work=work,
            request=request,
        ),),
        next_task_id=1,
        session_id="session",
        depth_active=True,
        goal_key=state.goal_key,
    )
    grown = runner._grow(checkpoint)

    assert grown.state.samples.log_likelihoods.shape[0] == 4
    assert grown.reservations.parent_delta.shape == (4,)
    assert grown.depth_active
    np.testing.assert_array_equal(
        np.asarray(grown.pending[0].request.keys),
        np.asarray(request.keys),
    )


def test_submit_failure_exposes_newest_resumable_checkpoint():
    class Client:
        def submit_many(self, session_id, tasks):
            del session_id, tasks
            raise RuntimeError("transport unavailable")

    model = make_toy_model()
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=2,
        max_samples=8,
        initial_capacity=8,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(11))
    try:
        runner._dispatch_threads(
            Client(),
            checkpoint,
            DepthCondition(),
            lane_capacity=1,
        )
    except DistributedRunError as exc:
        failed = exc.checkpoint
    else:  # pragma: no cover - defensive assertion form
        raise AssertionError("Transport failure did not expose a checkpoint.")

    assert len(failed.pending) == 1
    assert failed.pending[0].task_id == 0
    assert failed.next_task_id == 1
    assert int(failed.reservations.num_reserved) == 1
    assert int(failed.state.num_samples) == 2

    class UnavailableClient:
        def capacity(self, session_id, timeout_s):
            del session_id, timeout_s
            raise RuntimeError("coordinator unavailable")

    try:
        runner._dispatch_threads(
            UnavailableClient(),
            checkpoint,
            DepthCondition(),
        )
    except DistributedRunError as exc:
        unavailable = exc.checkpoint
    else:  # pragma: no cover - defensive assertion form
        raise AssertionError("Capacity failure did not expose a checkpoint.")

    assert unavailable.next_task_id == checkpoint.next_task_id
    assert unavailable.pending == checkpoint.pending
    assert int(unavailable.state.num_samples) == int(checkpoint.state.num_samples)


def test_worker_slots_are_not_refilled_after_sample_budget_terminates():
    class Client:
        def __init__(self):
            self.submitted = 0

        def submit(self, session_id, task_id, request):
            del session_id, task_id, request
            self.submitted += 1

    model = make_toy_model()
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=1,
        max_samples=2,
        initial_capacity=2,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(12))
    client = Client()

    returned = runner._dispatch_threads(
        client,
        checkpoint,
        DepthCondition(),
        lane_capacity=2,
    )

    assert client.submitted == 0
    assert not returned.pending
    assert int(returned.reservations.num_reserved) == 0


def test_distributed_dispatch_queues_scalar_threads_without_shell_barrier():
    class Client:
        def __init__(self):
            self.requests = []

        def submit_many(self, session_id, tasks):
            del session_id
            self.requests.extend(request for _, request in tasks)

    model = make_toy_model()
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=4,
        delta_K=1,
        max_samples=16,
        initial_capacity=12,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(31))
    client = Client()

    queued = runner._dispatch_threads(
        client,
        checkpoint,
        DepthCondition(),
        lane_capacity=8,
    )

    assert len(client.requests) > 1
    assert len(client.requests) == len(queued.pending)
    assert queued.state.scheduler_data.valid.shape[0] == 8
    assert len(queued.pending) <= queued.state.scheduler_data.valid.shape[0]
    assert all(
        request.log_L_constraints.shape == (1,)
        for request in client.requests
    )


def test_distributed_planning_width_tracks_worker_capacity():
    """A large compressed allocation gap must not manufacture seed lanes."""
    class Client:
        def submit_many(self, session_id, tasks):
            del session_id, tasks

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=128,
        max_samples=32,
        initial_capacity=2,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(32))

    queued = runner._dispatch_threads(
        Client(),
        checkpoint,
        DepthCondition(),
        lane_capacity=1,
    )

    assert queued.state.scheduler_data.valid.shape[0] == 1


def test_distributed_seed_refresh_waits_for_no_pending_tasks():
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=100,
    )
    state = _start_schedule_round(
        state,
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=2,
        delta_K=1,
    )
    schedule = state.scheduler_data
    assert schedule is not None
    refresh_rows = (
        schedule.seed_reservoir_idx.shape[0]
        * SEED_SOURCE_REFRESH_WINDOWS
    )
    state = dataclasses.replace(
        state,
        num_samples=state.num_samples + refresh_rows,
    )

    drained = _depth_status(
        state,
        ReservationState.empty(100),
        max_samples=None,
    )
    pending = _depth_status(
        state,
        dataclasses.replace(
            ReservationState.empty(100),
            num_reserved=jnp.asarray(1, dtype=jnp.int32),
        ),
        max_samples=None,
    )

    assert bool(drained.source_refresh_due)
    assert not bool(drained.schedule_drained)
    assert not bool(drained.has_work)
    assert not bool(pending.source_refresh_due)
    assert not bool(pending.schedule_drained)


def test_distributed_start_seed_storage_grows_without_losing_reservations():
    """The asynchronous boundary doubles exact same-contour coordination."""
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=100,
    )
    state = _start_schedule_round(
        state,
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=2,
        delta_K=1,
    )
    schedule = state.scheduler_data
    assert schedule is not None
    seed_indices = tuple(range(7))
    for reservation_count, seed_idx in enumerate(seed_indices, start=1):
        reservation_idx, reservation_group = _insert_seed_reservation(
            jnp.asarray(seed_idx, dtype=jnp.int32),
            schedule.current_start_group,
            schedule.start_seed_reservation_idx,
            schedule.start_seed_reservation_group,
        )
        schedule = dataclasses.replace(
            schedule,
            start_seed_reservation_idx=reservation_idx,
            start_seed_reservation_group=reservation_group,
            num_start_seeds=jnp.asarray(
                reservation_count,
                dtype=jnp.int32,
            ),
            num_published_start_seeds=jnp.asarray(
                min(reservation_count, 2),
                dtype=jnp.int32,
            ),
        )
    state = dataclasses.replace(state, scheduler_data=schedule)
    distributed = DistributedState(
        state=state,
        reservations=ReservationState.empty(100),
        pending=(),
        next_task_id=0,
        session_id="unit-test",
        depth_active=True,
        goal_key=state.goal_key,
    )
    status = _depth_status(
        state,
        distributed.reservations,
        max_samples=None,
    )
    assert bool(status.needs_seed_growth)

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=2,
        max_samples=100,
    )
    grown = runner._grow_seed_storage(distributed)
    grown_schedule = grown.state.scheduler_data
    assert (
        grown_schedule.start_seed_reservation_idx.shape[0]
        == 2 * schedule.start_seed_reservation_idx.shape[0]
    )
    for seed_idx in seed_indices:
        assert bool(_seed_reservation_contains(
            grown_schedule.start_seed_reservation_idx,
            grown_schedule.start_seed_reservation_group,
            grown_schedule.current_start_group,
            jnp.asarray(seed_idx, dtype=jnp.int32),
        ))


def test_distributed_dispatch_starts_evidence_utility_schedule():
    """Distributed dispatch must preserve the local utility target exactly."""
    class Client:
        def __init__(self):
            self.tasks = []

        def submit_many(self, session_id, tasks):
            del session_id
            self.tasks.extend(tasks)

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=4,
        allocation_target="evidence_improving",
        delta_K=3,
        max_samples=16,
        initial_capacity=12,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(33))
    expected_state = _start_schedule_round(
        checkpoint.state,
        DepthCondition(),
        shell_size=2,
        allocation_target="evidence_improving",
        root_degree=4,
        delta_K=3,
    )
    expected_schedule = expected_state.scheduler_data
    assert expected_schedule is not None
    client = Client()

    queued = runner._dispatch_threads(
        client,
        checkpoint,
        DepthCondition(),
        lane_capacity=2,
    )

    assert client.tasks
    assert len(queued.pending) == len(client.tasks)
    np.testing.assert_array_equal(
        np.asarray(queued.state.scheduler_data.target_K),
        np.asarray(expected_schedule.target_K),
    )


def test_distributed_refill_reserves_pending_same_contour_seed():
    class Client:
        def submit_many(self, session_id, tasks):
            del session_id, tasks

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=1,
        max_samples=8,
        initial_capacity=8,
    )
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(1.0, 2.0),
        log_L_constraints=(-np.inf, -np.inf),
        out_degree=(0, 0),
        max_samples=8,
    )
    state = dataclasses.replace(
        state,
        allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        random_key=jax.random.PRNGKey(290),
        goal_key=jax.random.PRNGKey(291),
    )
    state = _start_schedule_round(
        state,
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=2,
        delta_K=1,
    )
    schedule = state.scheduler_data
    assert schedule is not None
    # Isolate refill behavior with two already-materialised root threads. The
    # compressed queue is exhausted so only these equal-contour heads can be
    # dispatched across the two separate calls below.
    schedule = dataclasses.replace(
        schedule,
        parent_idx=jnp.full_like(schedule.parent_idx, -1),
        thread_id=jnp.arange(2, dtype=schedule.thread_id.dtype),
        log_L_constraint=jnp.asarray([-jnp.inf, -jnp.inf]),
        terminal_log_L=jnp.asarray([1.0, 1.0]),
        valid=jnp.asarray([True, True]),
        next_run=schedule.num_runs,
        remaining_in_run=jnp.asarray(
            0,
            dtype=schedule.remaining_in_run.dtype,
        ),
        next_thread_id=jnp.asarray(2, dtype=schedule.next_thread_id.dtype),
        active=jnp.asarray(True),
    )
    state = dataclasses.replace(state, scheduler_data=schedule)
    checkpoint = DistributedState(
        state=state,
        reservations=ReservationState.empty(8),
        pending=(),
        next_task_id=0,
        session_id="unit-test",
        depth_active=True,
        goal_key=state.goal_key,
    )

    first = runner._dispatch_threads(
        Client(),
        checkpoint,
        DepthCondition(),
        lane_capacity=1,
    )
    refilled = runner._dispatch_threads(
        Client(),
        first,
        DepthCondition(),
        lane_capacity=2,
    )

    assert len(refilled.pending) == 2
    first_work, second_work = (
        pending.work for pending in refilled.pending
    )
    assert float(first_work.log_L_constraint[0]) == float(
        second_work.log_L_constraint[0]
    )
    assert int(first_work.seed_idx[0]) != int(second_work.seed_idx[0])


def test_distributed_refill_reserves_pending_phantom_cluster():
    """A refill cannot reuse either representation of an in-flight cluster."""
    class Client:
        def submit_many(self, session_id, tasks):
            del session_id, tasks

    model = make_toy_model()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        collect_phantom_samples=True,
        max_phantom_samples=1,
    )
    runner = DistributedNestedSampler(
        model=model,
        coordinator_port=5555,
        root_allocation_degree=2,
        delta_K=2,
        max_samples=8,
        initial_capacity=8,
        sampler=sampler,
        phantom_seeding=True,
    )
    checkpoint = _local_checkpoint(runner, jax.random.PRNGKey(294))
    checkpoint = dataclasses.replace(
        checkpoint,
        state=dataclasses.replace(
            checkpoint.state,
            allocation_loop_iter=jnp.asarray(1, dtype=jnp.int32),
            random_key=jnp.asarray(
                [1894348366, 3556391082],
                dtype=jnp.uint32,
            ),
            phantom_seed_pool=_active_phantom_pool(checkpoint.state),
        ),
        depth_active=True,
    )

    first = runner._dispatch_threads(
        Client(),
        checkpoint,
        DepthCondition(),
        lane_capacity=1,
    )
    refilled = runner._dispatch_threads(
        Client(),
        first,
        DepthCondition(),
        lane_capacity=2,
    )

    assert len(first.pending) == 1
    assert int(first.pending[0].work.seed_pool_idx[0]) >= 0
    assert len(refilled.pending) == 2
    assert int(refilled.pending[1].work.seed_pool_idx[0]) == -1
    assert len({
        int(task.work.seed_idx[0])
        for task in refilled.pending
    }) == 2


def test_distributed_refills_reserve_starts_beyond_worker_capacity():
    """Worker width two must spread six root starts without replacement."""
    class Client:
        def submit_many(self, session_id, tasks):
            del session_id, tasks

    runner = DistributedNestedSampler(
        model=make_toy_model(),
        coordinator_port=5555,
        root_allocation_degree=6,
        delta_K=3,
        max_samples=12,
        initial_capacity=12,
    )
    state = make_state(
        root_out_degree=6,
        log_likelihoods=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        log_L_constraints=(-np.inf,) * 6,
        out_degree=(0,) * 6,
        max_samples=12,
    )
    state = dataclasses.replace(
        state,
        # At k=2, d_0 + Delta K k = 12 and the six missing root starts
        # require three successive width-two dispatches.
        allocation_loop_iter=jnp.asarray(2, dtype=jnp.int32),
        random_key=jax.random.PRNGKey(292),
        goal_key=jax.random.PRNGKey(293),
    )
    state = _start_schedule_round(
        state,
        DepthCondition(),
        shell_size=2,
        allocation_target="uniform",
        root_degree=6,
        delta_K=3,
    )
    checkpoint = DistributedState(
        state=state,
        reservations=ReservationState.empty(12),
        pending=(),
        next_task_id=0,
        session_id="unit-test",
        depth_active=True,
        goal_key=state.goal_key,
    )

    selected = []
    for _ in range(3):
        checkpoint = runner._dispatch_threads(
            Client(),
            checkpoint,
            DepthCondition(),
            lane_capacity=2,
        )
        selected.extend(
            int(task.work.seed_idx[0]) for task in checkpoint.pending
        )
        # This test isolates successive dispatch refills. Completed tasks are
        # intentionally omitted because only their persistent start-seed
        # reservation is under test here.
        checkpoint = dataclasses.replace(
            checkpoint,
            reservations=ReservationState.empty(12),
            pending=(),
        )

    assert set(selected) == set(range(6))


def test_worker_request_uses_scalar_and_vmap_paths_above_strict_contour():
    model = make_toy_model()
    sampler = UniDimSliceSampler(model=model, num_slices=2)
    seed_u = jnp.asarray([0.2, 0.8])
    seed_log_likelihood = jax.vmap(model.log_likelihood)(seed_u)

    def run(width):
        request = ConstrainedSampleRequest(
            keys=jax.random.split(jax.random.PRNGKey(10), width),
            valid=jnp.ones((width,), dtype=bool),
            log_L_constraints=jnp.full((width,), -1.0),
            seed_points=SeedPoint(
                U0=seed_u[:width],
                log_L0=seed_log_likelihood[:width],
            ),
            sampler_data=None,
        )
        return jax.jit(
            lambda value: sample_request(sampler, value)
        )(request)

    scalar = run(1)
    vmapped = run(2)
    assert scalar.log_likelihoods.shape == (1,)
    assert vmapped.log_likelihoods.shape == (2,)
    assert np.all(np.asarray(scalar.log_likelihoods) > -1.0)
    assert np.all(np.asarray(vmapped.log_likelihoods) > -1.0)
