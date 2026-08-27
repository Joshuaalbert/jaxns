"""Scientific contracts at the asynchronous constrained-sampling boundary."""

import dataclasses
import pickle

import jax
import numpy as np
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.constrained_sampler import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.core import CoreWorkBatch, NestedSampler
from jaxns.distributed_core import (
    DistributedNestedSampler,
    DistributedRunError,
    DistributedState,
    PendingTask,
    ReservationState,
    _accept_task,
    _change_reservations,
    _planning_state,
)
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.termination_condition import TerminationCondition


def _work() -> CoreWorkBatch:
    return CoreWorkBatch(
        valid=jnp.asarray([True, True, False]),
        parent_idx=jnp.asarray([0, -1, 0], dtype=jnp.int32),
        log_L_constraint=jnp.asarray([0.0, -jnp.inf, 0.0]),
        seed_idx=jnp.asarray([1, 0, 1], dtype=jnp.int32),
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

    accepted = _accept_task(state, reserved, _work(), batch)
    assert bool(accepted.accepted)
    assert int(accepted.state.num_samples) == 4
    assert int(accepted.state.root_out_degree) == 3
    assert int(accepted.state.samples.out_degree[0]) == 1
    assert int(accepted.reservations.num_reserved) == 0
    np.testing.assert_array_equal(
        np.asarray(accepted.reservations.parent_delta),
        np.zeros((6,), dtype=np.int32),
    )


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
        pending=(PendingTask(task_id=7, work=work, request=request),),
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
        config="unused.toml",
        root_allocation_degree=2,
        delta_K=1,
        unlimited_samples=True,
        initial_capacity=2,
    )
    state = runner.initialise(jax.random.PRNGKey(8)).state
    work = CoreWorkBatch(
        valid=jnp.asarray([True]),
        parent_idx=jnp.asarray([0], dtype=jnp.int32),
        log_L_constraint=state.samples.log_likelihoods[:1],
        seed_idx=jnp.asarray([1], dtype=jnp.int32),
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
        pending=(PendingTask(0, work, request),),
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
        config="unused.toml",
        root_allocation_degree=2,
        delta_K=1,
        max_samples=8,
        initial_capacity=8,
    )
    checkpoint = runner.initialise(jax.random.PRNGKey(11))
    try:
        runner._dispatch_threads(
            Client(),
            checkpoint,
            TerminationCondition(),
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
            TerminationCondition(),
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
        config="unused.toml",
        root_allocation_degree=2,
        delta_K=1,
        max_samples=2,
        initial_capacity=2,
    )
    checkpoint = runner.initialise(jax.random.PRNGKey(12))
    client = Client()

    returned = runner._dispatch_threads(
        client,
        checkpoint,
        TerminationCondition(),
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
        config="unused.toml",
        root_allocation_degree=4,
        delta_K=4,
        max_samples=16,
        initial_capacity=12,
    )
    checkpoint = runner.initialise(jax.random.PRNGKey(31))
    client = Client()

    queued = runner._dispatch_threads(
        client,
        checkpoint,
        TerminationCondition(),
        lane_capacity=8,
    )

    assert len(client.requests) > 1
    assert len(client.requests) == len(queued.pending)
    assert all(
        request.log_L_constraints.shape == (1,)
        for request in client.requests
    )


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
