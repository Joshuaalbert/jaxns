"""Asynchronous process-distributed nested-sampling orchestration."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import uuid4

import jax
import jax.numpy as jnp
from jaxctx import CtxParams

from jaxns.constrained_sampler import (
    AbstractSampler,
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    UniDimSliceSampler,
)
from jaxns.core import (
    MAX_SAMPLES_REACHED,
    CoreWorkBatch,
    NestedSampler,
    _accept_work_batch,
    _build_depth_view,
    _plan_work_batch,
    _prepare_sampler_data,
)
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import SeedPoint
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.types import BoolArray, IntArray, PRNGKey

if TYPE_CHECKING:
    from jaxns.results import NestedSamplerResults
    from jaxns.runtime_client import SupervisorClient


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerSession(PureDataclassPytree):
    """Model and sampler data registered once in every worker process."""

    sampler: AbstractSampler
    args: tuple  # [...] arbitrary model argument pytrees
    params: CtxParams | None  # [...] arbitrary parameter pytree leaves


WorkerSession.register_pytree()


class DistributedRunError(RuntimeError):
    """A distributed run failed with an exact resumable ``checkpoint``."""

    __slots__ = ("checkpoint",)

    def __init__(self, message: str, checkpoint: DistributedState):
        super().__init__(message)
        self.checkpoint = checkpoint


@dataclasses.dataclass(frozen=True, slots=True)
class ReservationState(PureDataclassPytree):
    """Provisional lineage starts used only by distributed planning."""

    parent_delta: IntArray  # [A] one row per physical sample slot
    root_delta: IntArray  # []
    num_reserved: IntArray  # [] valid result rows with storage reserved

    @classmethod
    def empty(cls, capacity: int) -> ReservationState:
        return cls(
            parent_delta=jnp.zeros((capacity,), mp_policy.count_dtype),
            root_delta=jnp.asarray(0, mp_policy.count_dtype),
            num_reserved=jnp.asarray(0, mp_policy.count_dtype),
        )

    def resize(self, capacity: int) -> ReservationState:
        current = self.parent_delta.shape[0]
        if capacity < current:
            raise ValueError("Reservation capacity cannot shrink.")
        return dataclasses.replace(
            self,
            parent_delta=jnp.pad(
                self.parent_delta,
                ((0, capacity - current),),
            ),
        )


ReservationState.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class PendingTask(PureDataclassPytree):
    """One retry-stable task and the reservation it will discharge."""

    task_id: int
    work: CoreWorkBatch  # [1] one logical lineage edge
    request: ConstrainedSampleRequest  # [1] one constrained chain


PendingTask.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedState(PureDataclassPytree):
    """Serializable scientific state plus asynchronous runtime continuation."""

    state: State
    reservations: ReservationState
    pending: tuple[PendingTask, ...]
    next_task_id: int
    session_id: str
    depth_active: bool
    goal_key: PRNGKey  # [2]

    def to_result(self) -> NestedSamplerResults:
        """Return the ordinary user-facing result from committed samples."""
        if self.pending or int(self.reservations.num_reserved) != 0:
            raise RuntimeError(
                "Results are unavailable while distributed work is pending."
            )
        return self.state.to_result()


DistributedState.register_pytree()


class PreparedTask(NamedTuple):
    state: State
    reservations: ReservationState
    work: CoreWorkBatch  # [S] static seed-stratification window
    request: ConstrainedSampleRequest  # [S] static planning window
    has_work: BoolArray  # []


class AcceptedTask(NamedTuple):
    state: State
    reservations: ReservationState
    accepted: BoolArray  # []


class DepthStatus(NamedTuple):
    has_work: BoolArray  # []
    needs_growth: BoolArray  # []
    depth_reached: BoolArray  # []
    termination_reason: IntArray  # []


def _planning_state(
        state: State,
        reservations: ReservationState,
) -> State:
    """Apply provisional degrees without exposing them as scientific state."""
    return dataclasses.replace(
        state,
        root_out_degree=state.root_out_degree + reservations.root_delta,
        samples=dataclasses.replace(
            state.samples,
            out_degree=(
                state.samples.out_degree + reservations.parent_delta
            ),
        ),
    )


def _change_reservations(
        reservations: ReservationState,
        work: CoreWorkBatch,
        sign: int,
) -> ReservationState:
    """Add or remove one exactly matching batch of provisional edges."""
    valid = work.valid.astype(mp_policy.count_dtype)
    parent_slots = jnp.maximum(work.parent_idx, 0)
    parent_values = (
        valid
        * (work.parent_idx >= 0).astype(mp_policy.count_dtype)
        * jnp.asarray(sign, mp_policy.count_dtype)
    )
    parent_delta = reservations.parent_delta.at[parent_slots].add(
        parent_values,
    )
    root_change = jnp.sum(
        valid * (work.parent_idx < 0).astype(mp_policy.count_dtype),
        dtype=mp_policy.count_dtype,
    )
    reserved_change = jnp.sum(valid, dtype=mp_policy.count_dtype)
    return ReservationState(
        parent_delta=parent_delta,
        root_delta=(
            reservations.root_delta
            + jnp.asarray(sign, mp_policy.count_dtype) * root_change
        ),
        num_reserved=(
            reservations.num_reserved
            + jnp.asarray(sign, mp_policy.count_dtype) * reserved_change
        ),
    )


def _sample_limit(
        state: State,
        depth_cond: TerminationCondition,
        max_samples: int | None,
) -> IntArray:
    limit = jnp.asarray(
        state.samples.log_likelihoods.shape[0],
        mp_policy.count_dtype,
    )
    if max_samples is not None:
        limit = jnp.asarray(max_samples, mp_policy.count_dtype)
    if depth_cond.max_samples is not None:
        limit = jnp.minimum(
            limit,
            depth_cond.max_samples.astype(mp_policy.count_dtype),
        )
    return limit


@partial(
    jax.jit,
    static_argnames=(
        "dispatch_width",
        "allocation_target",
        "root_degree",
        "delta_K",
        "max_samples",
    ),
)
def _prepare_task(
        state: State,
        reservations: ReservationState,
        sampler,
        depth_cond: TerminationCondition,
        *,
        dispatch_width: int,
        max_threads: IntArray,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
        max_samples: int | None,
) -> PreparedTask:
    """Plan, seed, and reserve one worker batch without sampling it."""
    if (
        isinstance(sampler, UniDimSliceSampler)
        and sampler.direction is not None
    ):
        plan_key, fit_key, sample_key, next_key = jax.random.split(
            state.random_key,
            4,
        )
    else:
        plan_key, sample_key, next_key = jax.random.split(
            state.random_key,
            3,
        )
        fit_key = sample_key

    provisional = _planning_state(state, reservations)
    block_state, plan, relevant, _ = _build_depth_view(
        provisional,
        depth_cond,
        allocation_target=allocation_target,
        root_degree=root_degree,
        delta_K=delta_K,
    )
    physical_free = (
        state.samples.log_likelihoods.shape[0]
        - state.num_samples
        - reservations.num_reserved
    )
    global_free = (
        _sample_limit(state, depth_cond, max_samples)
        - state.num_samples
        - reservations.num_reserved
    )
    available = jnp.maximum(
        jnp.minimum(physical_free, global_free),
        0,
    ).astype(mp_policy.index_dtype)
    available = jnp.minimum(
        available,
        max_threads.astype(mp_policy.index_dtype),
    )
    work = _plan_work_batch(
        plan_key,
        state,
        block_state,
        plan,
        relevant,
        dispatch_width,
        max_valid_lanes=available,
    )

    sampling_state = state
    if (
        isinstance(sampler, UniDimSliceSampler)
        and sampler.direction is not None
    ):
        # Reservations are scheduling evidence, not completed race-tree
        # observations. Direction fitting must use the committed block model.
        committed_blocks, _, _, _ = _build_depth_view(
            state,
            depth_cond,
            allocation_target=allocation_target,
            root_degree=root_degree,
            delta_K=delta_K,
        )
        sampling_state = _prepare_sampler_data(
            fit_key,
            state,
            sampler,
            work,
            committed_blocks,
        )

    seed_points = SeedPoint(
        U0=jax.tree.map(
            lambda values: values[work.seed_idx],
            sampling_state.samples.U_samples,
        ),
        log_L0=(
            sampling_state.samples.log_likelihoods[work.seed_idx]
        ),
    )
    request = ConstrainedSampleRequest(
        keys=jax.random.split(sample_key, dispatch_width),
        valid=work.valid,
        log_L_constraints=work.log_L_constraint,
        seed_points=seed_points,
        sampler_data=sampling_state.sampler_data,
    )
    has_work = work.num_valid > 0
    updated = dataclasses.replace(
        sampling_state,
        random_key=jnp.where(has_work, next_key, state.random_key),
    )
    reserved = jax.lax.cond(
        has_work,
        lambda unused: _change_reservations(reservations, work, 1),
        lambda unused: reservations,
        operand=None,
    )
    return PreparedTask(
        state=updated,
        reservations=reserved,
        work=work,
        request=request,
        has_work=has_work,
    )


@jax.jit
def _accept_task(
        state: State,
        reservations: ReservationState,
        work: CoreWorkBatch,
        batch: ConstrainedSampleBatch,
) -> AcceptedTask:
    """Validate and atomically convert one reservation into real samples."""
    accepted = jnp.all(
        jnp.logical_not(work.valid)
        | (
            jnp.logical_not(jnp.isnan(batch.log_likelihoods))
            & (batch.log_likelihoods > work.log_L_constraint)
        )
    )

    def commit(_):
        return AcceptedTask(
            state=_accept_work_batch(state, work, batch),
            reservations=_change_reservations(reservations, work, -1),
            accepted=jnp.asarray(True, mp_policy.bool_dtype),
        )

    return jax.lax.cond(
        accepted,
        commit,
        lambda unused: AcceptedTask(
            state=state,
            reservations=reservations,
            accepted=jnp.asarray(False, mp_policy.bool_dtype),
        ),
        operand=None,
    )


@partial(
    jax.jit,
    static_argnames=(
        "allocation_target",
        "root_degree",
        "delta_K",
        "max_samples",
    ),
)
def _depth_status(
        state: State,
        reservations: ReservationState,
        depth_cond: TerminationCondition,
        *,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
        max_samples: int | None,
) -> DepthStatus:
    """Classify whether to dispatch, grow, finish depth, or terminate."""
    provisional = _planning_state(state, reservations)
    _, plan, relevant, register = _build_depth_view(
        provisional,
        depth_cond,
        allocation_target=allocation_target,
        root_degree=root_degree,
        delta_K=delta_K,
    )
    has_gap = jnp.any(plan.under_allocated(relevant))
    scalar_cond = dataclasses.replace(
        depth_cond,
        dlogZ=None,
        cummax_XL_frac=None,
    )
    scalar_done, scalar_reason = register.is_done(scalar_cond)
    limit = _sample_limit(state, depth_cond, max_samples)
    hard_limit = state.num_samples >= limit
    termination_reason = jnp.where(
        state.termination_reason != 0,
        state.termination_reason,
        jnp.where(
            scalar_done,
            scalar_reason,
            jnp.where(
                hard_limit,
                jnp.asarray(MAX_SAMPLES_REACHED, mp_policy.count_dtype),
                jnp.asarray(0, mp_policy.count_dtype),
            ),
        ),
    )
    terminal = termination_reason != 0
    physical_full = (
        state.num_samples + reservations.num_reserved
        >= state.samples.log_likelihoods.shape[0]
    )
    below_limit = (
        state.num_samples + reservations.num_reserved < limit
    )
    has_work = has_gap & below_limit & jnp.logical_not(terminal)
    needs_growth = (
        has_work
        & physical_full
        & jnp.logical_not(terminal)
    )
    can_dispatch = has_work & jnp.logical_not(physical_full)
    depth_reached = jnp.logical_not(
        terminal | needs_growth | can_dispatch
    )
    return DepthStatus(
        has_work=can_dispatch,
        needs_growth=needs_growth,
        depth_reached=depth_reached,
        termination_reason=termination_reason,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedNestedSampler:
    """Run nested sampling over an asynchronous multi-node worker pool.

    Distributed allocation has no shell width. One pending task represents one
    logical lineage thread, and idle pool lanes are continuously filled from
    currently known allocation gaps. A worker's ``batch_size`` is a private
    device-execution choice used to combine compatible scalar tasks with
    ``jax.vmap``.

    Args:
        model: Scientific prior and scalar log-likelihood model.
        config: Main-node TOML identifying an already started coordinator.
        receive_timeout_s: Maximum time to wait for one task completion.
    """

    model: Model
    config: str | Path
    target_num_live_points: int | None = None
    root_allocation_degree: int | None = None
    max_samples: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    allocation_target: Literal[
        "uniform",
        "evidence_improving",
        "posterior_improving",
    ] = "uniform"
    delta_K: int | None = None
    initial_capacity: int | None = None
    unlimited_samples: bool = False
    receive_timeout_s: float = 300.0
    _core: NestedSampler = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        root_degree = self.root_allocation_degree
        if root_degree is None:
            root_degree = self.target_num_live_points
        if root_degree is None:
            root_degree = max(1, 30 * int(self.model.U_ndims(self.args, self.params)))
        # Allocation depth is scientific policy, not worker topology. A root-
        # sized default exposes enough independent lineage work for a pool
        # without changing when workers join or leave.
        delta_K = root_degree if self.delta_K is None else self.delta_K
        initial_capacity = self.initial_capacity
        if initial_capacity is None:
            initial_capacity = root_degree + 10 * delta_K
        core = NestedSampler(
            model=self.model,
            target_num_live_points=self.target_num_live_points,
            root_allocation_degree=self.root_allocation_degree,
            max_samples=self.max_samples,
            # This width is used only by NestedSampler configuration and root
            # storage initialization. Distributed replacement never reads it.
            shell_size=1,
            args=self.args,
            params=self.params,
            sampler=self.sampler,
            termination_condition=self.termination_condition,
            store_phantom_samples=self.store_phantom_samples,
            collect_phantom_samples=self.collect_phantom_samples,
            allocation_target=self.allocation_target,
            delta_K=delta_K,
            initial_capacity=initial_capacity,
            unlimited_samples=self.unlimited_samples,
        )
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "target_num_live_points", core.target_num_live_points)
        object.__setattr__(self, "root_allocation_degree", core.root_allocation_degree)
        object.__setattr__(self, "max_samples", core.max_samples)
        object.__setattr__(self, "sampler", core.sampler)
        object.__setattr__(self, "termination_condition", core.termination_condition)
        object.__setattr__(self, "delta_K", core.delta_K)
        object.__setattr__(self, "initial_capacity", core.initial_capacity)

    def initialise(self, key: PRNGKey | None = None) -> DistributedState:
        """Create an immutable distributed checkpoint from a run key.

        The finite root prior batch is bootstrapped in the scientific process;
        complete constrained replacement chains are the distributed work unit.
        """
        # Root draws have no stationary seed or parent-chain state. Keeping
        # this one-time vmap on the coordinator preserves the established
        # initial-state construction; the repeated expensive replacement work
        # begins at the worker boundary documented for this first release.
        state = self._core.initialise(key)
        capacity = state.samples.log_likelihoods.shape[0]
        return DistributedState(
            state=state,
            reservations=ReservationState.empty(capacity),
            pending=(),
            next_task_id=0,
            session_id=uuid4().hex,
            depth_active=False,
            goal_key=state.goal_key,
        )

    def run(self, key: PRNGKey | None = None) -> DistributedState:
        """Run until the configured expectation-based goal is met."""

        def default_goal(state: State) -> bool:
            if int(state.goal_loop_iter) == 0:
                return False
            done, _ = state.compute_termination_register().is_done(
                self.termination_condition
            )
            return bool(done)

        return self.run_until_goal(default_goal, key=key)

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
    ) -> DistributedState:
        """Initialise and run the asynchronous Python goal/depth loops.

        Args:
            goal_cond: User goal evaluated only on drained immutable states.
            depth_cond: Expectation-based boundary for each allocation epoch.
            key: Run random key; a deterministic default is used when absent.

        Returns:
            A drained distributed checkpoint suitable for results or resumption.
        """
        return self.resume_until_goal(
            self.initialise(key),
            goal_cond,
            depth_cond=depth_cond,
        )

    def resume_until_goal(
            self,
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
    ) -> DistributedState:
        """Resume an immutable checkpoint, including retry-stable tasks.

        Raises:
            DistributedRunError: If runtime work fails. Its ``checkpoint``
                field contains every reservation and task created before the
                failure and can be passed back to this method.
        """
        from jaxns.runtime_client import SupervisorClient

        if depth_cond is None:
            depth_cond = self.termination_condition
        session = WorkerSession(
            sampler=self.sampler,
            args=distributed.state.args,
            params=distributed.state.params,
        )
        with SupervisorClient.from_config(self.config) as client:
            try:
                capacities = client.register(distributed.session_id, session)
                if not capacities:
                    raise RuntimeError("The supervisor has no ready workers.")
                for pending in distributed.pending:
                    client.submit(
                        distributed.session_id,
                        pending.task_id,
                        pending.request,
                    )
                completed = self._run_connected(
                    client,
                    distributed,
                    goal_cond,
                    depth_cond,
                )
                distributed = completed
                client.release(completed.session_id)
                return completed
            except DistributedRunError:
                raise
            except Exception as exc:
                raise DistributedRunError(
                    f"Distributed execution failed: {exc}",
                    distributed,
                ) from exc

    def _run_connected(
            self,
            client: SupervisorClient,
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
    ) -> DistributedState:
        completed_tasks: dict[int, ConstrainedSampleBatch] = {}
        while True:
            state = distributed.state
            if (
                not distributed.pending
                and (
                    int(state.termination_reason) != 0
                    or (
                        not distributed.depth_active
                        and bool(goal_cond(state))
                    )
                )
            ):
                return distributed

            if not distributed.depth_active:
                depth_key, goal_key = jax.random.split(state.random_key)
                state = dataclasses.replace(
                    state,
                    random_key=depth_key,
                    goal_key=goal_key,
                    needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
                    depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
                )
                distributed = dataclasses.replace(
                    distributed,
                    state=state,
                    depth_active=True,
                    goal_key=goal_key,
                )

            distributed = self._grow_if_needed(distributed, depth_cond)
            distributed = self._dispatch_threads(
                client,
                distributed,
                depth_cond,
            )
            if distributed.pending:
                # Sampling completion is asynchronous, but scientific commits
                # follow task identity. Runtime latency (including the extra
                # bytes produced by phantom collection) must not choose a
                # different race tree for the same assigned random stream.
                pending = min(
                    distributed.pending,
                    key=lambda task: task.task_id,
                )
                while pending.task_id not in completed_tasks:
                    try:
                        task_id, batch = client.receive(
                            distributed.session_id,
                            timeout_s=self.receive_timeout_s,
                        )
                    except Exception as exc:
                        raise DistributedRunError(
                            f"Waiting for distributed task failed: {exc}",
                            distributed,
                        ) from exc
                    if not any(
                        task.task_id == task_id
                        for task in distributed.pending
                    ):
                        # A replay for an already committed task is safe to
                        # release because its state mutation already happened.
                        client.acknowledge(distributed.session_id, task_id)
                        continue
                    completed_tasks[task_id] = batch
                batch = completed_tasks.pop(pending.task_id)
                accepted = _accept_task(
                    distributed.state,
                    distributed.reservations,
                    pending.work,
                    batch,
                )
                if not bool(accepted.accepted):
                    raise DistributedRunError(
                        f"Worker task {pending.task_id} violated its strict contour.",
                        distributed,
                    )
                client.acknowledge(
                    distributed.session_id,
                    pending.task_id,
                )
                distributed = dataclasses.replace(
                    distributed,
                    state=accepted.state,
                    reservations=accepted.reservations,
                    pending=tuple(
                        task
                        for task in distributed.pending
                        if task.task_id != pending.task_id
                    ),
                )
                continue

            status = self._status(distributed, depth_cond)
            if bool(status.needs_growth):
                distributed = self._grow(distributed)
                continue
            if int(status.termination_reason) != 0:
                return dataclasses.replace(
                    distributed,
                    state=dataclasses.replace(
                        distributed.state,
                        termination_reason=status.termination_reason,
                        needs_growth=jnp.asarray(
                            False,
                            mp_policy.bool_dtype,
                        ),
                        depth_reached=jnp.asarray(
                            False,
                            mp_policy.bool_dtype,
                        ),
                    ),
                )
            if bool(status.depth_reached):
                state = dataclasses.replace(
                    distributed.state,
                    random_key=distributed.goal_key,
                    goal_key=distributed.goal_key,
                    depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
                    goal_loop_iter=(
                        distributed.state.goal_loop_iter
                        + jnp.asarray(
                            1,
                            distributed.state.goal_loop_iter.dtype,
                        )
                    ),
                )
                distributed = dataclasses.replace(
                    distributed,
                    state=state,
                    depth_active=False,
                )
                continue
            if bool(status.has_work):
                # A pool may temporarily have no live lanes while an operator
                # replaces a node. Keep the immutable scientific state in the
                # same depth and allow a dynamically joined worker to resume
                # it, bounded by the ordinary receive timeout.
                deadline = time.monotonic() + self.receive_timeout_s
                while time.monotonic() < deadline:
                    remaining_s = max(0.0, deadline - time.monotonic())
                    try:
                        lanes = client.capacity(
                            distributed.session_id,
                            timeout_s=remaining_s,
                        )
                    except Exception as exc:
                        raise DistributedRunError(
                            f"Observing distributed capacity failed: {exc}",
                            distributed,
                        ) from exc
                    if lanes > 0:
                        break
                    time.sleep(0.1)
                else:
                    raise DistributedRunError(
                        "No distributed worker capacity became available "
                        "before the receive timeout.",
                        distributed,
                    )
                continue
            raise DistributedRunError(
                "Distributed depth made no dispatch, growth, completion, "
                "or termination progress.",
                distributed,
            )

    def _dispatch_threads(
            self,
            client: SupervisorClient,
            distributed: DistributedState,
            depth_cond: TerminationCondition,
            lane_capacity: int | None = None,
    ) -> DistributedState:
        # Keep every compatible live lane fed without imposing a sampling
        # wave. One task credit per live lane fills the whole pool immediately
        # and avoids speculative scientific work beyond measured capacity.
        if lane_capacity is None:
            try:
                capacity = client.capacity(
                    distributed.session_id,
                    timeout_s=self.receive_timeout_s,
                )
            except Exception as exc:
                raise DistributedRunError(
                    f"Observing distributed capacity failed: {exc}",
                    distributed,
                ) from exc
        else:
            capacity = lane_capacity
        pending_limit = capacity
        free = max(0, pending_limit - len(distributed.pending))
        planning_width = int(self.delta_K)
        while free > 0 and bool(self._status(distributed, depth_cond).has_work):
            prepared = _prepare_task(
                distributed.state,
                distributed.reservations,
                self.sampler,
                depth_cond,
                # This is a seed-stratification window, not an execution
                # batch. Every valid lane becomes its own transport task.
                dispatch_width=planning_width,
                max_threads=jnp.asarray(free, mp_policy.index_dtype),
                allocation_target=self.allocation_target,
                root_degree=int(self.root_allocation_degree),
                delta_K=int(self.delta_K),
                max_samples=self.max_samples,
            )
            if not bool(prepared.has_work):
                break
            num_tasks = int(prepared.work.num_valid)
            # One transfer per planning window avoids a device synchronization
            # for every scalar task when pickle asks each JAX leaf for bytes.
            host_work, host_request = jax.device_get(
                (prepared.work, prepared.request)
            )
            tasks = tuple(
                PendingTask(
                    task_id=distributed.next_task_id + lane,
                    work=CoreWorkBatch(
                        valid=host_work.valid[lane:lane + 1],
                        parent_idx=host_work.parent_idx[lane:lane + 1],
                        log_L_constraint=(
                            host_work.log_L_constraint[lane:lane + 1]
                        ),
                        seed_idx=host_work.seed_idx[lane:lane + 1],
                    ),
                    request=ConstrainedSampleRequest(
                        keys=host_request.keys[lane:lane + 1],
                        valid=host_request.valid[lane:lane + 1],
                        log_L_constraints=(
                            host_request.log_L_constraints[lane:lane + 1]
                        ),
                        seed_points=SeedPoint(
                            U0=jax.tree.map(
                                lambda values, lane=lane: values[lane:lane + 1],
                                host_request.seed_points.U0,
                            ),
                            log_L0=(
                                host_request.seed_points.log_L0[lane:lane + 1]
                            ),
                        ),
                        sampler_data=host_request.sampler_data,
                    ),
                )
                for lane in range(num_tasks)
            )
            # Store the reservation before transport. A submit failure returns
            # a checkpoint that still owns the exact random key and parent.
            distributed = dataclasses.replace(
                distributed,
                state=prepared.state,
                reservations=prepared.reservations,
                pending=distributed.pending + tasks,
                next_task_id=distributed.next_task_id + num_tasks,
            )
            for task in tasks:
                try:
                    client.submit(
                        distributed.session_id,
                        task.task_id,
                        task.request,
                    )
                except Exception as exc:
                    raise DistributedRunError(
                        f"Submitting distributed task failed: {exc}",
                        distributed,
                    ) from exc
            free -= num_tasks
        return distributed

    def _status(
            self,
            distributed: DistributedState,
            depth_cond: TerminationCondition,
    ) -> DepthStatus:
        return _depth_status(
            distributed.state,
            distributed.reservations,
            depth_cond,
            allocation_target=self.allocation_target,
            root_degree=int(self.root_allocation_degree),
            delta_K=int(self.delta_K),
            max_samples=self.max_samples,
        )

    def _grow_if_needed(
            self,
            distributed: DistributedState,
            depth_cond: TerminationCondition,
    ) -> DistributedState:
        status = self._status(distributed, depth_cond)
        if bool(status.needs_growth):
            return self._grow(distributed)
        return distributed

    def _grow(self, distributed: DistributedState) -> DistributedState:
        capacity = distributed.state.samples.log_likelihoods.shape[0]
        required = (
            int(distributed.state.num_samples)
            + int(distributed.reservations.num_reserved)
            + max(1, max(
                (task.request.log_L_constraints.shape[0]
                 for task in distributed.pending),
                default=1,
            ))
        )
        new_capacity = max(2 * capacity, required)
        if self.max_samples is not None:
            new_capacity = min(
                new_capacity,
                self.max_samples,
            )
        if new_capacity <= capacity:
            return dataclasses.replace(
                distributed,
                state=dataclasses.replace(
                    distributed.state,
                    termination_reason=jnp.asarray(
                        MAX_SAMPLES_REACHED,
                        mp_policy.count_dtype,
                    ),
                ),
            )
        return dataclasses.replace(
            distributed,
            state=dataclasses.replace(
                distributed.state.resize(new_capacity),
                needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
                depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
            ),
            reservations=distributed.reservations.resize(new_capacity),
        )
