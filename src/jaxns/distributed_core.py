"""Asynchronous process-distributed nested-sampling orchestration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
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
    work: CoreWorkBatch
    request: ConstrainedSampleRequest


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
    work: CoreWorkBatch
    request: ConstrainedSampleRequest
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
    """Run a local JAXNS core over a trusted asynchronous worker stack.

    ``jaxns-cli`` owns the local supervisor and worker lifecycle. This object
    owns scientific state, reservations, keys, and the Python goal/depth event
    loop; the supervisor never interprets nested-sampling state.

    Args:
        nested_sampler: Scientific model, sampler, allocation, and capacity
            configuration shared with the established local runner.
        config: TOML file identifying an already started local worker stack.
        receive_timeout_s: Maximum time to wait for one task completion.
    """

    nested_sampler: NestedSampler
    config: str | Path
    receive_timeout_s: float = 300.0

    def initialise(self, key: PRNGKey | None = None) -> DistributedState:
        """Create an immutable distributed checkpoint from a run key.

        The finite root prior batch is bootstrapped in the scientific process;
        complete constrained replacement chains are the distributed work unit.
        """
        # Root draws have no stationary seed or parent-chain state. Keeping
        # this one-time vmap on the coordinator preserves the established
        # initial-state construction; the repeated expensive replacement work
        # begins at the worker boundary documented for this first release.
        state = self.nested_sampler.initialise(key)
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
                self.nested_sampler.termination_condition
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
            depth_cond = self.nested_sampler.termination_condition
        session = WorkerSession(
            sampler=self.nested_sampler.sampler,
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
                    capacities,
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
            capacities: tuple[int, ...],
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
    ) -> DistributedState:
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
            distributed = self._fill_workers(
                client,
                capacities,
                distributed,
                depth_cond,
            )
            if distributed.pending:
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
                matching = tuple(
                    task
                    for task in distributed.pending
                    if task.task_id == task_id
                )
                if not matching:
                    # The supervisor retains results until ACK. A replayed
                    # result for an already committed task is safe to discard.
                    client.acknowledge(distributed.session_id, task_id)
                    continue
                pending = matching[0]
                accepted = _accept_task(
                    distributed.state,
                    distributed.reservations,
                    pending.work,
                    batch,
                )
                if not bool(accepted.accepted):
                    raise DistributedRunError(
                        f"Worker task {task_id} violated its strict contour.",
                        distributed,
                    )
                client.acknowledge(distributed.session_id, task_id)
                distributed = dataclasses.replace(
                    distributed,
                    state=accepted.state,
                    reservations=accepted.reservations,
                    pending=tuple(
                        task
                        for task in distributed.pending
                        if task.task_id != task_id
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
            raise DistributedRunError(
                "Distributed depth made no dispatch, growth, completion, "
                "or termination progress.",
                distributed,
            )

    def _fill_workers(
            self,
            client: SupervisorClient,
            capacities: tuple[int, ...],
            distributed: DistributedState,
            depth_cond: TerminationCondition,
    ) -> DistributedState:
        occupied: dict[int, int] = {}
        for task in distributed.pending:
            width = task.request.log_L_constraints.shape[0]
            occupied[width] = occupied.get(width, 0) + 1
        available: dict[int, int] = {}
        for width in capacities:
            available[width] = available.get(width, 0) + 1

        for width, count in available.items():
            free = count - occupied.get(width, 0)
            for _ in range(max(0, free)):
                # Re-evaluate the stop boundary after every reservation.
                # Planning gaps alone are insufficient: scalar budgets may
                # already have ended this depth, and another reservation can
                # fill the remaining physical capacity.
                status = self._status(distributed, depth_cond)
                if not bool(status.has_work):
                    break
                prepared = _prepare_task(
                    distributed.state,
                    distributed.reservations,
                    self.nested_sampler.sampler,
                    depth_cond,
                    dispatch_width=width,
                    allocation_target=self.nested_sampler.allocation_target,
                    root_degree=int(
                        self.nested_sampler.root_allocation_degree
                    ),
                    delta_K=int(self.nested_sampler.delta_K),
                    max_samples=self.nested_sampler.max_samples,
                )
                if not bool(prepared.has_work):
                    break
                task = PendingTask(
                    task_id=distributed.next_task_id,
                    work=prepared.work,
                    request=prepared.request,
                )
                # Add the immutable reservation before transport. If submit
                # fails, the returned checkpoint still owns the exact task.
                distributed = dataclasses.replace(
                    distributed,
                    state=prepared.state,
                    reservations=prepared.reservations,
                    pending=distributed.pending + (task,),
                    next_task_id=distributed.next_task_id + 1,
                )
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
            allocation_target=self.nested_sampler.allocation_target,
            root_degree=int(self.nested_sampler.root_allocation_degree),
            delta_K=int(self.nested_sampler.delta_K),
            max_samples=self.nested_sampler.max_samples,
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
        if self.nested_sampler.max_samples is not None:
            new_capacity = min(
                new_capacity,
                self.nested_sampler.max_samples,
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
