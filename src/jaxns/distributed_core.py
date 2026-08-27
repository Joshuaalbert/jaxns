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
import numpy as np
from jaxctx import CtxParams

from jaxns.checkpoint import (
    CHECKPOINT_CADENCE_SECONDS,
    CheckpointManager,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    LikelihoodEvaluation,
    LikelihoodRequest,
)
from jaxns.core import (
    MAX_SAMPLES_REACHED,
    CoreWorkBatch,
    NestedSampler,
    _accept_work_batch,
    _build_depth_view,
    _build_init_state,
    _plan_work_batch,
    _prepare_sampler_data,
)
from jaxns.logging import jaxns_logger
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

    model: Model
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


@partial(jax.jit, inline=True)
def _sample_prior_points(
        keys: PRNGKey,
        model: Model,
        args,
        params,
):
    """Draw prior-space points without evaluating their likelihoods."""
    return jax.vmap(
        lambda key: model.sample_U(key, args=args, params=params)
    )(keys)


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
    if sampler.uses_adaptive_directions():
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
    if sampler.uses_adaptive_directions():
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


@partial(jax.jit, inline=True)
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


class DistributedNestedSampler:
    """Run nested sampling over an asynchronous multi-node worker pool.

    Distributed allocation has no shell width. One pending task represents one
    logical lineage thread, and idle pool lanes are continuously filled from
    currently known allocation gaps. A worker's ``batch_size`` is a private
    device-execution choice used to combine compatible tasks and batch their
    ready likelihood proposals while each constrained chain continues
    independently.

    Args:
        model: Scientific prior and scalar log-likelihood model.
        coordinator_port: TCP port identifying the already started local
            coordinator. The scientific connection itself stays on same-user
            IPC derived from this port.
        receive_timeout_s: Maximum time to wait for a coordinator health
            response. Worker completion itself has no deadline because an
            empty pool is a recoverable operational state.
    """

    __slots__ = (
        "_core",
        "allocation_target",
        "args",
        "collect_phantom_samples",
        "coordinator_port",
        "delta_K",
        "initial_capacity",
        "max_samples",
        "model",
        "params",
        "receive_timeout_s",
        "root_allocation_degree",
        "sampler",
        "store_phantom_samples",
        "target_num_live_points",
        "termination_condition",
        "unlimited_samples",
    )

    def __init__(
            self,
            model: Model,
            coordinator_port: int,
            target_num_live_points: int | None = None,
            root_allocation_degree: int | None = None,
            max_samples: int | None = None,
            args: tuple = (),
            params: CtxParams | None = None,
            sampler: AbstractSampler | None = None,
            termination_condition: TerminationCondition | None = None,
            store_phantom_samples: bool = False,
            collect_phantom_samples: bool = False,
            allocation_target: Literal[
                "uniform",
                "evidence_improving",
                "posterior_improving",
            ] = "uniform",
            delta_K: int | None = None,
            initial_capacity: int | None = None,
            unlimited_samples: bool = False,
            receive_timeout_s: float = 300.0,
    ) -> None:
        # This object owns a runtime service configuration and is deliberately
        # mutable. Manual slots make that lifecycle explicit without posing as
        # one of the immutable scientific-state dataclasses.
        self.model = model
        self.coordinator_port = coordinator_port
        self.target_num_live_points = target_num_live_points
        self.root_allocation_degree = root_allocation_degree
        self.max_samples = max_samples
        self.args = args
        self.params = params
        self.sampler = sampler
        self.termination_condition = termination_condition
        self.store_phantom_samples = store_phantom_samples
        self.collect_phantom_samples = collect_phantom_samples
        self.allocation_target = allocation_target
        self.delta_K = delta_K
        self.initial_capacity = initial_capacity
        self.unlimited_samples = unlimited_samples
        self.receive_timeout_s = receive_timeout_s

        if (
            type(self.coordinator_port) is not int
            or not 1 <= self.coordinator_port <= 65_535
        ):
            raise ValueError(
                "coordinator_port must be an integer from 1 to 65,535."
            )
        if self.receive_timeout_s <= 0.0:
            raise ValueError("receive_timeout_s must be positive.")
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
        self._core = core
        self.target_num_live_points = core.target_num_live_points
        self.root_allocation_degree = core.root_allocation_degree
        self.max_samples = core.max_samples
        self.sampler = core.sampler
        self.termination_condition = core.termination_condition
        self.delta_K = core.delta_K
        self.initial_capacity = core.initial_capacity

    def initialise(self, key: PRNGKey | None = None) -> DistributedState:
        """Create a root checkpoint without a local likelihood evaluation.

        The scientific process may have no likelihood-capable device. It draws
        unit-hypercube coordinates locally, but every root likelihood is an
        explicit worker task just like later constrained-chain evaluations.
        """
        from jaxns.runtime_client import SupervisorClient

        session_id = uuid4().hex
        session = WorkerSession(
            model=self.model,
            sampler=self.sampler,
            args=self.args,
            params=self.params,
        )
        with SupervisorClient.from_port(self.coordinator_port) as client:
            client.register(session_id, session)
            distributed = self._initialise_connected(
                client,
                session_id,
                key,
            )
            client.release(session_id)
            return distributed

    def _initialise_connected(
            self,
            client: SupervisorClient,
            session_id: str,
            key: PRNGKey | None,
    ) -> DistributedState:
        """Build roots while retaining one already registered worker session."""
        if key is None:
            key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        root_keys = jax.random.split(
            init_key,
            int(self.root_allocation_degree),
        )
        U_samples = _sample_prior_points(
            root_keys,
            self.model,
            self.args,
            self.params,
        )
        num_evals = jnp.ones(
            (int(self.root_allocation_degree),),
            mp_policy.count_dtype,
        )
        next_task_id = 0

        log_likelihoods, next_task_id = self._evaluate_likelihoods(
            client,
            session_id,
            U_samples,
            next_task_id,
        )
        invalid = np.flatnonzero(np.asarray(
            jax.device_get(log_likelihoods <= -jnp.inf)
        ))
        while invalid.size:
            # Preserve one independent retry stream per root. Only roots
            # rejected by the sentinel contour consume another key and
            # likelihood call, matching ordinary initialization exactly.
            key_pairs = jax.vmap(
                lambda value: jax.random.split(value, 2)
            )(root_keys[invalid])
            root_keys = root_keys.at[invalid].set(key_pairs[:, 0])
            proposals = _sample_prior_points(
                key_pairs[:, 1],
                self.model,
                self.args,
                self.params,
            )
            U_samples = jax.tree.map(
                lambda current, proposal, slots=invalid: current.at[
                    slots
                ].set(proposal),
                U_samples,
                proposals,
            )
            replacements, next_task_id = self._evaluate_likelihoods(
                client,
                session_id,
                proposals,
                next_task_id,
            )
            log_likelihoods = log_likelihoods.at[invalid].set(
                replacements
            )
            num_evals = num_evals.at[invalid].add(1)
            invalid = np.flatnonzero(np.asarray(
                jax.device_get(log_likelihoods <= -jnp.inf)
            ))

        state = _build_init_state(
            self.model,
            self.args,
            self.params,
            U_samples,
            log_likelihoods,
            num_evals,
            sample_capacity=int(self.initial_capacity),
            num_phantom=int(self.sampler.num_phantom()),
        )
        sampler_data = self.sampler.initial_sampler_data(
            int(self.model.U_ndims(self.args, self.params))
        )
        state = dataclasses.replace(
            state,
            sampler_data=sampler_data,
            random_key=run_key,
            goal_key=run_key,
            depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
        )
        capacity = state.samples.log_likelihoods.shape[0]
        return DistributedState(
            state=state,
            reservations=ReservationState.empty(capacity),
            pending=(),
            # Root likelihood operations shared this session and already used
            # its initial IDs. Continue monotonically so an acknowledgement
            # still in transit can never erase a newly queued sampling task.
            next_task_id=next_task_id,
            session_id=session_id,
            depth_active=False,
            goal_key=state.goal_key,
        )

    def _evaluate_likelihoods(
            self,
            client: SupervisorClient,
            session_id: str,
            U_samples,
            next_task_id: int,
    ) -> tuple[jax.Array, int]:
        """Evaluate one finite prior batch entirely in worker processes."""
        host_samples = jax.device_get(U_samples)
        width = jax.tree.leaves(host_samples)[0].shape[0]
        tasks = tuple(
            (
                next_task_id + row,
                LikelihoodRequest(U_samples=jax.tree.map(
                    lambda values, row=row: values[row:row + 1],
                    host_samples,
                )),
            )
            for row in range(width)
        )
        client.evaluate_many(session_id, tasks)
        task_ids = {task_id for task_id, _ in tasks}
        evaluations: dict[int, LikelihoodEvaluation] = {}
        while evaluations.keys() != task_ids:
            for task_id, result in self._receive_group_waiting_for_workers(
                    client,
                    session_id,
            ):
                if task_id not in task_ids:
                    raise RuntimeError(
                        f"Received unexpected initialization task {task_id}."
                    )
                evaluations[task_id] = result
                client.acknowledge(session_id, task_id)
        ordered = tuple(
            evaluations[task_id].log_likelihoods
            for task_id, _ in tasks
        )
        return jnp.concatenate(ordered), next_task_id + width

    def _receive_group_waiting_for_workers(
            self,
            client,
            session_id: str,
    ):
        """Wait for results while distinguishing starvation from coordinator loss."""
        from jaxns.runtime_client import RuntimeUnavailableError

        waiting_for_workers = False
        probe_s = min(5.0, max(0.1, self.receive_timeout_s))
        while True:
            try:
                group = client.receive_group(
                    session_id,
                    timeout_s=probe_s,
                )
            except RuntimeUnavailableError:
                # A long-running task and a starved pool are both legitimate.
                # Ask the coordinator which state applies; failure of this
                # health exchange is the actual resumable runtime error.
                capacity = client.capacity(
                    session_id,
                    timeout_s=self.receive_timeout_s,
                )
                if capacity == 0 and not waiting_for_workers:
                    jaxns_logger.warning(
                        "Distributed run has no live workers; scientific "
                        "tasks remain queued until capacity recovers."
                    )
                    waiting_for_workers = True
                elif capacity > 0 and waiting_for_workers:
                    jaxns_logger.info(
                        "Distributed worker capacity recovered; resuming "
                        "queued scientific tasks."
                    )
                    waiting_for_workers = False
                continue
            if waiting_for_workers:
                jaxns_logger.info(
                    "Distributed worker capacity recovered; received queued "
                    "scientific results."
                )
            return group

    def _wait_for_worker_capacity(
            self,
            client,
            session_id: str,
    ) -> None:
        """Wait without a scientific deadline until any compatible lane joins."""
        waiting_logged = False
        while True:
            capacity = client.capacity(
                session_id,
                timeout_s=self.receive_timeout_s,
            )
            if capacity > 0:
                if waiting_logged:
                    jaxns_logger.info(
                        "Distributed worker capacity recovered; resuming "
                        "scientific scheduling."
                    )
                return
            if not waiting_logged:
                jaxns_logger.warning(
                    "Distributed run has no live workers; scientific state "
                    "is unchanged while waiting for capacity."
                )
                waiting_logged = True
            time.sleep(min(1.0, max(0.1, self.receive_timeout_s)))

    def run(
            self,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> DistributedState:
        """Run until the configured expectation-based goal is met.

        Args:
            key: Random key used only when starting a new run.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between drained depth-boundary
                checkpoints. The final state is always saved when changed.

        Returns:
            A drained distributed state suitable for results or resumption.
        """

        def default_goal(state: State) -> bool:
            if int(state.goal_loop_iter) == 0:
                return False
            done, _ = state.compute_termination_register().is_done(
                self.termination_condition
            )
            return bool(done)

        return self.run_until_goal(
            default_goal,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> DistributedState:
        """Initialise and run the asynchronous Python goal/depth loops.

        A valid checkpoint in ``checkpoint_dir`` takes precedence over
        ``key`` and resumes its stored random stream and pending work. JAXNS
        verifies storage integrity, while compatible model, sampler, and run
        configuration remain the caller's responsibility.

        Args:
            goal_cond: User goal evaluated only on drained immutable states.
            depth_cond: Expectation-based boundary for each allocation epoch.
            key: Run random key; a deterministic default is used when absent.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between drained depth-boundary
                checkpoints. The default is one hour.

        Returns:
            A drained distributed checkpoint suitable for results or
            resumption.
        """
        if checkpoint_dir is not None:
            with CheckpointManager[DistributedState](
                checkpoint_dir,
                checkpoint_cadence,
            ) as checkpoint_manager:
                distributed = checkpoint_manager.load()
                if distributed is None:
                    completed = self._run_until_goal(
                        goal_cond,
                        depth_cond=depth_cond,
                        key=key,
                        checkpoint_manager=checkpoint_manager,
                    )
                else:
                    completed = self._resume_until_goal(
                        distributed,
                        goal_cond,
                        depth_cond=depth_cond,
                        checkpoint_manager=checkpoint_manager,
                    )
                checkpoint_manager.save_if_changed(completed)
                return completed
        return self._run_until_goal(
            goal_cond,
            depth_cond=depth_cond,
            key=key,
            checkpoint_manager=None,
        )

    def _run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            *,
            depth_cond: TerminationCondition | None,
            key: PRNGKey | None,
            checkpoint_manager: CheckpointManager[DistributedState] | None,
    ) -> DistributedState:
        """Start a new distributed session after checkpoint resolution."""
        from jaxns.runtime_client import SupervisorClient

        if depth_cond is None:
            depth_cond = self.termination_condition
        session_id = uuid4().hex
        session = WorkerSession(
            model=self.model,
            sampler=self.sampler,
            args=self.args,
            params=self.params,
        )
        distributed = None
        with SupervisorClient.from_port(self.coordinator_port) as client:
            try:
                client.register(session_id, session)
                # A new run retains one registration from root likelihoods
                # through constrained sampling. This avoids serialising the
                # same model twice and preserves the worker executable cache.
                distributed = self._initialise_connected(
                    client,
                    session_id,
                    key,
                )
                completed = self._run_connected(
                    client,
                    distributed,
                    goal_cond,
                    depth_cond,
                    checkpoint_manager,
                )
                distributed = completed
                client.release(session_id)
                return completed
            except DistributedRunError as exc:
                if checkpoint_manager is not None:
                    checkpoint_manager.save_if_changed(exc.checkpoint)
                raise
            except Exception as exc:
                if distributed is None:
                    raise RuntimeError(
                        f"Distributed initialization failed: {exc}"
                    ) from exc
                error = DistributedRunError(
                    f"Distributed execution failed: {exc}",
                    distributed,
                )
                if checkpoint_manager is not None:
                    checkpoint_manager.save_if_changed(distributed)
                raise error from exc

    def resume_until_goal(
            self,
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> DistributedState:
        """Resume an immutable checkpoint, including retry-stable tasks.

        If ``checkpoint_dir`` already contains a valid checkpoint, its state
        takes precedence over the explicit ``distributed`` state.

        Args:
            distributed: Explicit state used when no checkpoint exists.
            goal_cond: User goal evaluated only on drained immutable states.
            depth_cond: Expectation-based boundary for each allocation epoch.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between drained depth-boundary
                checkpoints. The default is one hour.

        Returns:
            A drained distributed state suitable for results or resumption.

        Raises:
            DistributedRunError: If runtime work fails. Its ``checkpoint``
                field contains every reservation and task created before the
                failure and can be passed back to this method.
        """
        if checkpoint_dir is not None:
            with CheckpointManager[DistributedState](
                checkpoint_dir,
                checkpoint_cadence,
            ) as checkpoint_manager:
                restored = checkpoint_manager.load()
                if restored is not None:
                    distributed = restored
                completed = self._resume_until_goal(
                    distributed,
                    goal_cond,
                    depth_cond=depth_cond,
                    checkpoint_manager=checkpoint_manager,
                )
                checkpoint_manager.save_if_changed(completed)
                return completed
        return self._resume_until_goal(
            distributed,
            goal_cond,
            depth_cond=depth_cond,
            checkpoint_manager=None,
        )

    def _resume_until_goal(
            self,
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            *,
            depth_cond: TerminationCondition | None,
            checkpoint_manager: CheckpointManager[DistributedState] | None,
    ) -> DistributedState:
        """Reconnect one resolved immutable distributed state."""
        from jaxns.runtime_client import SupervisorClient

        if depth_cond is None:
            depth_cond = self.termination_condition
        session = WorkerSession(
            model=self.model,
            sampler=self.sampler,
            args=distributed.state.args,
            params=distributed.state.params,
        )
        with SupervisorClient.from_port(self.coordinator_port) as client:
            try:
                capacities = client.register(distributed.session_id, session)
                if not capacities:
                    raise RuntimeError("The supervisor has no ready workers.")
                if distributed.pending:
                    client.submit_many(
                        distributed.session_id,
                        tuple(
                            (pending.task_id, pending.request)
                            for pending in distributed.pending
                        ),
                    )
                completed = self._run_connected(
                    client,
                    distributed,
                    goal_cond,
                    depth_cond,
                    checkpoint_manager,
                )
                distributed = completed
                client.release(completed.session_id)
                return completed
            except DistributedRunError as exc:
                if checkpoint_manager is not None:
                    checkpoint_manager.save_if_changed(exc.checkpoint)
                raise
            except Exception as exc:
                error = DistributedRunError(
                    f"Distributed execution failed: {exc}",
                    distributed,
                )
                if checkpoint_manager is not None:
                    checkpoint_manager.save_if_changed(distributed)
                raise error from exc

    def _run_connected(
            self,
            client: SupervisorClient,
            distributed: DistributedState,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition,
            checkpoint_manager: CheckpointManager[DistributedState] | None,
    ) -> DistributedState:
        completed_tasks: dict[
            int,
            tuple[ConstrainedSampleBatch, int],
        ] = {}
        next_completion_group = 0
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
                        completed_group = self._receive_group_waiting_for_workers(
                            client,
                            distributed.session_id,
                        )
                    except Exception as exc:
                        raise DistributedRunError(
                            f"Waiting for distributed task failed: {exc}",
                            distributed,
                        ) from exc
                    completion_group = next_completion_group
                    next_completion_group += 1
                    for task_id, batch in completed_group:
                        if not any(
                            task.task_id == task_id
                            for task in distributed.pending
                        ):
                            # A replay for an already committed task is safe to
                            # release because its state mutation already happened.
                            client.acknowledge(
                                distributed.session_id,
                                task_id,
                            )
                            continue
                        completed_tasks[task_id] = (
                            batch,
                            completion_group,
                        )

                # Results from one worker vmap arrive atomically. Commit only
                # the contiguous IDs in that assignment before refill. A
                # faster sibling worker may already have returned later IDs,
                # but network timing must not merge two scientific boundaries.
                commit_group = completed_tasks[pending.task_id][1]
                while distributed.pending:
                    pending = min(
                        distributed.pending,
                        key=lambda task: task.task_id,
                    )
                    if pending.task_id not in completed_tasks:
                        break
                    batch, completion_group = completed_tasks[pending.task_id]
                    if completion_group != commit_group:
                        break
                    completed_tasks.pop(pending.task_id)
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
                # At this point every task from the depth is committed and no
                # provisional reservation is present. The checkpoint therefore
                # exposes exactly the same immutable state seen by goal_cond.
                if checkpoint_manager is not None:
                    checkpoint_manager.maybe_save(distributed)
                continue
            if bool(status.has_work):
                # A pool may temporarily have no live lanes while an operator
                # replaces a node. Zero capacity is not a scientific error:
                # keep this exact depth open until a worker recovers or joins.
                try:
                    self._wait_for_worker_capacity(
                        client,
                        distributed.session_id,
                    )
                except Exception as exc:
                    raise DistributedRunError(
                        f"Observing distributed capacity failed: {exc}",
                        distributed,
                    ) from exc
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
        # Fill every currently measured worker lane without imposing a
        # completion barrier. The capacity bound also prevents scheduling
        # policy from changing the number of scientific observations merely
        # because a large allocation gap is visible to the coordinator.
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
            try:
                client.submit_many(
                    distributed.session_id,
                    tuple(
                        (task.task_id, task.request)
                        for task in tasks
                    ),
                )
            except Exception as exc:
                raise DistributedRunError(
                    "Submitting distributed tasks failed: "
                    f"{exc}",
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
