from __future__ import annotations

import dataclasses
import hashlib
import pickle
import re
import threading
import time
from collections.abc import Iterable
from typing import TypeVar

import jax
from jaxctx import CtxParams

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.samples import SeedPoint


_WORKER_SPEC_FORMAT_MESSAGE = (
    "Worker spec must have form "
    "'device_type:device_ids:num_workers_per_device', e.g. 'cpu:*:5'."
)
_WORKER_SPEC_PATTERN = re.compile(
    r"^(?P<device_type>[A-Za-z][A-Za-z0-9_]*):"
    r"(?P<device_ids>\*|[0-9]+(?:,[0-9]+)*):"
    r"(?P<count>[0-9]+)$"
)
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "gpu"})
SENTINEL_PARENT_IDX = -1

T = TypeVar("T")
_MISSING = object()


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerSpec:
    device_type: str
    device_ids: tuple[str, ...]
    workers_per_device: int

    @property
    def num_workers_per_device(self) -> int:
        return self.workers_per_device


@dataclasses.dataclass(frozen=True, slots=True)
class ComputeSector:
    """Coordinator view of a worker pool advertised by a host."""

    sector_id: str
    device_type: str
    device_id: str
    num_workers: int
    source_worker_spec: WorkerSpec


@dataclasses.dataclass(frozen=True, slots=True)
class ModelProblem:
    model: object
    args: tuple[object, ...]
    params: object | None
    collect_phantoms: bool


@dataclasses.dataclass(frozen=True, slots=True)
class SerializedModelProblem:
    model_bytes: bytes
    args_bytes: bytes
    params_bytes: bytes
    collect_phantoms: bool

    @classmethod
    def from_problem(
            cls,
            *,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
            collect_phantoms: bool = False,
    ) -> "SerializedModelProblem":
        return cls(
            model_bytes=serialize_model(model),
            args_bytes=serialize_args(args),
            params_bytes=serialize_params(params),
            collect_phantoms=collect_phantoms,
        )

    def deserialize_problem(self) -> ModelProblem:
        return ModelProblem(
            model=deserialize_model(self.model_bytes),
            args=deserialize_args(self.args_bytes),
            params=deserialize_params(self.params_bytes),
            collect_phantoms=self.collect_phantoms,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeCompileIdentity:
    serialized_model_digest: str
    serialized_args_digest: str
    serialized_params_digest: str
    collect_phantoms: bool
    sampler_type: str
    serialized_sampler_digest: str
    worker_device_types: tuple[str, ...]
    identity_digest: str


@dataclasses.dataclass(frozen=True, slots=True)
class RunnerIdentity:
    runner_id: str
    client_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class TaskIdentity:
    task_id: str
    runner_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptIdentity:
    attempt_id: str
    task_id: str
    attempt_number: int


@dataclasses.dataclass(frozen=True, slots=True)
class TransportIdentity:
    transport_id: str
    attempt_id: str
    delivery_number: int


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeTaskMetadata:
    task_id: str
    runner_id: str
    attempt_id: str
    requested_parent_idx: int | None
    effective_parent_idx: int = SENTINEL_PARENT_IDX
    effective_log_L_constraint: float | None = None
    seed_id: str | None = None
    phantom_cluster_id: str | None = None
    transport_id: str | None = None

    @property
    def effective_strict_contour(self) -> float | None:
        return self.effective_log_L_constraint


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerResultIdentity:
    task_id: str
    attempt_id: str
    transport_id: str
    worker_id: str | None = None
    sector_id: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerResult:
    identity: WorkerResultIdentity
    payload: object | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class LocalWorkerAssignment:
    worker_id: str
    sector_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class SerializedWorkerTask:
    serialized_problem: SerializedModelProblem
    sampler_bytes: bytes
    key_bytes: bytes
    log_L_constraint_bytes: bytes
    seed_point_bytes: bytes
    adaptation_context_bytes: bytes | None = None
    runtime_compile_identity: RuntimeCompileIdentity | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerRuntimePayload:
    problem: ModelProblem
    sampler: object


@dataclasses.dataclass(frozen=True, slots=True)
class _WorkerTaskExecutionStats:
    output: object
    sampler_loop_mode: str
    dispatch_latency_seconds: float
    payload_cache_latency_seconds: float
    sampler_execution_latency_seconds: float


@dataclasses.dataclass(slots=True)
class CoordinatorDispatchRecord:
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    requested_parent_idx: int
    effective_parent_idx: int
    accepted_parent_idx: int
    in_flight_parent_target: int
    status: str
    worker_id: str
    sector_id: str
    identity_owner: str
    client_id: str
    runtime_compile_identity: RuntimeCompileIdentity
    serialized_problem: SerializedModelProblem
    dispatch_sequence: int = -1
    sampler_loop_mode: str = "python"
    dispatch_latency_seconds: float = 0.0
    payload_cache_latency_seconds: float = 0.0
    sampler_execution_latency_seconds: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class CoordinatorLifecycleRecord:
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    attempt_number: int
    delivery_number: int
    requested_parent_idx: int
    effective_parent_idx: int
    accepted_parent_idx: int
    in_flight_parent_target: int
    effective_log_L_constraint: float
    accepted_log_L_constraint: float
    seed_id: str | None
    phantom_cluster_id: str | None
    status: str
    worker_id: str
    sector_id: str
    identity_owner: str
    client_id: str
    runtime_compile_identity: RuntimeCompileIdentity
    serialized_problem: SerializedModelProblem | None = None
    error: str | None = None
    reason: str | None = None
    current_parent_idx: int | None = None
    current_effective_log_L_constraint: float | None = None
    dispatch_sequence: int = -1
    sampler_loop_mode: str = "python"
    dispatch_latency_seconds: float = 0.0
    payload_cache_latency_seconds: float = 0.0
    sampler_execution_latency_seconds: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class _LifecycleRecordMetadata:
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    attempt_number: int
    delivery_number: int
    requested_parent_idx: int
    effective_parent_idx: int
    accepted_parent_idx: int
    in_flight_parent_target: int
    effective_log_L_constraint: float
    accepted_log_L_constraint: float
    seed_id: str | None
    phantom_cluster_id: str | None
    worker_id: str
    sector_id: str
    client_id: str
    runtime_compile_identity: RuntimeCompileIdentity
    serialized_problem: SerializedModelProblem | None
    sampler_loop_mode: str = "python"
    dispatch_latency_seconds: float = 0.0
    payload_cache_latency_seconds: float = 0.0
    sampler_execution_latency_seconds: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    task_id: str
    accepted: bool
    reason: str
    accepted_identity: WorkerResultIdentity | None


class AcceptanceLedger:
    """Runner-owned idempotence guard keyed by statistical task id."""

    def __init__(self) -> None:
        self._accepted_by_task_id: dict[str, WorkerResultIdentity] = {}
        self._lock = threading.Lock()

    def accept(
            self,
            result_identity: WorkerResultIdentity,
    ) -> AcceptanceDecision:
        with self._lock:
            accepted_identity = self._accepted_by_task_id.get(
                result_identity.task_id
            )
            if accepted_identity is not None:
                return AcceptanceDecision(
                    task_id=result_identity.task_id,
                    accepted=False,
                    reason="duplicate_task_result",
                    accepted_identity=accepted_identity,
                )
            self._accepted_by_task_id[
                result_identity.task_id
            ] = result_identity
            return AcceptanceDecision(
                task_id=result_identity.task_id,
                accepted=True,
                reason="accepted",
                accepted_identity=result_identity,
            )

    def has_accepted(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._accepted_by_task_id

    @property
    def accepted_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._accepted_by_task_id)


@dataclasses.dataclass(slots=True)
class LocalLoadBalancerState:
    """In-process coordinator state for the first runtime slice."""

    address: str = "local"
    compute_sectors: dict[str, ComputeSector] = dataclasses.field(
        default_factory=dict
    )
    coordinator_dispatch_records: list[object] = (
        dataclasses.field(default_factory=list)
    )
    _client_counter: int = 0
    _runner_counter: int = 0
    _task_counter: int = 0
    _attempt_counter: int = 0
    _transport_counter: int = 0
    _sector_counter: int = 0
    _worker_assignment_counter: int = 0
    _attempt_numbers_by_task_id: dict[str, int] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    _delivery_numbers_by_attempt_id: dict[str, int] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    _active_client_ids: set[str] = dataclasses.field(
        default_factory=set,
        repr=False,
    )
    _sector_ids_by_client_id: dict[str, list[str]] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    _worker_runtime_cache: dict[str, WorkerRuntimePayload] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    shutdown_event: threading.Event = dataclasses.field(
        default_factory=threading.Event,
        repr=False,
        compare=False,
    )
    _lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )
    _shutdown_condition: threading.Condition = dataclasses.field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._shutdown_condition = threading.Condition(self._lock)

    def allocate_client_id(self) -> str:
        with self._lock:
            self._client_counter += 1
            client_id = f"client-{self._client_counter:06d}"
            self._active_client_ids.add(client_id)
            return client_id

    def unregister_client_id(self, client_id: str) -> bool:
        with self._shutdown_condition:
            self.unregister_compute_sectors_for_client(client_id)
            self._active_client_ids.discard(client_id)
            no_active_clients = len(self._active_client_ids) == 0
            if no_active_clients:
                self.clear_worker_runtime_cache()
            self._shutdown_condition.notify_all()
            return no_active_clients

    def allocate_runner_identity(self, client_id: str) -> RunnerIdentity:
        with self._lock:
            self._runner_counter += 1
            runner_id = f"runner-{self._runner_counter:06d}"
        return RunnerIdentity(runner_id=runner_id, client_id=client_id)

    def allocate_task_identity(self, runner_id: str) -> TaskIdentity:
        with self._lock:
            self._task_counter += 1
            task_id = f"task-{self._task_counter:06d}"
        return TaskIdentity(task_id=task_id, runner_id=runner_id)

    def allocate_attempt_identity(self, task_id: str) -> AttemptIdentity:
        with self._lock:
            self._attempt_counter += 1
            attempt_id = f"attempt-{self._attempt_counter:06d}"
            attempt_number = (
                self._attempt_numbers_by_task_id.get(task_id, 0) + 1
            )
            self._attempt_numbers_by_task_id[task_id] = attempt_number
        return AttemptIdentity(
            attempt_id=attempt_id,
            task_id=task_id,
            attempt_number=attempt_number,
        )

    def allocate_transport_identity(
            self,
            attempt_id: str,
    ) -> TransportIdentity:
        with self._lock:
            self._transport_counter += 1
            transport_id = f"transport-{self._transport_counter:06d}"
            delivery_number = (
                self._delivery_numbers_by_attempt_id.get(attempt_id, 0) + 1
            )
            self._delivery_numbers_by_attempt_id[attempt_id] = delivery_number
        return TransportIdentity(
            transport_id=transport_id,
            attempt_id=attempt_id,
            delivery_number=delivery_number,
        )

    def register_compute_sectors(
            self,
            client_id: str,
            worker_specs: Iterable[WorkerSpec],
    ) -> list[ComputeSector]:
        registered: list[ComputeSector] = []
        with self._lock:
            for worker_spec in worker_specs:
                for device_id in worker_spec.device_ids:
                    self._sector_counter += 1
                    sector_id = f"sector-{self._sector_counter:06d}"
                    sector = ComputeSector(
                        sector_id=sector_id,
                        device_type=worker_spec.device_type,
                        device_id=device_id,
                        num_workers=worker_spec.workers_per_device,
                        source_worker_spec=worker_spec,
                    )
                    self.compute_sectors[sector_id] = sector
                    self._sector_ids_by_client_id.setdefault(
                        client_id,
                        [],
                    ).append(sector_id)
                    registered.append(sector)
        return registered

    def unregister_compute_sectors_for_client(
            self,
            client_id: str,
    ) -> None:
        with self._lock:
            sector_ids = self._sector_ids_by_client_id.pop(client_id, ())
            for sector_id in sector_ids:
                self.compute_sectors.pop(sector_id, None)

    def request_shutdown(self) -> None:
        with self._shutdown_condition:
            self.shutdown_event.set()
            self._shutdown_condition.notify_all()

    def wait_until_shutdown(
            self,
            client_id: str,
            timeout: float | None = None,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._shutdown_condition:
            while True:
                if self.shutdown_event.is_set():
                    return True
                if client_id not in self._active_client_ids:
                    return True
                if deadline is None:
                    self._shutdown_condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._shutdown_condition.wait(timeout=remaining)

    def allocate_worker_assignment(self) -> LocalWorkerAssignment:
        with self._lock:
            sectors = tuple(self.compute_sectors.values())
            if not sectors:
                raise RuntimeError(
                    "No compute sectors are registered with the load balancer."
                )
            self._worker_assignment_counter += 1
            dispatch_idx = self._worker_assignment_counter - 1
            sector = sectors[dispatch_idx % len(sectors)]
            worker_ordinal = (dispatch_idx % sector.num_workers) + 1
            return LocalWorkerAssignment(
                worker_id=(
                    f"worker-{sector.sector_id}-{worker_ordinal:06d}"
                ),
                sector_id=sector.sector_id,
            )

    def worker_device_types(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    {
                        sector.device_type
                        for sector in self.compute_sectors.values()
                    }
                )
            )

    def worker_runtime_payload(
            self,
            *,
            task: SerializedWorkerTask,
            runtime_compile_identity: RuntimeCompileIdentity,
    ) -> WorkerRuntimePayload:
        cache_key = runtime_compile_identity.identity_digest
        with self._lock:
            payload = self._worker_runtime_cache.get(cache_key)
        if payload is not None:
            return payload

        payload = _deserialize_worker_runtime_payload(task)
        with self._lock:
            existing = self._worker_runtime_cache.get(cache_key)
            if existing is not None:
                return existing
            self._worker_runtime_cache[cache_key] = payload
            return payload

    def clear_worker_runtime_cache(self) -> None:
        with self._lock:
            self._worker_runtime_cache.clear()
        jax.clear_caches()

    def worker_runtime_cache_size(self) -> int:
        with self._lock:
            return len(self._worker_runtime_cache)

    def record_dispatch(self, record: object) -> object:
        with self._lock:
            sequence = len(self.coordinator_dispatch_records)
            if dataclasses.is_dataclass(record):
                try:
                    setattr(record, "dispatch_sequence", sequence)
                except (AttributeError, TypeError):
                    try:
                        record = dataclasses.replace(
                            record,
                            dispatch_sequence=sequence,
                        )
                    except (TypeError, ValueError):
                        pass
            self.coordinator_dispatch_records.append(record)
        return record


_LOCAL_LOAD_BALANCER_REGISTRY_LOCK = threading.Lock()
_LOCAL_LOAD_BALANCER_REGISTRY: dict[str, LocalLoadBalancerState] = {}


def _register_load_balancer_client(
        address: str,
) -> tuple[LocalLoadBalancerState, str]:
    with _LOCAL_LOAD_BALANCER_REGISTRY_LOCK:
        state = _LOCAL_LOAD_BALANCER_REGISTRY.get(address)
        if state is None:
            state = LocalLoadBalancerState(address=address)
            _LOCAL_LOAD_BALANCER_REGISTRY[address] = state
        client_id = state.allocate_client_id()
        return state, client_id


def _release_load_balancer_state(
        *,
        address: str,
        state: LocalLoadBalancerState,
        client_id: str,
) -> None:
    with _LOCAL_LOAD_BALANCER_REGISTRY_LOCK:
        no_active_clients = state.unregister_client_id(client_id)
        if _LOCAL_LOAD_BALANCER_REGISTRY.get(address) is state:
            if no_active_clients:
                del _LOCAL_LOAD_BALANCER_REGISTRY[address]


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pickle_payload(value: T) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_payload(payload: bytes) -> T:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    return pickle.loads(payload)


def pickle_round_trip(value: T) -> T:
    return unpickle_payload(pickle_payload(value))


def serialize_model(model: T) -> bytes:
    return pickle_payload(model)


def deserialize_model(payload: bytes) -> T:
    return unpickle_payload(payload)


def round_trip_model(model: T) -> T:
    return deserialize_model(serialize_model(model))


def serialize_args(args: tuple[object, ...]) -> bytes:
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple of positional arguments.")
    return pickle_payload(args)


def deserialize_args(payload: bytes) -> tuple[object, ...]:
    args = unpickle_payload(payload)
    if not isinstance(args, tuple):
        raise TypeError("deserialized args payload must contain a tuple.")
    return args


def round_trip_args(args: tuple[object, ...]) -> tuple[object, ...]:
    return deserialize_args(serialize_args(args))


def serialize_params(params: T | None) -> bytes:
    return pickle_payload(params)


def deserialize_params(payload: bytes) -> T | None:
    return unpickle_payload(payload)


def round_trip_params(params: T | None) -> T | None:
    return deserialize_params(serialize_params(params))


def serialize_sampler(sampler: T) -> bytes:
    return pickle_payload(sampler)


def deserialize_sampler(payload: bytes) -> T:
    return unpickle_payload(payload)


def _sampler_type_name(sampler: object) -> str:
    sampler_type = type(sampler)
    return f"{sampler_type.__module__}.{sampler_type.__qualname__}"


def _build_runtime_compile_identity(
        *,
        serialized_problem: SerializedModelProblem,
        sampler: object,
        sampler_bytes: bytes | None = None,
        worker_device_types: tuple[str, ...],
) -> RuntimeCompileIdentity:
    if sampler_bytes is None:
        sampler_bytes = serialize_sampler(sampler)
    serialized_model_digest = _digest_bytes(serialized_problem.model_bytes)
    serialized_args_digest = _digest_bytes(serialized_problem.args_bytes)
    serialized_params_digest = _digest_bytes(serialized_problem.params_bytes)
    serialized_sampler_digest = _digest_bytes(sampler_bytes)
    sampler_type = _sampler_type_name(sampler)
    identity_parts = (
        serialized_model_digest,
        serialized_args_digest,
        serialized_params_digest,
        bool(serialized_problem.collect_phantoms),
        sampler_type,
        serialized_sampler_digest,
        tuple(worker_device_types),
    )
    identity_digest = _digest_bytes(pickle_payload(identity_parts))
    return RuntimeCompileIdentity(
        serialized_model_digest=serialized_model_digest,
        serialized_args_digest=serialized_args_digest,
        serialized_params_digest=serialized_params_digest,
        collect_phantoms=bool(serialized_problem.collect_phantoms),
        sampler_type=sampler_type,
        serialized_sampler_digest=serialized_sampler_digest,
        worker_device_types=tuple(worker_device_types),
        identity_digest=identity_digest,
    )


def _inject_problem_model(sampler: object, problem: ModelProblem) -> None:
    if not hasattr(sampler, "model"):
        return
    try:
        sampler.model = problem.model
    except dataclasses.FrozenInstanceError:
        object.__setattr__(sampler, "model", problem.model)


def _deserialize_worker_runtime_payload(
        task: SerializedWorkerTask,
) -> WorkerRuntimePayload:
    problem = task.serialized_problem.deserialize_problem()
    sampler = deserialize_sampler(task.sampler_bytes)
    _inject_problem_model(sampler, problem)
    return WorkerRuntimePayload(problem=problem, sampler=sampler)


def _nonnegative_elapsed_seconds(start: float, end: float) -> float:
    return max(float(end - start), 0.0)


def _execute_serialized_worker_task(
        task: SerializedWorkerTask,
        *,
        runtime_lb_state: LocalLoadBalancerState | None = None,
):
    return _execute_serialized_worker_task_with_stats(
        task,
        runtime_lb_state=runtime_lb_state,
    ).output


def _execute_serialized_worker_task_with_stats(
        task: SerializedWorkerTask,
        *,
        runtime_lb_state: LocalLoadBalancerState | None = None,
) -> _WorkerTaskExecutionStats:
    dispatch_start = time.perf_counter()
    payload_start = dispatch_start
    if (
            runtime_lb_state is not None
            and task.runtime_compile_identity is not None
    ):
        payload = runtime_lb_state.worker_runtime_payload(
            task=task,
            runtime_compile_identity=task.runtime_compile_identity,
        )
    else:
        payload = _deserialize_worker_runtime_payload(task)
    payload_end = time.perf_counter()
    problem = payload.problem
    sampler = payload.sampler
    key = unpickle_payload(task.key_bytes)
    log_L_constraint = unpickle_payload(task.log_L_constraint_bytes)
    seed_point = unpickle_payload(task.seed_point_bytes)
    adaptation_context = (
        None
        if task.adaptation_context_bytes is None
        else unpickle_payload(task.adaptation_context_bytes)
    )
    sampler_loop_mode = "python"
    sampler_start = time.perf_counter()
    if isinstance(sampler, UniDimSliceSampler):
        output = sampler.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=problem.args,
            params=problem.params,
            adaptation_context={
                "force_python_loop": True,
                "sampler_loop_mode": sampler_loop_mode,
                "direction_adaptation_context": adaptation_context,
            },
        )
    elif adaptation_context is not None:
        output = sampler.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=problem.args,
            params=problem.params,
            adaptation_context=adaptation_context,
        )
    else:
        output = sampler.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=problem.args,
            params=problem.params,
        )
    sampler_end = time.perf_counter()
    dispatch_end = sampler_end
    return _WorkerTaskExecutionStats(
        output=output,
        sampler_loop_mode=sampler_loop_mode,
        dispatch_latency_seconds=_nonnegative_elapsed_seconds(
            dispatch_start,
            dispatch_end,
        ),
        payload_cache_latency_seconds=_nonnegative_elapsed_seconds(
            payload_start,
            payload_end,
        ),
        sampler_execution_latency_seconds=_nonnegative_elapsed_seconds(
            sampler_start,
            sampler_end,
        ),
    )


def _append_dispatch_trace(dispatch_trace: object, record: object) -> None:
    if hasattr(dispatch_trace, "record_dispatch"):
        dispatch_trace.record_dispatch(record)
        return
    if hasattr(dispatch_trace, "append"):
        dispatch_trace.append(record)
        return
    if hasattr(dispatch_trace, "records"):
        dispatch_trace.records.append(record)
        return
    raise TypeError(
        "dispatch_trace must provide record_dispatch(record), append(record), "
        "or a records list."
    )


def parse_worker_spec(spec: str) -> WorkerSpec:
    if not isinstance(spec, str):
        raise ValueError(_WORKER_SPEC_FORMAT_MESSAGE)
    if spec != spec.strip():
        raise ValueError(_WORKER_SPEC_FORMAT_MESSAGE)
    match = _WORKER_SPEC_PATTERN.match(spec)
    if match is None:
        raise ValueError(_WORKER_SPEC_FORMAT_MESSAGE)
    count = int(match.group("count"))
    if count <= 0:
        raise ValueError("num_workers_per_device must be positive.")
    device_ids_raw = match.group("device_ids")
    device_ids = (
        ("*",)
        if device_ids_raw == "*"
        else tuple(device_ids_raw.split(","))
    )
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("Worker spec device_ids must not contain duplicates.")
    device_type = match.group("device_type").lower()
    if device_type not in _SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"Unsupported worker device type {device_type!r}.")
    return WorkerSpec(
        device_type=device_type,
        device_ids=device_ids,
        workers_per_device=count,
    )


@dataclasses.dataclass(slots=True)
class RuntimeNestedSampler(NestedSampler):
    runtime_lb_state: LocalLoadBalancerState | None = dataclasses.field(
        default=None,
        repr=False,
    )
    runtime_runner_identity: RunnerIdentity | None = None
    runtime_problem_payload: SerializedModelProblem | None = None
    runtime_compile_identity: RuntimeCompileIdentity | None = None
    runtime_sampler_bytes: bytes | None = dataclasses.field(
        default=None,
        repr=False,
    )
    runtime_acceptance_ledger: AcceptanceLedger = dataclasses.field(
        default_factory=AcceptanceLedger,
        repr=False,
    )
    coordinator_dispatch_records: list[object] = (
        dataclasses.field(default_factory=list, repr=False)
    )
    runtime_dispatch_trace: object | None = dataclasses.field(
        default=None,
        repr=False,
    )
    _runtime_terminal_dispatch_status: dict[
        tuple[str, str, str],
        str,
    ] = dataclasses.field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _runtime_record_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )
    _runtime_lifecycle_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        NestedSampler.__post_init__(self)
        if (
                self.runtime_compile_identity is None
                and self.runtime_lb_state is not None
                and self.runtime_problem_payload is not None
        ):
            sampler_bytes = serialize_sampler(self.sampler)
            self.runtime_compile_identity = _build_runtime_compile_identity(
                serialized_problem=self.runtime_problem_payload,
                sampler=self.sampler,
                sampler_bytes=sampler_bytes,
                worker_device_types=(
                    self.runtime_lb_state.worker_device_types()
                ),
            )
            self.runtime_sampler_bytes = sampler_bytes

    def _sample_constrained(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            *,
            requested_parent_idx: int,
            effective_parent_idx: int,
            accepted_parent_idx: int,
            adaptation_context=None,
    ):
        _, _, problem_payload = self._require_runtime_context()
        worker_task = SerializedWorkerTask(
            serialized_problem=problem_payload,
            sampler_bytes=self._require_runtime_sampler_bytes(),
            key_bytes=pickle_payload(key),
            log_L_constraint_bytes=pickle_payload(log_L_constraint),
            seed_point_bytes=pickle_payload(seed_point),
            adaptation_context_bytes=(
                None
                if adaptation_context is None
                else pickle_payload(adaptation_context)
            ),
            runtime_compile_identity=self._require_runtime_compile_identity(),
        )
        accepted_log_l = float(log_L_constraint)
        dispatch_record = self.prepare_runtime_dispatch(
            requested_parent_idx=int(requested_parent_idx),
            effective_parent_idx=int(effective_parent_idx),
            accepted_parent_idx=int(accepted_parent_idx),
            effective_log_L_constraint=accepted_log_l,
            accepted_log_L_constraint=accepted_log_l,
            seed_id="seed-runtime-worker",
            phantom_cluster_id="phantom-cluster-runtime-worker",
        )
        max_attempts = 2
        for attempt_idx in range(max_attempts):
            try:
                worker_execution = _execute_serialized_worker_task_with_stats(
                    worker_task,
                    runtime_lb_state=self.runtime_lb_state,
                )
            except Exception as error:
                failed_record = self.mark_runtime_dispatch_failed(
                    dispatch_record=dispatch_record,
                    error=error,
                )
                if attempt_idx + 1 >= max_attempts:
                    raise
                dispatch_record = self.retry_runtime_dispatch(
                    dispatch_record=failed_record,
                    reason="worker failure",
                )
                continue

            result_identity = WorkerResultIdentity(
                task_id=str(
                    self._dispatch_record_field(
                        dispatch_record,
                        "task_id",
                    )
                ),
                attempt_id=str(
                    self._dispatch_record_field(
                        dispatch_record,
                        "attempt_id",
                    )
                ),
                transport_id=str(
                    self._dispatch_record_field(
                        dispatch_record,
                        "transport_id",
                    )
                ),
                worker_id=str(
                    self._dispatch_record_field(
                        dispatch_record,
                        "worker_id",
                    )
                ),
                sector_id=str(
                    self._dispatch_record_field(
                        dispatch_record,
                        "sector_id",
                    )
                ),
            )
            self.complete_runtime_dispatch(
                dispatch_record=dispatch_record,
                result=WorkerResult(
                    identity=result_identity,
                    payload=worker_execution,
                ),
                current_parent_idx=int(accepted_parent_idx),
                current_effective_log_L_constraint=accepted_log_l,
            )
            return worker_execution.output

        raise RuntimeError("worker execution exhausted retry attempts.")

    def prepare_runtime_dispatch(
            self,
            *,
            requested_parent_idx: int,
            effective_parent_idx: int,
            accepted_parent_idx: int,
            effective_log_L_constraint: float,
            accepted_log_L_constraint: float,
            seed_id: str | None,
            phantom_cluster_id: str | None,
    ) -> CoordinatorLifecycleRecord:
        lb_state, runner_identity, problem_payload = (
            self._require_runtime_context()
        )
        task_identity = lb_state.allocate_task_identity(
            runner_identity.runner_id
        )
        attempt_identity = lb_state.allocate_attempt_identity(
            task_identity.task_id
        )
        transport_identity = lb_state.allocate_transport_identity(
            attempt_identity.attempt_id
        )
        assignment = lb_state.allocate_worker_assignment()
        metadata = _LifecycleRecordMetadata(
            runner_id=runner_identity.runner_id,
            task_id=task_identity.task_id,
            attempt_id=attempt_identity.attempt_id,
            transport_id=transport_identity.transport_id,
            attempt_number=attempt_identity.attempt_number,
            delivery_number=transport_identity.delivery_number,
            requested_parent_idx=int(requested_parent_idx),
            effective_parent_idx=int(effective_parent_idx),
            accepted_parent_idx=int(accepted_parent_idx),
            in_flight_parent_target=int(accepted_parent_idx),
            effective_log_L_constraint=float(effective_log_L_constraint),
            accepted_log_L_constraint=float(accepted_log_L_constraint),
            seed_id=seed_id,
            phantom_cluster_id=phantom_cluster_id,
            worker_id=assignment.worker_id,
            sector_id=assignment.sector_id,
            client_id=runner_identity.client_id,
            runtime_compile_identity=(
                self._require_runtime_compile_identity()
            ),
            serialized_problem=problem_payload,
        )
        return self._record_lifecycle_event(metadata, status="pending")

    def mark_runtime_dispatch_failed(
            self,
            *,
            dispatch_record: object,
            error: object,
    ) -> CoordinatorLifecycleRecord:
        metadata = self._lifecycle_metadata_from_record(dispatch_record)
        self._validate_lifecycle_record_owner(metadata)
        self._mark_dispatch_terminal(metadata, "failed")
        return self._record_lifecycle_event(
            metadata,
            status="failed",
            error=str(error),
        )

    def retry_runtime_dispatch(
            self,
            *,
            dispatch_record: object,
            reason: str,
    ) -> CoordinatorLifecycleRecord:
        lb_state, _, _ = self._require_runtime_context()
        metadata = self._lifecycle_metadata_from_record(dispatch_record)
        self._validate_lifecycle_record_owner(metadata)
        attempt_identity = lb_state.allocate_attempt_identity(metadata.task_id)
        transport_identity = lb_state.allocate_transport_identity(
            attempt_identity.attempt_id
        )
        assignment = lb_state.allocate_worker_assignment()
        retry_metadata = dataclasses.replace(
            metadata,
            attempt_id=attempt_identity.attempt_id,
            transport_id=transport_identity.transport_id,
            attempt_number=attempt_identity.attempt_number,
            delivery_number=transport_identity.delivery_number,
            worker_id=assignment.worker_id,
            sector_id=assignment.sector_id,
        )
        return self._record_lifecycle_event(
            retry_metadata,
            status="retried",
            reason=reason,
        )

    def revoke_runtime_dispatch(
            self,
            *,
            dispatch_record: object,
            reason: str,
    ) -> CoordinatorLifecycleRecord:
        metadata = self._lifecycle_metadata_from_record(dispatch_record)
        self._validate_lifecycle_record_owner(metadata)
        self._mark_dispatch_terminal(metadata, "revoked")
        return self._record_lifecycle_event(
            metadata,
            status="revoked",
            reason=reason,
        )

    def complete_runtime_dispatch(
            self,
            *,
            dispatch_record: object,
            result: WorkerResult,
            current_parent_idx: int,
            current_effective_log_L_constraint: float,
    ) -> CoordinatorLifecycleRecord:
        metadata = self._lifecycle_metadata_from_record(dispatch_record)
        self._validate_lifecycle_record_owner(metadata)
        metadata = self._metadata_with_worker_execution_stats(
            metadata,
            result.payload,
        )
        result_identity = result.identity
        mismatched_fields = []
        if result_identity.task_id != metadata.task_id:
            mismatched_fields.append("task_id")
        if result_identity.attempt_id != metadata.attempt_id:
            mismatched_fields.append("attempt_id")
        if result_identity.transport_id != metadata.transport_id:
            mismatched_fields.append("transport_id")
        if mismatched_fields:
            return self._record_lifecycle_event(
                metadata,
                status="mismatched_result_identity",
                reason=", ".join(mismatched_fields),
            )

        current_log_l = float(current_effective_log_L_constraint)
        if (
                int(current_parent_idx) != metadata.accepted_parent_idx
                or current_log_l != metadata.accepted_log_L_constraint
        ):
            return self._record_lifecycle_event(
                metadata,
                status="stale_parent_target",
                current_parent_idx=int(current_parent_idx),
                current_effective_log_L_constraint=current_log_l,
            )

        terminal_status = self._terminal_dispatch_status(metadata)
        if terminal_status is not None:
            return self._record_lifecycle_event(
                metadata,
                status=terminal_status,
            )

        decision = self.runtime_acceptance_ledger.accept(result_identity)
        if decision.accepted:
            status = "accepted"
        elif (
                decision.accepted_identity is not None
                and decision.accepted_identity.attempt_id
                == metadata.attempt_id
        ):
            status = "duplicate_task_result"
        else:
            status = "stale_task_result"
        return self._record_lifecycle_event(metadata, status=status)

    def _require_runtime_context(
            self,
    ) -> tuple[LocalLoadBalancerState, RunnerIdentity, SerializedModelProblem]:
        if self.runtime_lb_state is None:
            raise RuntimeError(
                "RuntimeNestedSampler has no load-balancer state."
            )
        if self.runtime_runner_identity is None:
            raise RuntimeError("RuntimeNestedSampler has no runner identity.")
        if self.runtime_problem_payload is None:
            raise RuntimeError("RuntimeNestedSampler has no problem payload.")
        return (
            self.runtime_lb_state,
            self.runtime_runner_identity,
            self.runtime_problem_payload,
        )

    def _require_runtime_compile_identity(self) -> RuntimeCompileIdentity:
        if self.runtime_compile_identity is None:
            raise RuntimeError(
                "RuntimeNestedSampler has no compile identity."
            )
        return self.runtime_compile_identity

    def _require_runtime_sampler_bytes(self) -> bytes:
        if self.runtime_sampler_bytes is None:
            self.runtime_sampler_bytes = serialize_sampler(self.sampler)
        return self.runtime_sampler_bytes

    def _validate_lifecycle_record_owner(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> None:
        _, runner_identity, _ = self._require_runtime_context()
        if (
                metadata.runner_id != runner_identity.runner_id
                or metadata.client_id != runner_identity.client_id
        ):
            raise ValueError(
                "dispatch record owner does not match this runtime runner"
            )

    def _record_lifecycle_event(
            self,
            metadata: _LifecycleRecordMetadata,
            *,
            status: str,
            error: str | None = None,
            reason: str | None = None,
            current_parent_idx: int | None = None,
            current_effective_log_L_constraint: float | None = None,
    ) -> CoordinatorLifecycleRecord:
        record = CoordinatorLifecycleRecord(
            runner_id=metadata.runner_id,
            task_id=metadata.task_id,
            attempt_id=metadata.attempt_id,
            transport_id=metadata.transport_id,
            attempt_number=metadata.attempt_number,
            delivery_number=metadata.delivery_number,
            requested_parent_idx=metadata.requested_parent_idx,
            effective_parent_idx=metadata.effective_parent_idx,
            accepted_parent_idx=metadata.accepted_parent_idx,
            in_flight_parent_target=metadata.in_flight_parent_target,
            effective_log_L_constraint=(
                metadata.effective_log_L_constraint
            ),
            accepted_log_L_constraint=metadata.accepted_log_L_constraint,
            seed_id=metadata.seed_id,
            phantom_cluster_id=metadata.phantom_cluster_id,
            status=status,
            worker_id=metadata.worker_id,
            sector_id=metadata.sector_id,
            identity_owner="coordinator",
            client_id=metadata.client_id,
            runtime_compile_identity=metadata.runtime_compile_identity,
            serialized_problem=metadata.serialized_problem,
            error=error,
            reason=reason,
            current_parent_idx=current_parent_idx,
            current_effective_log_L_constraint=(
                current_effective_log_L_constraint
            ),
            sampler_loop_mode=metadata.sampler_loop_mode,
            dispatch_latency_seconds=metadata.dispatch_latency_seconds,
            payload_cache_latency_seconds=(
                metadata.payload_cache_latency_seconds
            ),
            sampler_execution_latency_seconds=(
                metadata.sampler_execution_latency_seconds
            ),
        )
        self._record_dispatch(record)
        return record

    @staticmethod
    def _metadata_with_worker_execution_stats(
            metadata: _LifecycleRecordMetadata,
            payload: object,
    ) -> _LifecycleRecordMetadata:
        if not isinstance(payload, _WorkerTaskExecutionStats):
            return metadata
        return dataclasses.replace(
            metadata,
            sampler_loop_mode=payload.sampler_loop_mode,
            dispatch_latency_seconds=payload.dispatch_latency_seconds,
            payload_cache_latency_seconds=(
                payload.payload_cache_latency_seconds
            ),
            sampler_execution_latency_seconds=(
                payload.sampler_execution_latency_seconds
            ),
        )

    def _lifecycle_metadata_from_record(
            self,
            record: object,
    ) -> _LifecycleRecordMetadata:
        serialized_problem = self._optional_dispatch_record_field(
            record,
            "serialized_problem",
            default=self.runtime_problem_payload,
        )
        return _LifecycleRecordMetadata(
            runner_id=str(self._dispatch_record_field(record, "runner_id")),
            task_id=str(self._dispatch_record_field(record, "task_id")),
            attempt_id=str(self._dispatch_record_field(record, "attempt_id")),
            transport_id=str(
                self._dispatch_record_field(record, "transport_id")
            ),
            attempt_number=int(
                self._dispatch_record_field(record, "attempt_number")
            ),
            delivery_number=int(
                self._dispatch_record_field(record, "delivery_number")
            ),
            requested_parent_idx=int(
                self._dispatch_record_field(record, "requested_parent_idx")
            ),
            effective_parent_idx=int(
                self._dispatch_record_field(record, "effective_parent_idx")
            ),
            accepted_parent_idx=int(
                self._dispatch_record_field(record, "accepted_parent_idx")
            ),
            in_flight_parent_target=int(
                self._dispatch_record_field(
                    record,
                    "in_flight_parent_target",
                )
            ),
            effective_log_L_constraint=float(
                self._dispatch_record_field(
                    record,
                    "effective_log_L_constraint",
                )
            ),
            accepted_log_L_constraint=float(
                self._dispatch_record_field(
                    record,
                    "accepted_log_L_constraint",
                )
            ),
            seed_id=self._optional_str(
                self._dispatch_record_field(record, "seed_id")
            ),
            phantom_cluster_id=self._optional_str(
                self._dispatch_record_field(record, "phantom_cluster_id")
            ),
            worker_id=str(self._dispatch_record_field(record, "worker_id")),
            sector_id=str(self._dispatch_record_field(record, "sector_id")),
            client_id=str(
                self._optional_dispatch_record_field(
                    record,
                    "client_id",
                    default=(
                        self._require_runtime_context()[1].client_id
                    ),
                )
            ),
            runtime_compile_identity=(
                self._optional_dispatch_record_field(
                    record,
                    "runtime_compile_identity",
                    default=self._require_runtime_compile_identity(),
                )
            ),
            serialized_problem=serialized_problem,
            sampler_loop_mode=str(
                self._optional_dispatch_record_field(
                    record,
                    "sampler_loop_mode",
                    default="python",
                )
            ),
            dispatch_latency_seconds=float(
                self._optional_dispatch_record_field(
                    record,
                    "dispatch_latency_seconds",
                    default=0.0,
                )
            ),
            payload_cache_latency_seconds=float(
                self._optional_dispatch_record_field(
                    record,
                    "payload_cache_latency_seconds",
                    default=0.0,
                )
            ),
            sampler_execution_latency_seconds=float(
                self._optional_dispatch_record_field(
                    record,
                    "sampler_execution_latency_seconds",
                    default=0.0,
                )
            ),
        )

    def _mark_dispatch_terminal(
            self,
            metadata: _LifecycleRecordMetadata,
            status: str,
    ) -> None:
        with self._runtime_lifecycle_lock:
            self._runtime_terminal_dispatch_status[
                self._lifecycle_dispatch_key(metadata)
            ] = status

    def _terminal_dispatch_status(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> str | None:
        with self._runtime_lifecycle_lock:
            return self._runtime_terminal_dispatch_status.get(
                self._lifecycle_dispatch_key(metadata)
            )

    @staticmethod
    def _lifecycle_dispatch_key(
            metadata: _LifecycleRecordMetadata,
    ) -> tuple[str, str, str]:
        return metadata.task_id, metadata.attempt_id, metadata.transport_id

    @staticmethod
    def _dispatch_record_field(record: object, field_name: str) -> object:
        value = RuntimeNestedSampler._optional_dispatch_record_field(
            record,
            field_name,
        )
        if value is _MISSING:
            raise ValueError(
                f"Dispatch record is missing field {field_name!r}."
            )
        return value

    @staticmethod
    def _optional_dispatch_record_field(
            record: object,
            field_name: str,
            *,
            default: object = _MISSING,
    ) -> object:
        if isinstance(record, dict) and field_name in record:
            return record[field_name]
        if hasattr(record, field_name):
            return getattr(record, field_name)
        return default

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    def _record_dispatch(self, record: object) -> None:
        if self.runtime_lb_state is not None:
            record = self.runtime_lb_state.record_dispatch(record)
        with self._runtime_record_lock:
            self.coordinator_dispatch_records.append(record)
        if self.runtime_dispatch_trace is not None:
            _append_dispatch_trace(self.runtime_dispatch_trace, record)


class LoadBalancerClient:
    """Local v3 load-balancer facade matching the target run pattern.

    This is intentionally a small user-facing runtime boundary. It does not
    wrap the legacy p2p/fabric lease API. The local implementation records
    worker compute sectors and creates isolated `NestedSampler` runners
    in-process.
    """

    def __init__(self, address: str = "local", dispatch_trace=None):
        valid_address = isinstance(address, str) and (
            address == "local"
            or (
                address.startswith("tcp://")
                and len(address) > len("tcp://")
            )
        )
        if not valid_address:
            raise ValueError("address must be 'local' or a tcp:// endpoint.")
        self.address = address
        self.dispatch_trace = dispatch_trace
        self.worker_specs: list[WorkerSpec] = []
        self._state, self._client_id = _register_load_balancer_client(address)
        self._closed = False

    def __enter__(self) -> "LoadBalancerClient":
        if self._closed:
            self._state, self._client_id = _register_load_balancer_client(
                self.address
            )
            self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.shutdown()

    def add_workers(self, specs: Iterable[str]) -> list[WorkerSpec]:
        self._ensure_open()
        if isinstance(specs, str):
            raise ValueError(
                "specs must be an iterable of worker spec strings, "
                "not a single worker spec string."
            )
        try:
            iterator = iter(specs)
        except TypeError as e:
            raise ValueError(
                "specs must be an iterable of worker spec strings."
            ) from e
        parsed = [parse_worker_spec(spec) for spec in iterator]
        self.worker_specs.extend(parsed)
        self._state.register_compute_sectors(self._client_id, parsed)
        return parsed

    def get_nested_sampler(
            self,
            *,
            model: Model,
            args: tuple = (),
            params: CtxParams | None = None,
            collect_phantoms: bool = False,
            **kwargs,
    ) -> NestedSampler:
        self._ensure_open()
        runner_identity = self._state.allocate_runner_identity(self._client_id)
        problem_payload = SerializedModelProblem.from_problem(
            model=model,
            args=args,
            params=params,
            collect_phantoms=collect_phantoms,
        )
        runner = RuntimeNestedSampler(
            model=model,
            args=args,
            params=params,
            collect_phantom_samples=collect_phantoms,
            runtime_lb_state=self._state,
            runtime_runner_identity=runner_identity,
            runtime_problem_payload=problem_payload,
            runtime_acceptance_ledger=AcceptanceLedger(),
            runtime_dispatch_trace=self.dispatch_trace,
            **kwargs,
        )
        return runner

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def load_balancer_state(self) -> LocalLoadBalancerState:
        return self._state

    @property
    def compute_sectors(self) -> tuple[ComputeSector, ...]:
        return tuple(self._state.compute_sectors.values())

    @property
    def coordinator_dispatch_records(
            self,
    ) -> tuple[object, ...]:
        return tuple(self._state.coordinator_dispatch_records)

    def allocate_task_identity(self, runner_id: str) -> TaskIdentity:
        self._ensure_open()
        return self._state.allocate_task_identity(runner_id)

    def set_dispatch_trace(self, dispatch_trace) -> None:
        self.dispatch_trace = dispatch_trace

    def set_dispatch_ledger(self, dispatch_ledger) -> None:
        self.set_dispatch_trace(dispatch_ledger)

    def set_coordinator_dispatch_trace(self, dispatch_trace) -> None:
        self.set_dispatch_trace(dispatch_trace)

    def set_coordinator_dispatch_ledger(self, dispatch_ledger) -> None:
        self.set_dispatch_trace(dispatch_ledger)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        _release_load_balancer_state(
            address=self.address,
            state=self._state,
            client_id=self._client_id,
        )

    def request_shutdown(self) -> None:
        self._ensure_open()
        self._state.request_shutdown()

    def wait_until_shutdown(self, timeout: float | None = None) -> bool:
        return self._state.wait_until_shutdown(
            client_id=self._client_id,
            timeout=timeout,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LoadBalancerClient is closed.")
