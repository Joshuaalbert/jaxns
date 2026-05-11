from __future__ import annotations

import dataclasses
import hashlib
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jaxctx import CtxParams

from jaxns.allocation import ParentWork
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.core import _phantom_coordinates_like_state
from jaxns.diagnostics import LikelihoodDispatchDiagnostics as _BaseLikelihoodDispatchDiagnostics
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.samples import PhantomSamples
from jaxns.samples import Samples
from jaxns.samples import SeedPoint
from jaxns.state import State


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
class ShapeDtypeTree:
    tree_structure: tuple[object, ...]
    leaf_shapes: tuple[tuple[int, ...], ...]
    leaf_dtypes: tuple[str, ...]

    @classmethod
    def from_pytree(cls, value: object) -> "ShapeDtypeTree":
        return _shape_dtype_tree_from_pytree(value, tree_cls=cls)

    @classmethod
    def from_value(cls, value: object) -> "ShapeDtypeTree":
        return cls.from_pytree(value)

    @classmethod
    def from_tree(cls, value: object) -> "ShapeDtypeTree":
        return cls.from_pytree(value)

    def mismatch_error_type(
            self,
            other: "ShapeDtypeTree",
    ) -> str | None:
        if (
                self.tree_structure != other.tree_structure
                or len(self.leaf_shapes) != len(other.leaf_shapes)
        ):
            return "PytreeMismatch"
        if self.leaf_shapes != other.leaf_shapes:
            return "ShapeMismatch"
        if self.leaf_dtypes != other.leaf_dtypes:
            return "DtypeMismatch"
        return None


def _shape_tree_mismatch_error_type(
        left: object,
        right: object,
) -> str | None:
    if not isinstance(left, ShapeDtypeTree):
        return "MalformedUShapeTreeMetadata"
    if not isinstance(right, ShapeDtypeTree):
        return "MalformedUShapeTreeMetadata"
    try:
        return left.mismatch_error_type(right)
    except (AttributeError, TypeError, ValueError):
        return "MalformedUShapeTreeMetadata"


def _stable_pytree_structure(tree_def: jax.tree_util.PyTreeDef) -> tuple[object, ...]:
    node_data = tree_def.node_data()
    children = tuple(
        _stable_pytree_structure(child)
        for child in tree_def.children()
    )
    if node_data is None:
        node = ("leaf",)
    else:
        node = tuple(_stable_tree_metadata(item) for item in node_data)
    return node, children


def _stable_tree_metadata(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return "type", value.__module__, value.__qualname__
    if isinstance(value, (tuple, list)):
        return tuple(_stable_tree_metadata(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _stable_tree_metadata(key),
                    _stable_tree_metadata(item),
                )
                for key, item in value.items()
            )
        )
    return "repr", repr(value)


def _shape_dtype_tree_from_pytree(
        value: object,
        tree_cls: type[ShapeDtypeTree] = ShapeDtypeTree,
) -> ShapeDtypeTree:
    leaves, tree_def = jax.tree.flatten(value)
    leaf_shapes = []
    leaf_dtypes = []
    for leaf in leaves:
        array = np.asarray(leaf)
        leaf_shapes.append(tuple(int(dim) for dim in array.shape))
        leaf_dtypes.append(str(array.dtype))
    return tree_cls(
        tree_structure=_stable_pytree_structure(tree_def),
        leaf_shapes=tuple(leaf_shapes),
        leaf_dtypes=tuple(leaf_dtypes),
    )


def shape_dtype_tree_from_pytree(value: object) -> ShapeDtypeTree:
    return ShapeDtypeTree.from_pytree(value)


def make_shape_dtype_tree(value: object) -> ShapeDtypeTree:
    return ShapeDtypeTree.from_pytree(value)


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodEvalRequest:
    protocol_version: int
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    compile_identity_digest: str
    eval_id: str
    U_bytes: bytes
    U_shape_tree: ShapeDtypeTree
    requested_dtype_policy: str
    deadline_ms: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodEvalResponse:
    protocol_version: int
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    compile_identity_digest: str
    eval_id: str
    status: str
    log_L: float | None
    error_type: str | None
    error_message: str | None
    worker_id: str
    cache_event: str
    elapsed_seconds: float


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodDispatchDiagnostics(_BaseLikelihoodDispatchDiagnostics):
    """Runtime diagnostics with compact per-worker completion summaries."""


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeCompileIdentity:
    serialized_model_digest: str
    serialized_args_digest: str
    serialized_params_digest: str
    dtype_policy: str
    device_class: str
    U_shape_tree: ShapeDtypeTree
    identity_digest: str

    @classmethod
    def from_problem(
            cls,
            *,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
            dtype_policy: str = "float32",
            device_class: str = "cpu",
            U_shape_tree: ShapeDtypeTree,
    ) -> "RuntimeCompileIdentity":
        serialized_model_digest = _digest_bytes(serialize_model(model))
        serialized_args_digest = _digest_bytes(serialize_args(args))
        serialized_params_digest = _digest_bytes(serialize_params(params))
        identity_parts = (
            serialized_model_digest,
            serialized_args_digest,
            serialized_params_digest,
            str(dtype_policy),
            str(device_class),
            U_shape_tree,
        )
        identity_digest = _digest_bytes(pickle_payload(identity_parts))
        return cls(
            serialized_model_digest=serialized_model_digest,
            serialized_args_digest=serialized_args_digest,
            serialized_params_digest=serialized_params_digest,
            dtype_policy=str(dtype_policy),
            device_class=str(device_class),
            U_shape_tree=U_shape_tree,
            identity_digest=identity_digest,
        )

    @classmethod
    def from_serialized_problem(
            cls,
            *,
            serialized_problem: SerializedModelProblem,
            dtype_policy: str = "float32",
            device_class: str = "cpu",
            U_shape_tree: ShapeDtypeTree,
    ) -> "RuntimeCompileIdentity":
        serialized_model_digest = _digest_bytes(serialized_problem.model_bytes)
        serialized_args_digest = _digest_bytes(serialized_problem.args_bytes)
        serialized_params_digest = _digest_bytes(
            serialized_problem.params_bytes
        )
        identity_parts = (
            serialized_model_digest,
            serialized_args_digest,
            serialized_params_digest,
            str(dtype_policy),
            str(device_class),
            U_shape_tree,
        )
        identity_digest = _digest_bytes(pickle_payload(identity_parts))
        return cls(
            serialized_model_digest=serialized_model_digest,
            serialized_args_digest=serialized_args_digest,
            serialized_params_digest=serialized_params_digest,
            dtype_policy=str(dtype_policy),
            device_class=str(device_class),
            U_shape_tree=U_shape_tree,
            identity_digest=identity_digest,
        )


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
class ConstrainedSamplerCompletionPayload:
    """Validated child-sample payload allowed to enter acceptance bookkeeping."""

    U_sample: object
    log_L: object
    num_likelihood_evaluations: object
    phantom_samples: PhantomSamples

    @classmethod
    def from_output(
            cls,
            output: object,
    ) -> "ConstrainedSamplerCompletionPayload":
        if isinstance(output, cls):
            return output
        if not isinstance(output, tuple) or len(output) != 4:
            raise TypeError(
                "Constrained-sampler completion payload must be a "
                "4-tuple of (U_sample, log_L, num_likelihood_evaluations, "
                "phantom_samples)."
            )
        U_sample, log_L, num_likelihood_evaluations, phantom_samples = output
        if not isinstance(phantom_samples, PhantomSamples):
            raise TypeError(
                "Constrained-sampler completion payload requires "
                "PhantomSamples."
            )
        return cls(
            U_sample=U_sample,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=phantom_samples,
        )

    @property
    def output(self) -> tuple[object, object, object, PhantomSamples]:
        return (
            self.U_sample,
            self.log_L,
            self.num_likelihood_evaluations,
            self.phantom_samples,
        )


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


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodEvalDispatchRecord:
    request: LikelihoodEvalRequest
    status: str
    worker_id: str = ""
    response: LikelihoodEvalResponse | None = None
    dispatch_latency_seconds: float = 0.0

    @property
    def runner_id(self) -> str:
        return self.request.runner_id

    @property
    def task_id(self) -> str:
        return self.request.task_id

    @property
    def attempt_id(self) -> str:
        return self.request.attempt_id

    @property
    def transport_id(self) -> str:
        return self.request.transport_id

    @property
    def eval_id(self) -> str:
        return self.request.eval_id

    @property
    def compile_identity_digest(self) -> str:
        return self.request.compile_identity_digest


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodEvalCompletion:
    worker_id: str
    response: LikelihoodEvalResponse
    completed_record: LikelihoodEvalDispatchRecord | None
    started_record: LikelihoodEvalDispatchRecord | None
    status: str = "completed"

    @property
    def next_started_record(self) -> LikelihoodEvalDispatchRecord | None:
        return self.started_record


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodEvalRoutingResult:
    status: str
    response: LikelihoodEvalResponse


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodDispatchSnapshot:
    accepted_sample_count: int = 0
    out_degree_update_count: int = 0
    phantom_cluster_update_count: int = 0
    allocation_update_count: int = 0
    pending_eval_count: int = 0
    delivered_eval_count: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _RegisteredLikelihoodIdentity:
    identity: RuntimeCompileIdentity
    model: object
    args: tuple[object, ...]
    params: object | None
    serialized_problem: SerializedModelProblem | None = None


def _registered_likelihood_identity_from_problem(
        *,
        identity: RuntimeCompileIdentity,
        model: object,
        args: tuple[object, ...],
        params: object | None,
) -> _RegisteredLikelihoodIdentity:
    model_bytes = serialize_model(model)
    args_bytes = serialize_args(args)
    params_bytes = serialize_params(params)
    mismatches = []
    if _digest_bytes(model_bytes) != identity.serialized_model_digest:
        mismatches.append("model")
    if _digest_bytes(args_bytes) != identity.serialized_args_digest:
        mismatches.append("args")
    if _digest_bytes(params_bytes) != identity.serialized_params_digest:
        mismatches.append("params")
    if mismatches:
        raise ValueError(
            "Runtime compile identity drift for "
            f"{', '.join(mismatches)}."
        )
    serialized_problem = SerializedModelProblem(
        model_bytes=model_bytes,
        args_bytes=args_bytes,
        params_bytes=params_bytes,
        collect_phantoms=False,
    )
    return _RegisteredLikelihoodIdentity(
        identity=identity,
        model=deserialize_model(model_bytes),
        args=deserialize_args(args_bytes),
        params=deserialize_params(params_bytes),
        serialized_problem=serialized_problem,
    )


def _validate_serialized_problem_identity_digests(
        *,
        identity: RuntimeCompileIdentity,
        serialized_problem: SerializedModelProblem,
) -> None:
    mismatches = []
    if _digest_bytes(serialized_problem.model_bytes) != (
            identity.serialized_model_digest
    ):
        mismatches.append("model")
    if _digest_bytes(serialized_problem.args_bytes) != (
            identity.serialized_args_digest
    ):
        mismatches.append("args")
    if _digest_bytes(serialized_problem.params_bytes) != (
            identity.serialized_params_digest
    ):
        mismatches.append("params")
    if mismatches:
        raise ValueError(
            "Runtime compile identity drift for "
            f"{', '.join(mismatches)}."
        )


def _registered_likelihood_identity_from_serialized_problem(
        *,
        identity: RuntimeCompileIdentity,
        serialized_problem: SerializedModelProblem,
) -> _RegisteredLikelihoodIdentity:
    _validate_serialized_problem_identity_digests(
        identity=identity,
        serialized_problem=serialized_problem,
    )
    return _RegisteredLikelihoodIdentity(
        identity=identity,
        model=deserialize_model(serialized_problem.model_bytes),
        args=deserialize_args(serialized_problem.args_bytes),
        params=deserialize_params(serialized_problem.params_bytes),
        serialized_problem=serialized_problem,
    )


class LikelihoodEvalWorker:
    """In-process first layer for the Ticket 0018 likelihood worker boundary."""

    def __init__(
            self,
            *,
            worker_id: str,
            device_class: str = "cpu",
    ) -> None:
        self.worker_id = str(worker_id)
        self.device_class = str(device_class)
        self._identities: dict[str, _RegisteredLikelihoodIdentity] = {}
        self._compiled_likelihoods: dict[
            tuple[str, str],
            object,
        ] = {}
        self._compile_count = 0
        self._cache_hit_count = 0
        self._rejected_shape_cache_count = 0
        self._failed_eval_count = 0
        self._completed_eval_count = 0
        self._dispatch_latencies: list[float] = []
        self._active_eval_count = 0
        self._max_active_eval_count = 0
        self._lock = threading.Lock()

    def register_compile_identity(
            self,
            *,
            identity: RuntimeCompileIdentity,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
    ) -> None:
        payload = _registered_likelihood_identity_from_problem(
            identity=identity,
            model=model,
            args=args,
            params=params,
        )
        self._register_likelihood_identity_payload(payload)

    def _register_likelihood_identity_payload(
            self,
            payload: _RegisteredLikelihoodIdentity,
    ) -> None:
        serialized_problem = payload.serialized_problem
        if serialized_problem is not None:
            payload = _registered_likelihood_identity_from_serialized_problem(
                identity=payload.identity,
                serialized_problem=serialized_problem,
            )
        with self._lock:
            self._identities[payload.identity.identity_digest] = payload

    def register_runtime_compile_identity(
            self,
            *,
            identity: RuntimeCompileIdentity,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
    ) -> None:
        self.register_compile_identity(
            identity=identity,
            model=model,
            args=args,
            params=params,
        )

    def evaluate_likelihood(
            self,
            request: LikelihoodEvalRequest,
    ) -> LikelihoodEvalResponse:
        start = time.perf_counter()
        if request.deadline_ms is not None and request.deadline_ms <= 0:
            return self._failed_response(
                request,
                error_type="LikelihoodEvalTimeout",
                error_message="Likelihood evaluation deadline expired.",
                start=start,
                count_shape_rejection=False,
            )

        if not self._begin_active_eval():
            return self._failed_response(
                request,
                error_type="WorkerCapacityExceeded",
                error_message=(
                    "Likelihood worker already has one active evaluation."
                ),
                start=start,
                count_shape_rejection=False,
            )

        try:
            return self._evaluate_likelihood_active(request, start)
        finally:
            self._end_active_eval()

    def _evaluate_likelihood_active(
            self,
            request: LikelihoodEvalRequest,
            start: float,
    ) -> LikelihoodEvalResponse:
        with self._lock:
            registered = self._identities.get(
                request.compile_identity_digest
            )
        if registered is None:
            return self._failed_response(
                request,
                error_type="UnknownCompileIdentity",
                error_message=(
                    "No registered runtime compile identity matches "
                    f"{request.compile_identity_digest!r}."
                ),
                start=start,
                count_shape_rejection=False,
            )

        if registered.identity.device_class != self.device_class:
            return self._failed_response(
                request,
                error_type="DeviceClassMismatch",
                error_message=(
                    "Registered runtime compile identity device class "
                    f"{registered.identity.device_class!r} does not match "
                    f"worker device class {self.device_class!r}."
                ),
                start=start,
                count_shape_rejection=False,
            )

        if request.requested_dtype_policy != registered.identity.dtype_policy:
            return self._failed_response(
                request,
                error_type="DtypePolicyMismatch",
                error_message=(
                    "Requested dtype policy does not match the registered "
                    "runtime compile identity."
                ),
                start=start,
                count_shape_rejection=True,
            )

        try:
            U = unpickle_payload(request.U_bytes)
        except Exception as error:
            return self._failed_response(
                request,
                error_type="MalformedUPayload",
                error_message=str(error),
                start=start,
                count_shape_rejection=False,
            )

        actual_shape_tree = _shape_dtype_tree_from_pytree(U)
        expected_shape_tree = registered.identity.U_shape_tree
        metadata_mismatch_error = _shape_tree_mismatch_error_type(
            request.U_shape_tree,
            actual_shape_tree
        )
        if metadata_mismatch_error is not None:
            return self._failed_response(
                request,
                error_type=f"RequestMetadata{metadata_mismatch_error}",
                error_message=(
                    "Likelihood U_shape_tree metadata does not match the "
                    "unpickled U payload."
                ),
                start=start,
                count_shape_rejection=True,
            )
        mismatch_error = expected_shape_tree.mismatch_error_type(
            actual_shape_tree
        )
        if mismatch_error is not None:
            return self._failed_response(
                request,
                error_type=mismatch_error,
                error_message=(
                    "Likelihood U pytree does not match the registered "
                    "static shape/dtype contract."
                ),
                start=start,
                count_shape_rejection=True,
            )
        identity_mismatch_error = expected_shape_tree.mismatch_error_type(
            request.U_shape_tree
        )
        if identity_mismatch_error is not None:
            return self._failed_response(
                request,
                error_type=identity_mismatch_error,
                error_message=(
                    "Likelihood U_shape_tree metadata does not match the "
                    "registered static shape/dtype contract."
                ),
                start=start,
                count_shape_rejection=True,
            )

        compiled_callable, cache_event = self._compiled_likelihood(
            request,
            registered,
        )
        try:
            log_L = compiled_callable(U)
            if hasattr(log_L, "block_until_ready"):
                log_L.block_until_ready()
            log_L_value = float(np.asarray(log_L))
        except Exception as error:
            return self._failed_response(
                request,
                error_type="WorkerException",
                error_message=str(error),
                start=start,
                count_shape_rejection=False,
            )

        elapsed = _nonnegative_elapsed_seconds(start, time.perf_counter())
        response = LikelihoodEvalResponse(
            protocol_version=request.protocol_version,
            runner_id=request.runner_id,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            transport_id=request.transport_id,
            compile_identity_digest=request.compile_identity_digest,
            eval_id=request.eval_id,
            status="ok",
            log_L=log_L_value,
            error_type=None,
            error_message=None,
            worker_id=self.worker_id,
            cache_event=cache_event,
            elapsed_seconds=elapsed,
        )
        with self._lock:
            if cache_event == "hit":
                self._cache_hit_count += 1
            else:
                self._compiled_likelihoods[
                    self._compiled_cache_key(request)
                ] = compiled_callable
                self._compile_count += 1
            self._completed_eval_count += 1
            self._dispatch_latencies.append(elapsed)
        return response

    def _begin_active_eval(self) -> bool:
        with self._lock:
            if self._active_eval_count >= 1:
                return False
            self._active_eval_count += 1
            self._max_active_eval_count = max(
                self._max_active_eval_count,
                self._active_eval_count,
            )
            return True

    def _end_active_eval(self) -> None:
        with self._lock:
            self._active_eval_count = max(0, self._active_eval_count - 1)

    def _compiled_cache_key(
            self,
            request: LikelihoodEvalRequest,
    ) -> tuple[str, str]:
        return request.compile_identity_digest, self.device_class

    def _compiled_likelihood(
            self,
            request: LikelihoodEvalRequest,
            registered: _RegisteredLikelihoodIdentity,
    ):
        cache_key = self._compiled_cache_key(request)
        with self._lock:
            compiled_callable = self._compiled_likelihoods.get(cache_key)
        if compiled_callable is not None:
            return compiled_callable, "hit"

        def likelihood(U):
            return registered.model.log_likelihood(
                U,
                args=registered.args,
                params=registered.params,
                allow_nan=False,
            )

        return jax.jit(likelihood), "compile"

    def handle_likelihood_eval(
            self,
            request: LikelihoodEvalRequest,
    ) -> LikelihoodEvalResponse:
        return self.evaluate_likelihood(request)

    def _failed_response(
            self,
            request: LikelihoodEvalRequest,
            *,
            error_type: str,
            error_message: str,
            start: float,
            count_shape_rejection: bool,
    ) -> LikelihoodEvalResponse:
        elapsed = _nonnegative_elapsed_seconds(start, time.perf_counter())
        response = LikelihoodEvalResponse(
            protocol_version=request.protocol_version,
            runner_id=request.runner_id,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            transport_id=request.transport_id,
            compile_identity_digest=request.compile_identity_digest,
            eval_id=request.eval_id,
            status="failed",
            log_L=None,
            error_type=error_type,
            error_message=error_message,
            worker_id=self.worker_id,
            cache_event="rejected",
            elapsed_seconds=elapsed,
        )
        with self._lock:
            self._failed_eval_count += 1
            if count_shape_rejection:
                self._rejected_shape_cache_count += 1
            self._dispatch_latencies.append(elapsed)
        return response

    def likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        with self._lock:
            latencies = tuple(self._dispatch_latencies)
            compile_count = self._compile_count
            cache_hit_count = self._cache_hit_count
            rejected_shape_cache_count = self._rejected_shape_cache_count
            failed_eval_count = self._failed_eval_count
            completed_eval_count = self._completed_eval_count
            identity_count = len(self._identities)
            max_active_eval_count = self._max_active_eval_count
        return _make_likelihood_diagnostics(
            requested_worker_specs=(f"{self.device_class}:*:1",),
            observed_worker_count=1,
            observed_worker_device_classes=(self.device_class,),
            dispatch_latency_seconds=latencies,
            compile_count=compile_count,
            cache_hit_count=cache_hit_count,
            rejected_shape_cache_count=rejected_shape_cache_count,
            distinct_compile_identity_count=identity_count,
            max_active_evals_per_worker={
                self.worker_id: max_active_eval_count,
            },
            max_active_evals_pool=max_active_eval_count,
            completed_eval_count_by_worker={
                self.worker_id: completed_eval_count,
            },
            failed_eval_count=failed_eval_count,
        )

    def get_diagnostics(self) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()

    def diagnostics(self) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()


class LikelihoodEvalScheduler:
    """Capacity-one scheduler for local likelihood-eval workers."""

    def __init__(
            self,
            *,
            requested_worker_specs: Iterable[str] = ("cpu:*:1",),
    ) -> None:
        self.requested_worker_specs = tuple(requested_worker_specs)
        self._workers: dict[str, LikelihoodEvalWorker] = {}
        self._worker_order: list[str] = []
        self._active_by_worker: dict[str, LikelihoodEvalDispatchRecord] = {}
        self._queue: list[LikelihoodEvalRequest] = []
        self._started_by_eval_key: dict[
            tuple[str, str, str, str, str, str],
            LikelihoodEvalDispatchRecord,
        ] = {}
        self._cancelled_eval_keys: set[
            tuple[str, str, str, str, str, str],
        ] = set()
        self._registered_identities: dict[
            str,
            _RegisteredLikelihoodIdentity,
        ] = {}
        self._max_active_by_worker: dict[str, int] = {}
        self._completed_eval_count_by_worker: dict[str, int] = {}
        self._max_active_pool = 0
        self._next_idle_worker_offset = 0
        self._queued_eval_count = 0
        self._scheduler_failed_eval_count = 0
        self._worker_failed_eval_count = 0
        self._public_compile_count = 0
        self._public_cache_hit_count = 0
        self._public_rejected_shape_cache_count = 0
        self._public_dispatch_latencies: list[float] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._build_workers()

    def _build_workers(self) -> None:
        worker_index = 0
        for spec_text in self.requested_worker_specs:
            spec = parse_worker_spec(spec_text)
            for _ in spec.device_ids:
                for _ in range(spec.workers_per_device):
                    worker_index += 1
                    worker_id = f"worker-{worker_index:06d}"
                    worker = LikelihoodEvalWorker(
                        worker_id=worker_id,
                        device_class=spec.device_type,
                    )
                    self._workers[worker_id] = worker
                    self._worker_order.append(worker_id)
                    self._max_active_by_worker[worker_id] = 0
                    self._completed_eval_count_by_worker[worker_id] = 0

    def register_compile_identity(
            self,
            *,
            identity: RuntimeCompileIdentity,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
    ) -> None:
        payload = _registered_likelihood_identity_from_problem(
            identity=identity,
            model=model,
            args=args,
            params=params,
        )
        self._register_likelihood_identity_payload(payload)

    def _register_likelihood_identity_payload(
            self,
            payload: _RegisteredLikelihoodIdentity,
    ) -> None:
        with self._lock:
            self._registered_identities[
                payload.identity.identity_digest
            ] = payload
            workers = tuple(self._workers.values())
        for worker in workers:
            worker._register_likelihood_identity_payload(payload)

    def register_runtime_compile_identity(
            self,
            *,
            identity: RuntimeCompileIdentity,
            model: object,
            args: tuple[object, ...] = (),
            params: object | None = None,
    ) -> None:
        self.register_compile_identity(
            identity=identity,
            model=model,
            args=args,
            params=params,
        )

    def start_likelihood_eval(
            self,
            *,
            request: LikelihoodEvalRequest,
    ) -> LikelihoodEvalDispatchRecord:
        with self._condition:
            idle_worker_id = self._next_idle_worker_id(request=request)
            if idle_worker_id is None:
                self._queue.append(request)
                self._queued_eval_count += 1
                record = LikelihoodEvalDispatchRecord(
                    request=request,
                    status="queued",
                )
                return record
            return self._start_on_worker_locked(request, idle_worker_id)

    def complete_likelihood_eval(
            self,
            *,
            worker_id: str,
            response: LikelihoodEvalResponse,
    ) -> LikelihoodEvalCompletion:
        with self._condition:
            response_key = _likelihood_scheduler_key(response)
            active_record = self._active_by_worker.get(worker_id)
            if (
                    active_record is not None
                    and _likelihood_scheduler_key(active_record.request)
                    == response_key
            ):
                physical_completed_record = self._active_by_worker.pop(
                    worker_id
                )
            else:
                physical_completed_record = None
            completed_record = None
            if physical_completed_record is not None:
                was_cancelled = response_key in self._cancelled_eval_keys
                self._started_by_eval_key.pop(
                    _likelihood_scheduler_key(
                        physical_completed_record.request
                    ),
                    None,
                )
                if not was_cancelled:
                    completed_record = physical_completed_record
                    self._public_dispatch_latencies.append(
                        float(response.elapsed_seconds)
                    )
                    if response.status == "ok":
                        self._completed_eval_count_by_worker[worker_id] = (
                            self._completed_eval_count_by_worker.get(
                                worker_id,
                                0,
                            )
                            + 1
                        )
                        if response.cache_event == "compile":
                            self._public_compile_count += 1
                        elif response.cache_event == "hit":
                            self._public_cache_hit_count += 1
                    else:
                        self._worker_failed_eval_count += 1
                        if _is_shape_cache_rejection(response.error_type):
                            self._public_rejected_shape_cache_count += 1
                self._cancelled_eval_keys.discard(response_key)
            elif response_key in self._cancelled_eval_keys:
                self._cancelled_eval_keys.discard(response_key)
            started_record = (
                self._start_next_queued_on_worker_locked(worker_id)
                if physical_completed_record is not None
                else None
            )
            self._condition.notify_all()
            return LikelihoodEvalCompletion(
                worker_id=worker_id,
                response=response,
                completed_record=completed_record,
                started_record=started_record,
            )

    def cancel_likelihood_eval(
            self,
            *,
            request: LikelihoodEvalRequest,
            error_type: str = "LikelihoodEvalTimeout",
            error_message: str = (
                "Timed out waiting for a queued likelihood evaluation to "
                "start."
            ),
    ) -> LikelihoodEvalResponse:
        start = time.perf_counter()
        with self._condition:
            return self._cancel_likelihood_eval_locked(
                request=request,
                error_type=error_type,
                error_message=error_message,
                start=start,
            )

    def wait_for_started_record(
            self,
            request: LikelihoodEvalRequest,
            timeout_seconds: float | None = None,
    ) -> LikelihoodEvalDispatchRecord | None:
        key = _likelihood_scheduler_key(request)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while True:
                if key in self._cancelled_eval_keys:
                    return None
                record = self._started_by_eval_key.get(key)
                if record is not None:
                    return record
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def wait_for_started_record_or_cancel_likelihood_eval(
            self,
            request: LikelihoodEvalRequest,
            timeout_seconds: float | None = None,
            error_type: str = "LikelihoodEvalTimeout",
            error_message: str = (
                "Timed out waiting for a queued likelihood evaluation to "
                "start."
            ),
    ) -> LikelihoodEvalDispatchRecord | LikelihoodEvalResponse:
        key = _likelihood_scheduler_key(request)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        start = time.perf_counter()
        with self._condition:
            while True:
                record = self._started_by_eval_key.get(key)
                if record is not None:
                    return record
                if key in self._cancelled_eval_keys:
                    return self._cancel_likelihood_eval_locked(
                        request=request,
                        error_type=error_type,
                        error_message=error_message,
                        start=start,
                    )
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return self._cancel_likelihood_eval_locked(
                        request=request,
                        error_type=error_type,
                        error_message=error_message,
                        start=start,
                    )
                self._condition.wait(timeout=remaining)

    def worker_for_id(self, worker_id: str) -> LikelihoodEvalWorker:
        return self._workers[worker_id]

    def has_workers(self) -> bool:
        return bool(self._workers)

    def _next_idle_worker_id(
            self,
            *,
            request: LikelihoodEvalRequest,
    ) -> str | None:
        worker_count = len(self._worker_order)
        if worker_count == 0:
            return None
        for offset in range(worker_count):
            worker_index = (
                self._next_idle_worker_offset + offset
            ) % worker_count
            worker_id = self._worker_order[worker_index]
            if (
                    worker_id not in self._active_by_worker
                    and self._worker_can_run_request_locked(
                        worker_id,
                        request,
                    )
            ):
                self._next_idle_worker_offset = (
                    worker_index + 1
                ) % worker_count
                return worker_id
        return None

    def _start_on_worker_locked(
            self,
            request: LikelihoodEvalRequest,
            worker_id: str,
    ) -> LikelihoodEvalDispatchRecord:
        record = LikelihoodEvalDispatchRecord(
            request=request,
            status="started",
            worker_id=worker_id,
        )
        self._active_by_worker[worker_id] = record
        self._started_by_eval_key[_likelihood_scheduler_key(request)] = record
        active_for_worker = 1
        self._max_active_by_worker[worker_id] = max(
            self._max_active_by_worker.get(worker_id, 0),
            active_for_worker,
        )
        self._max_active_pool = max(
            self._max_active_pool,
            len(self._active_by_worker),
        )
        return record

    def _start_next_queued_on_worker_locked(
            self,
            worker_id: str,
    ) -> LikelihoodEvalDispatchRecord | None:
        queue_index = 0
        while queue_index < len(self._queue):
            next_request = self._queue[queue_index]
            next_key = _likelihood_scheduler_key(next_request)
            if next_key in self._cancelled_eval_keys:
                self._queue.pop(queue_index)
                continue
            if not self._worker_can_run_request_locked(
                    worker_id,
                    next_request,
            ):
                queue_index += 1
                continue
            self._queue.pop(queue_index)
            return self._start_on_worker_locked(next_request, worker_id)
        return None

    def _worker_can_run_request_locked(
            self,
            worker_id: str,
            request: LikelihoodEvalRequest,
    ) -> bool:
        payload = self._registered_identities.get(
            request.compile_identity_digest
        )
        if payload is None:
            return True
        worker = self._workers[worker_id]
        return payload.identity.device_class == worker.device_class

    def _cancel_likelihood_eval_locked(
            self,
            *,
            request: LikelihoodEvalRequest,
            error_type: str,
            error_message: str,
            start: float,
    ) -> LikelihoodEvalResponse:
        key = _likelihood_scheduler_key(request)
        already_cancelled = key in self._cancelled_eval_keys
        self._cancelled_eval_keys.add(key)
        self._started_by_eval_key.pop(key, None)
        self._queue = [
            queued_request
            for queued_request in self._queue
            if _likelihood_scheduler_key(queued_request) != key
        ]
        elapsed = _nonnegative_elapsed_seconds(start, time.perf_counter())
        response = LikelihoodEvalResponse(
            protocol_version=request.protocol_version,
            runner_id=request.runner_id,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            transport_id=request.transport_id,
            compile_identity_digest=request.compile_identity_digest,
            eval_id=request.eval_id,
            status="failed",
            log_L=None,
            error_type=error_type,
            error_message=error_message,
            worker_id="",
            cache_event="rejected",
            elapsed_seconds=elapsed,
        )
        if not already_cancelled:
            self._scheduler_failed_eval_count += 1
            self._public_dispatch_latencies.append(elapsed)
        self._condition.notify_all()
        return response

    def likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        with self._lock:
            worker_items = tuple(self._workers.items())
            max_active_by_worker = dict(self._max_active_by_worker)
            completed_eval_count_by_worker = dict(
                self._completed_eval_count_by_worker
            )
            max_active_pool = self._max_active_pool
            queued_eval_count = self._queued_eval_count
            identity_count = len(self._registered_identities)
            scheduler_failed_eval_count = self._scheduler_failed_eval_count
            worker_failed_eval_count = self._worker_failed_eval_count
            compile_count = self._public_compile_count
            cache_hit_count = self._public_cache_hit_count
            rejected_shape_cache_count = (
                self._public_rejected_shape_cache_count
            )
            public_dispatch_latencies = tuple(self._public_dispatch_latencies)
        failed_eval_count = scheduler_failed_eval_count + worker_failed_eval_count
        latencies = public_dispatch_latencies
        device_classes = tuple(
            sorted({worker.device_class for _, worker in worker_items})
        )
        completed_eval_count = sum(
            completed_eval_count_by_worker.values()
        )
        dispatch_eval_count = completed_eval_count + failed_eval_count
        return _make_likelihood_diagnostics(
            requested_worker_specs=self.requested_worker_specs,
            observed_worker_count=len(worker_items),
            observed_worker_device_classes=device_classes,
            dispatch_latency_seconds=latencies,
            compile_count=compile_count,
            cache_hit_count=cache_hit_count,
            rejected_shape_cache_count=rejected_shape_cache_count,
            distinct_compile_identity_count=identity_count,
            max_active_evals_per_worker=max_active_by_worker,
            max_active_evals_pool=max_active_pool,
            completed_eval_count_by_worker=completed_eval_count_by_worker,
            queued_eval_count=queued_eval_count,
            failed_eval_count=failed_eval_count,
            dispatch_eval_count=dispatch_eval_count,
        )

    def get_diagnostics(self) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()

    def diagnostics(self) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()


class LikelihoodEvalResponseRouter:
    """Routes raw likelihood responses without mutating statistical state."""

    def __init__(self) -> None:
        self._pending: set[tuple[str, str, str, str, str, str]] = set()
        self._delivered: set[tuple[str, str, str, str, str, str]] = set()
        self._lock = threading.Lock()

    def register_pending(self, request: LikelihoodEvalRequest) -> None:
        with self._lock:
            self._pending.add(_likelihood_eval_key(request))

    def route_eval_response(
            self,
            response: LikelihoodEvalResponse,
    ) -> LikelihoodEvalRoutingResult:
        key = _likelihood_eval_key(response)
        with self._lock:
            if key in self._pending:
                self._pending.remove(key)
                self._delivered.add(key)
                status = "delivered"
            elif key in self._delivered:
                status = "duplicate_eval_response"
            else:
                status = "stale_eval_response"
        return LikelihoodEvalRoutingResult(status=status, response=response)

    def route_response(
            self,
            response: LikelihoodEvalResponse,
    ) -> LikelihoodEvalRoutingResult:
        return self.route_eval_response(response)

    def snapshot(self) -> LikelihoodDispatchSnapshot:
        with self._lock:
            return LikelihoodDispatchSnapshot(
                pending_eval_count=len(self._pending),
                delivered_eval_count=len(self._delivered),
            )


def _likelihood_eval_key(
        value: LikelihoodEvalRequest | LikelihoodEvalResponse,
) -> tuple[str, str, str, str, str, str]:
    return (
        str(value.runner_id),
        str(value.task_id),
        str(value.attempt_id),
        str(value.transport_id),
        str(value.compile_identity_digest),
        str(value.eval_id),
    )


def _likelihood_scheduler_key(
        value: LikelihoodEvalRequest | LikelihoodEvalResponse,
) -> tuple[str, str, str, str, str, str]:
    return (
        str(value.runner_id),
        str(value.task_id),
        str(value.attempt_id),
        str(value.transport_id),
        str(value.compile_identity_digest),
        str(value.eval_id),
    )


def _is_shape_cache_rejection(error_type: str | None) -> bool:
    if error_type is None:
        return False
    normalized = str(error_type).replace("_", "").replace("-", "").lower()
    return normalized in {
        "pytreemismatch",
        "shapemismatch",
        "dtypemismatch",
        "requestmetadatapytreemismatch",
        "requestmetadatashapemismatch",
        "requestmetadatadtypemismatch",
        "requestmetadatamalformedushapetreemetadata",
        "dtypepolicymismatch",
    }


def _make_likelihood_diagnostics(
        *,
        requested_worker_specs: tuple[str, ...],
        observed_worker_count: int,
        observed_worker_device_classes: tuple[str, ...],
        dispatch_latency_seconds: tuple[float, ...],
        compile_count: int,
        cache_hit_count: int,
        rejected_shape_cache_count: int,
        distinct_compile_identity_count: int,
        max_active_evals_per_worker: object,
        max_active_evals_pool: int,
        completed_eval_count_by_worker: object | None = None,
        likelihood_eval_records: tuple[object, ...] = (),
        queued_eval_count: int = 0,
        failed_eval_count: int = 0,
        dispatch_eval_count: int | None = None,
) -> LikelihoodDispatchDiagnostics:
    if dispatch_eval_count is None:
        dispatch_eval_count = len(dispatch_latency_seconds)
    total_latency = float(sum(dispatch_latency_seconds))
    if total_latency > 0.0:
        throughput = int(dispatch_eval_count) / total_latency
    else:
        throughput = float(dispatch_eval_count)
    return LikelihoodDispatchDiagnostics(
        requested_worker_specs=tuple(requested_worker_specs),
        observed_worker_count=int(observed_worker_count),
        observed_worker_device_classes=tuple(observed_worker_device_classes),
        dispatch_eval_count=int(dispatch_eval_count),
        dispatch_latency_seconds=tuple(dispatch_latency_seconds),
        dispatch_throughput_per_second=float(throughput),
        compile_count=int(compile_count),
        cache_hit_count=int(cache_hit_count),
        rejected_shape_cache_count=int(rejected_shape_cache_count),
        distinct_compile_identity_count=int(distinct_compile_identity_count),
        max_active_evals_per_worker=max_active_evals_per_worker,
        max_active_evals_pool=int(max_active_evals_pool),
        completed_eval_count_by_worker=(
            {} if completed_eval_count_by_worker is None
            else completed_eval_count_by_worker
        ),
        likelihood_eval_records=tuple(likelihood_eval_records),
        queued_eval_count=int(queued_eval_count),
        failed_eval_count=int(failed_eval_count),
    )


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
    _likelihood_identity_payloads: dict[
        str,
        _RegisteredLikelihoodIdentity,
    ] = dataclasses.field(
        default_factory=dict,
        repr=False,
    )
    _likelihood_scheduler: LikelihoodEvalScheduler | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )
    _likelihood_scheduler_generation: int = dataclasses.field(
        default=0,
        repr=False,
        compare=False,
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

    def requested_worker_spec_strings(self) -> tuple[str, ...]:
        specs: list[str] = []
        seen: set[tuple[str, tuple[str, ...], int]] = set()
        with self._lock:
            sectors = tuple(self.compute_sectors.values())
        for sector in sectors:
            spec = sector.source_worker_spec
            key = (
                spec.device_type,
                tuple(spec.device_ids),
                int(spec.workers_per_device),
            )
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                (
                    f"{spec.device_type}:"
                    f"{','.join(spec.device_ids)}:"
                    f"{spec.workers_per_device}"
                )
            )
        return tuple(specs)

    def likelihood_worker_spec_strings(self) -> tuple[str, ...]:
        with self._lock:
            return self._likelihood_worker_spec_strings_locked()

    def _likelihood_worker_spec_strings_locked(self) -> tuple[str, ...]:
        return tuple(
            (
                f"{sector.device_type}:"
                f"{sector.device_id}:"
                f"{sector.num_workers}"
            )
            for sector in self.compute_sectors.values()
        )

    def likelihood_scheduler_signature(self) -> tuple[tuple[str, ...], int]:
        with self._lock:
            return (
                self._likelihood_worker_spec_strings_locked(),
                self._likelihood_scheduler_generation,
            )

    def likelihood_scheduler_generation(self) -> int:
        with self._lock:
            return self._likelihood_scheduler_generation

    def likelihood_eval_scheduler(self) -> LikelihoodEvalScheduler:
        with self._lock:
            requested_specs = self._likelihood_worker_spec_strings_locked()
            scheduler = self._likelihood_scheduler
            if (
                    scheduler is not None
                    and scheduler.requested_worker_specs == requested_specs
            ):
                return scheduler
            identity_payloads = tuple(
                self._likelihood_identity_payloads.values()
            )
            scheduler = LikelihoodEvalScheduler(
                requested_worker_specs=requested_specs,
            )
            self._likelihood_scheduler = scheduler
            self._likelihood_scheduler_generation += 1
        for payload in identity_payloads:
            scheduler._register_likelihood_identity_payload(payload)
        return scheduler

    def register_likelihood_compile_identity(
            self,
            *,
            identity: RuntimeCompileIdentity,
            serialized_problem: SerializedModelProblem,
    ) -> None:
        _validate_serialized_problem_identity_digests(
            identity=identity,
            serialized_problem=serialized_problem,
        )
        with self._lock:
            existing = self._likelihood_identity_payloads.get(
                identity.identity_digest
            )
            if existing is not None:
                return
        payload = _registered_likelihood_identity_from_serialized_problem(
            identity=identity,
            serialized_problem=serialized_problem,
        )
        with self._lock:
            existing = self._likelihood_identity_payloads.get(
                identity.identity_digest
            )
            if existing is not None:
                return
            self._likelihood_identity_payloads[
                identity.identity_digest
            ] = payload
            scheduler = self._likelihood_scheduler
        if scheduler is not None:
            scheduler._register_likelihood_identity_payload(payload)

    def likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_eval_scheduler().likelihood_dispatch_diagnostics()

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


def _sample_u_for_identity(
        *,
        model: object,
        args: tuple[object, ...],
        params: object | None,
) -> object:
    return model.sample_U(
        jax.random.PRNGKey(0),
        args=args,
        params=params,
    )


def _u_for_dtype_policy(value: object, dtype_policy: str) -> object:
    if dtype_policy != "float32":
        return value

    def coerce_leaf(leaf):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float32)
        return leaf

    return jax.tree.map(coerce_leaf, value)


def _device_class_from_worker_device_types(
        worker_device_types: tuple[str, ...],
        *,
        previous_device_class: str | None = None,
) -> str:
    if not worker_device_types:
        return "cpu"
    device_types = tuple(sorted(str(device_type) for device_type in worker_device_types))
    if previous_device_class in device_types:
        return str(previous_device_class)
    return device_types[0]


def build_runtime_compile_identity(
        *,
        model: object,
        args: tuple[object, ...] = (),
        params: object | None = None,
        dtype_policy: str = "float32",
        device_class: str = "cpu",
        U_shape_tree: ShapeDtypeTree,
) -> RuntimeCompileIdentity:
    return RuntimeCompileIdentity.from_problem(
        model=model,
        args=args,
        params=params,
        dtype_policy=dtype_policy,
        device_class=device_class,
        U_shape_tree=U_shape_tree,
    )


def _build_runtime_compile_identity(
        *,
        serialized_problem: SerializedModelProblem,
        sampler: object,
        sampler_bytes: bytes | None = None,
        worker_device_types: tuple[str, ...],
        dtype_policy: str = "float32",
        device_class: str | None = None,
        U_shape_tree: ShapeDtypeTree | None = None,
) -> RuntimeCompileIdentity:
    del sampler, sampler_bytes
    if U_shape_tree is None:
        problem = serialized_problem.deserialize_problem()
        identity_u = _sample_u_for_identity(
            model=problem.model,
            args=problem.args,
            params=problem.params,
        )
        U_shape_tree = ShapeDtypeTree.from_pytree(
            _u_for_dtype_policy(identity_u, dtype_policy)
        )
    if device_class is None:
        device_class = _device_class_from_worker_device_types(
            tuple(worker_device_types)
        )
    return RuntimeCompileIdentity.from_serialized_problem(
        serialized_problem=serialized_problem,
        dtype_policy=dtype_policy,
        device_class=device_class,
        U_shape_tree=U_shape_tree,
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
    runtime_likelihood_router: LikelihoodEvalResponseRouter = (
        dataclasses.field(
            default_factory=LikelihoodEvalResponseRouter,
            repr=False,
        )
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
    _runtime_live_dispatch_by_task_id: dict[
        str,
        tuple[str, str, str],
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
    _runtime_likelihood_scheduler: LikelihoodEvalScheduler | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )
    _runtime_likelihood_scheduler_specs: tuple[str, ...] | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )
    _runtime_likelihood_scheduler_generation: int | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )
    _runtime_registered_likelihood_identity_digest: str | None = (
        dataclasses.field(default=None, repr=False, compare=False)
    )

    def __post_init__(self):
        NestedSampler.__post_init__(self)
        if (
                self.runtime_lb_state is not None
                and self.runtime_problem_payload is not None
        ):
            self._ensure_current_runtime_compile_identity()
            self._ensure_likelihood_scheduler()

    def run_until_goal(
            self,
            goal_cond,
            depth_cond,
            allocation_target="uniform",
            key=None,
            max_goal_iterations: int = 100,
            **options,
    ):
        options = self._runtime_goal_options(options)
        return NestedSampler.run_until_goal(
            self,
            goal_cond=goal_cond,
            depth_cond=depth_cond,
            allocation_target=allocation_target,
            key=key,
            max_goal_iterations=max_goal_iterations,
            **options,
        )

    def resume_until_goal(
            self,
            state,
            goal_cond,
            depth_cond,
            allocation_target="uniform",
            key=None,
            max_goal_iterations: int = 100,
            **options,
    ):
        options = self._runtime_goal_options(options)
        return NestedSampler.resume_until_goal(
            self,
            state=state,
            goal_cond=goal_cond,
            depth_cond=depth_cond,
            allocation_target=allocation_target,
            key=key,
            max_goal_iterations=max_goal_iterations,
            **options,
        )

    def _runtime_goal_options(self, options: dict[str, object]) -> dict[str, object]:
        options = dict(options)
        if "delta_K" not in options:
            options["delta_K"] = self._runtime_default_delta_K()
        return options

    def _runtime_default_delta_K(self) -> int:
        if self.runtime_lb_state is None:
            return 1
        shell_size = max(int(self.shell_size), 1)
        scheduler = self._ensure_likelihood_scheduler()
        diagnostics = scheduler.likelihood_dispatch_diagnostics()
        return max(1, min(shell_size, int(diagnostics.observed_worker_count)))

    def _sample_parent_work(
            self,
            key,
            state: State,
            parent_work: ParentWork,
            adaptation_context=None,
    ) -> tuple[ParentWork, Samples]:
        parent_count = int(parent_work.parent_idxs.shape[0])
        max_workers = self._runtime_parent_shell_concurrency(parent_count)
        if max_workers <= 1 or not isinstance(self.sampler, UniDimSliceSampler):
            return NestedSampler._sample_parent_work(
                self,
                key,
                state,
                parent_work,
                adaptation_context=adaptation_context,
            )

        work_items = self._prepare_runtime_parent_shell_work(
            key=key,
            state=state,
            parent_work=parent_work,
            adaptation_context=adaptation_context,
            max_workers=max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._execute_runtime_parent_work_item, item)
                for item in work_items
            ]

            outputs = []
            for item, future in zip(work_items, futures, strict=True):
                try:
                    worker_result = future.result()
                except Exception as error:
                    self.mark_runtime_dispatch_failed(
                        dispatch_record=item["dispatch_record"],
                        error=error,
                    )
                    raise
                self.complete_runtime_dispatch(
                    dispatch_record=item["dispatch_record"],
                    result=worker_result,
                    current_parent_idx=int(item["effective_parent_idx"]),
                    current_effective_log_L_constraint=float(
                        item["constraint"]
                    ),
                )
                payload = self._validated_constrained_sampler_completion(
                    worker_result.payload
                )
                outputs.append(payload.output)

        return self._assemble_runtime_parent_shell_samples(
            state=state,
            work_items=work_items,
            outputs=outputs,
        )

    def _runtime_parent_shell_concurrency(self, parent_count: int) -> int:
        if parent_count <= 1:
            return 1
        worker_count = self._runtime_likelihood_worker_count()
        if worker_count <= 1:
            return 1
        return min(parent_count, worker_count)

    def _runtime_likelihood_worker_count(self) -> int:
        if self.runtime_lb_state is None:
            return 0
        scheduler = self._ensure_likelihood_scheduler()
        diagnostics = scheduler.likelihood_dispatch_diagnostics()
        return int(diagnostics.observed_worker_count)

    def _prepare_runtime_parent_shell_work(
            self,
            *,
            key,
            state: State,
            parent_work: ParentWork,
            adaptation_context,
            max_workers: int,
    ) -> list[dict[str, object]]:
        work_items: list[dict[str, object]] = []
        num_samples = int(state.num_samples)
        active_log_likelihoods = np.asarray(
            state.samples.log_likelihoods[:num_samples],
        )
        sortable_log_likelihoods = np.where(
            np.isnan(active_log_likelihoods),
            -np.inf,
            active_log_likelihoods,
        )
        sorted_active_offsets = np.argsort(
            sortable_log_likelihoods,
            kind="stable",
        )
        sorted_active_log_likelihoods = (
            sortable_log_likelihoods[sorted_active_offsets]
        )
        sample_keys = jax.random.split(
            key,
            int(parent_work.parent_idxs.shape[0]),
        )
        for work_idx, sample_key in enumerate(sample_keys):
            seed_key, sampler_key = jax.random.split(sample_key, 2)
            constraint = parent_work.parent_log_L_constraints[work_idx]
            constraint_value = float(constraint)
            if np.isnan(constraint_value):
                first_seed_offset = num_samples
            else:
                first_seed_offset = int(
                    np.searchsorted(
                        sorted_active_log_likelihoods,
                        constraint_value,
                        side="right",
                    )
                )
            no_seed = first_seed_offset == num_samples
            if no_seed:
                constraint = jnp.asarray(
                    -jnp.inf,
                    dtype=mp_policy.measure_dtype,
                )
                first_seed_offset = int(
                    np.searchsorted(
                        sorted_active_log_likelihoods,
                        float(constraint),
                        side="right",
                    )
                )

            candidate_count = num_samples - first_seed_offset
            seed_choice = jax.random.randint(
                seed_key,
                (),
                minval=0,
                maxval=candidate_count,
            )
            seed_offset = first_seed_offset + int(seed_choice)
            seed_idx = int(sorted_active_offsets[seed_offset])
            seed_point = SeedPoint(
                U0=jax.tree.map(
                    lambda u: u[seed_idx],
                    state.samples.U_samples,
                ),
                log_L0=state.samples.log_likelihoods[seed_idx],
            )
            requested_parent_idx = int(parent_work.parent_idxs[work_idx])
            effective_parent_idx = -1 if no_seed else requested_parent_idx
            dispatch_record = self.prepare_runtime_dispatch(
                requested_parent_idx=requested_parent_idx,
                effective_parent_idx=effective_parent_idx,
                accepted_parent_idx=effective_parent_idx,
                effective_log_L_constraint=float(constraint),
                accepted_log_L_constraint=float(constraint),
                seed_id="seed-runtime-worker",
                phantom_cluster_id="phantom-cluster-runtime-worker",
            )
            work_items.append(
                {
                    "sampler_key": sampler_key,
                    "constraint": constraint,
                    "seed_point": seed_point,
                    "requested_parent_idx": requested_parent_idx,
                    "effective_parent_idx": effective_parent_idx,
                    "target_block_idx": int(
                        parent_work.target_block_idxs[work_idx]
                    ),
                    "parent_block_idx": (
                        -1
                        if no_seed
                        else int(parent_work.parent_block_idxs[work_idx])
                    ),
                    "fallback_to_root": bool(
                        no_seed
                        or bool(parent_work.fallback_to_root[work_idx])
                    ),
                    "adaptation_context": adaptation_context,
                    "dispatch_record": dispatch_record,
                }
            )
        return work_items

    def _execute_runtime_parent_work_item(
            self,
            item: dict[str, object],
    ) -> WorkerResult:
        dispatch_record = item["dispatch_record"]
        proposal_log_likelihood_fn = (
            self._proposal_dispatch_likelihood_fn(dispatch_record)
        )
        sampler_adaptation_context = (
            self._sampler_dispatch_adaptation_context(
                adaptation_context=item["adaptation_context"],
                proposal_log_likelihood_fn=proposal_log_likelihood_fn,
            )
        )
        sampler_start = time.perf_counter()
        output = self.sampler.get_sample(
            item["sampler_key"],
            item["constraint"],
            item["seed_point"],
            args=self.args,
            params=self.params,
            adaptation_context=sampler_adaptation_context,
        )
        sampler_end = time.perf_counter()
        worker_execution = _WorkerTaskExecutionStats(
            output=ConstrainedSamplerCompletionPayload.from_output(output),
            sampler_loop_mode="python",
            dispatch_latency_seconds=_nonnegative_elapsed_seconds(
                sampler_start,
                sampler_end,
            ),
            payload_cache_latency_seconds=0.0,
            sampler_execution_latency_seconds=_nonnegative_elapsed_seconds(
                sampler_start,
                sampler_end,
            ),
        )
        result_identity = WorkerResultIdentity(
            task_id=str(
                self._dispatch_record_field(dispatch_record, "task_id")
            ),
            attempt_id=str(
                self._dispatch_record_field(dispatch_record, "attempt_id")
            ),
            transport_id=str(
                self._dispatch_record_field(dispatch_record, "transport_id")
            ),
            worker_id=str(
                self._dispatch_record_field(dispatch_record, "worker_id")
            ),
            sector_id=str(
                self._dispatch_record_field(dispatch_record, "sector_id")
            ),
        )
        return WorkerResult(
            identity=result_identity,
            payload=worker_execution,
        )

    def _assemble_runtime_parent_shell_samples(
            self,
            *,
            state: State,
            work_items: list[dict[str, object]],
            outputs: list[tuple[object, object, object, PhantomSamples]],
    ) -> tuple[ParentWork, Samples]:
        U_samples = jax.tree.map(
            lambda *values: jnp.stack(values, axis=0),
            *[output[0] for output in outputs],
        )
        phantom_outputs = [output[3] for output in outputs]
        first_phantom = phantom_outputs[0]
        if first_phantom.U_samples is None:
            phantom_U_samples = _phantom_coordinates_like_state(
                state,
                batch_size=len(outputs),
                num_phantom=int(first_phantom.log_L.shape[0]),
            )
        else:
            phantom_U_samples = jax.tree.map(
                lambda *values: jnp.stack(values, axis=0),
                *[phantom.U_samples for phantom in phantom_outputs],
            )
        adjusted_parent_work = ParentWork(
            parent_idxs=jnp.asarray(
                [item["effective_parent_idx"] for item in work_items],
                dtype=mp_policy.index_dtype,
            ),
            parent_log_L_constraints=jnp.asarray(
                [item["constraint"] for item in work_items],
                dtype=mp_policy.measure_dtype,
            ),
            target_block_idxs=jnp.asarray(
                [item["target_block_idx"] for item in work_items],
                dtype=mp_policy.index_dtype,
            ),
            parent_block_idxs=jnp.asarray(
                [item["parent_block_idx"] for item in work_items],
                dtype=mp_policy.index_dtype,
            ),
            fallback_to_root=jnp.asarray(
                [item["fallback_to_root"] for item in work_items],
                dtype=mp_policy.bool_dtype,
            ),
        )
        new_samples = Samples(
            log_L_constraints=adjusted_parent_work.parent_log_L_constraints,
            log_likelihoods=jnp.stack(
                [output[1] for output in outputs],
                axis=0,
            ),
            U_samples=U_samples,
            out_degree=jnp.zeros(
                (len(outputs),),
                dtype=mp_policy.count_dtype,
            ),
            num_likelihood_evaluations=jnp.stack(
                [output[2] for output in outputs],
                axis=0,
            ).astype(mp_policy.count_dtype),
            phantom_samples=PhantomSamples(
                U_samples=phantom_U_samples,
                valid_mask=jnp.stack(
                    [phantom.valid_mask for phantom in phantom_outputs],
                    axis=0,
                ),
                log_L=jnp.stack(
                    [phantom.log_L for phantom in phantom_outputs],
                    axis=0,
                ),
            ),
        )
        if (
                state.samples.phantom_samples.U_samples is None
                and not self.store_phantom_samples
        ):
            new_samples.phantom_samples.U_samples = None
        return adjusted_parent_work, new_samples

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
        self._require_runtime_context()
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
        proposal_log_likelihood_fn = (
            self._proposal_dispatch_likelihood_fn(dispatch_record)
        )
        sampler_adaptation_context = (
            self._sampler_dispatch_adaptation_context(
                adaptation_context=adaptation_context,
                proposal_log_likelihood_fn=proposal_log_likelihood_fn,
            )
        )
        sampler_start = time.perf_counter()
        try:
            if isinstance(self.sampler, UniDimSliceSampler):
                output = self.sampler.get_sample(
                    key,
                    log_L_constraint,
                    seed_point,
                    args=self.args,
                    params=self.params,
                    adaptation_context=sampler_adaptation_context,
                )
            else:
                output = NestedSampler._sample_constrained(
                    self,
                    key,
                    log_L_constraint,
                    seed_point,
                    requested_parent_idx=requested_parent_idx,
                    effective_parent_idx=effective_parent_idx,
                    accepted_parent_idx=accepted_parent_idx,
                    adaptation_context=adaptation_context,
                )
        except Exception as error:
            self.mark_runtime_dispatch_failed(
                dispatch_record=dispatch_record,
                error=error,
            )
            raise
        sampler_end = time.perf_counter()
        worker_execution = _WorkerTaskExecutionStats(
            output=ConstrainedSamplerCompletionPayload.from_output(output),
            sampler_loop_mode="python",
            dispatch_latency_seconds=_nonnegative_elapsed_seconds(
                sampler_start,
                sampler_end,
            ),
            payload_cache_latency_seconds=0.0,
            sampler_execution_latency_seconds=_nonnegative_elapsed_seconds(
                sampler_start,
                sampler_end,
            ),
        )
        result_identity = WorkerResultIdentity(
            task_id=str(
                self._dispatch_record_field(dispatch_record, "task_id")
            ),
            attempt_id=str(
                self._dispatch_record_field(dispatch_record, "attempt_id")
            ),
            transport_id=str(
                self._dispatch_record_field(dispatch_record, "transport_id")
            ),
            worker_id=str(
                self._dispatch_record_field(dispatch_record, "worker_id")
            ),
            sector_id=str(
                self._dispatch_record_field(dispatch_record, "sector_id")
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
        return output

    def _proposal_dispatch_likelihood_fn(self, dispatch_record: object):
        task_id = str(self._dispatch_record_field(dispatch_record, "task_id"))
        attempt_id = str(
            self._dispatch_record_field(dispatch_record, "attempt_id")
        )
        transport_id = str(
            self._dispatch_record_field(dispatch_record, "transport_id")
        )
        eval_counter = 0

        def log_likelihood_fn(U):
            nonlocal eval_counter
            eval_counter += 1
            response = self.dispatch_likelihood_eval(
                U=U,
                task_id=task_id,
                attempt_id=attempt_id,
                transport_id=transport_id,
                eval_id=f"{transport_id}-eval-{eval_counter:06d}",
            )
            if response.status != "ok":
                raise RuntimeError(
                    "Dispatched likelihood evaluation failed: "
                    f"{response.error_type}: {response.error_message}"
                )
            return response.log_L

        return log_likelihood_fn

    @staticmethod
    def _sampler_dispatch_adaptation_context(
            *,
            adaptation_context,
            proposal_log_likelihood_fn,
    ) -> dict[str, object]:
        return {
            "force_python_loop": True,
            "sampler_loop_mode": "python",
            "direction_adaptation_context": adaptation_context,
            "proposal_log_likelihood_fn": proposal_log_likelihood_fn,
        }

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
                self._ensure_current_runtime_compile_identity()
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
        self._supersede_live_dispatch(metadata)
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
            result: object,
            current_parent_idx: int,
            current_effective_log_L_constraint: float,
    ) -> CoordinatorLifecycleRecord:
        metadata = self._lifecycle_metadata_from_record(dispatch_record)
        self._validate_lifecycle_record_owner(metadata)
        if not isinstance(result, WorkerResult):
            return self._record_lifecycle_event(
                metadata,
                status="invalid_result_payload",
                reason=(
                    "complete_runtime_dispatch requires WorkerResult with a "
                    "validated constrained-sampler completion payload."
                ),
            )
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

        try:
            self._validated_constrained_sampler_completion(result.payload)
        except TypeError as error:
            return self._record_lifecycle_event(
                metadata,
                status="invalid_result_payload",
                reason=str(error),
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

        terminal_status = self._non_live_dispatch_status(metadata)
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

    def _ensure_current_runtime_compile_identity(
            self,
    ) -> RuntimeCompileIdentity:
        lb_state, _, problem_payload = self._require_runtime_context()
        previous_identity = self.runtime_compile_identity
        previous_device_class = (
            None
            if previous_identity is None
            else previous_identity.device_class
        )
        current_device_class = _device_class_from_worker_device_types(
            lb_state.worker_device_types(),
            previous_device_class=previous_device_class,
        )
        if (
                previous_identity is not None
                and previous_identity.device_class == current_device_class
        ):
            return previous_identity
        self.runtime_compile_identity = _build_runtime_compile_identity(
            serialized_problem=problem_payload,
            sampler=self.sampler,
            sampler_bytes=self.runtime_sampler_bytes,
            worker_device_types=(current_device_class,),
            device_class=current_device_class,
        )
        self._runtime_registered_likelihood_identity_digest = None
        return self.runtime_compile_identity

    def _require_runtime_sampler_bytes(self) -> bytes:
        if self.runtime_sampler_bytes is None:
            self.runtime_sampler_bytes = serialize_sampler(self.sampler)
        return self.runtime_sampler_bytes

    def _ensure_likelihood_scheduler(self) -> LikelihoodEvalScheduler:
        lb_state, _, problem_payload = self._require_runtime_context()
        identity = self._ensure_current_runtime_compile_identity()
        requested_specs, scheduler_generation = (
            lb_state.likelihood_scheduler_signature()
        )
        if (
                self._runtime_likelihood_scheduler is not None
                and self._runtime_likelihood_scheduler_specs
                == requested_specs
                and self._runtime_likelihood_scheduler_generation
                == scheduler_generation
                and (
                    self._runtime_registered_likelihood_identity_digest
                    == identity.identity_digest
                )
        ):
            return self._runtime_likelihood_scheduler

        if (
                self._runtime_registered_likelihood_identity_digest
                != identity.identity_digest
        ):
            lb_state.register_likelihood_compile_identity(
                identity=identity,
                serialized_problem=problem_payload,
            )
            self._runtime_registered_likelihood_identity_digest = (
                identity.identity_digest
            )
        scheduler = lb_state.likelihood_eval_scheduler()
        self._runtime_likelihood_scheduler = scheduler
        self._runtime_likelihood_scheduler_specs = (
            scheduler.requested_worker_specs
        )
        self._runtime_likelihood_scheduler_generation = (
            lb_state.likelihood_scheduler_generation()
        )
        return scheduler

    def make_likelihood_eval_request(
            self,
            *,
            U: object,
            task_id: str,
            attempt_id: str,
            transport_id: str,
            eval_id: str,
            deadline_ms: int | None = None,
            requested_dtype_policy: str | None = None,
    ) -> LikelihoodEvalRequest:
        lb_state, runner_identity, _ = self._require_runtime_context()
        del lb_state
        identity = self._ensure_current_runtime_compile_identity()
        request_U = _u_for_dtype_policy(U, identity.dtype_policy)
        return LikelihoodEvalRequest(
            protocol_version=1,
            runner_id=runner_identity.runner_id,
            task_id=str(task_id),
            attempt_id=str(attempt_id),
            transport_id=str(transport_id),
            compile_identity_digest=identity.identity_digest,
            eval_id=str(eval_id),
            U_bytes=pickle_payload(request_U),
            U_shape_tree=identity.U_shape_tree,
            requested_dtype_policy=(
                identity.dtype_policy
                if requested_dtype_policy is None
                else str(requested_dtype_policy)
            ),
            deadline_ms=deadline_ms,
        )

    def prepare_likelihood_eval_request(
            self,
            *,
            U: object,
            task_id: str,
            attempt_id: str,
            transport_id: str,
            eval_id: str,
            deadline_ms: int | None = None,
            requested_dtype_policy: str | None = None,
    ) -> LikelihoodEvalRequest:
        return self.make_likelihood_eval_request(
            U=U,
            task_id=task_id,
            attempt_id=attempt_id,
            transport_id=transport_id,
            eval_id=eval_id,
            deadline_ms=deadline_ms,
            requested_dtype_policy=requested_dtype_policy,
        )

    def dispatch_likelihood_eval(
            self,
            *,
            U: object,
            task_id: str,
            attempt_id: str,
            transport_id: str,
            eval_id: str,
            deadline_ms: int | None = None,
    ) -> LikelihoodEvalResponse:
        request = self.make_likelihood_eval_request(
            U=U,
            task_id=task_id,
            attempt_id=attempt_id,
            transport_id=transport_id,
            eval_id=eval_id,
            deadline_ms=deadline_ms,
        )
        scheduler = self._ensure_likelihood_scheduler()
        self.runtime_likelihood_router.register_pending(request)
        record = scheduler.start_likelihood_eval(request=request)
        if not record.worker_id:
            if not scheduler.has_workers():
                raise RuntimeError(
                    "No likelihood workers are available for local dispatch."
                )
            start_or_cancel = (
                scheduler.wait_for_started_record_or_cancel_likelihood_eval(
                    request,
                    timeout_seconds=(
                        None
                        if deadline_ms is None
                        else max(float(deadline_ms) / 1000.0, 0.0)
                    ),
                    error_type="LikelihoodEvalTimeout",
                    error_message=(
                        "Timed out waiting for a queued likelihood "
                        "evaluation to start."
                    ),
                )
            )
            if isinstance(start_or_cancel, LikelihoodEvalResponse):
                self.runtime_likelihood_router.route_eval_response(
                    start_or_cancel
                )
                return start_or_cancel
            record = start_or_cancel
        worker = scheduler.worker_for_id(record.worker_id)
        response = worker.evaluate_likelihood(request)
        scheduler.complete_likelihood_eval(
            worker_id=record.worker_id,
            response=response,
        )
        self.runtime_likelihood_router.route_eval_response(response)
        return response

    def evaluate_likelihood_via_dispatch(
            self,
            *,
            U: object,
            task_id: str,
            attempt_id: str,
            transport_id: str,
            eval_id: str,
            deadline_ms: int | None = None,
    ) -> LikelihoodEvalResponse:
        return self.dispatch_likelihood_eval(
            U=U,
            task_id=task_id,
            attempt_id=attempt_id,
            transport_id=transport_id,
            eval_id=eval_id,
            deadline_ms=deadline_ms,
        )

    def request_likelihood_eval(
            self,
            *,
            U: object,
            task_id: str,
            attempt_id: str,
            transport_id: str,
            eval_id: str,
            deadline_ms: int | None = None,
    ) -> LikelihoodEvalResponse:
        return self.dispatch_likelihood_eval(
            U=U,
            task_id=task_id,
            attempt_id=attempt_id,
            transport_id=transport_id,
            eval_id=eval_id,
            deadline_ms=deadline_ms,
        )

    def likelihood_dispatch_snapshot(self) -> LikelihoodDispatchSnapshot:
        return self.runtime_likelihood_router.snapshot()

    def get_likelihood_dispatch_snapshot(self) -> LikelihoodDispatchSnapshot:
        return self.likelihood_dispatch_snapshot()

    def runtime_dispatch_snapshot(self) -> LikelihoodDispatchSnapshot:
        return self.likelihood_dispatch_snapshot()

    def get_runtime_dispatch_snapshot(self) -> LikelihoodDispatchSnapshot:
        return self.likelihood_dispatch_snapshot()

    def likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        scheduler = self._ensure_likelihood_scheduler()
        return scheduler.likelihood_dispatch_diagnostics()

    def get_likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()

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
        if status in {"pending", "retried"}:
            self._set_live_dispatch(metadata)
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

    @staticmethod
    def _validated_constrained_sampler_completion(
            payload: object,
    ) -> ConstrainedSamplerCompletionPayload:
        if isinstance(payload, _WorkerTaskExecutionStats):
            payload = payload.output
        if isinstance(payload, LikelihoodEvalResponse):
            raise TypeError(
                "Raw LikelihoodEvalResponse is not a constrained-sampler "
                "child result."
            )
        return ConstrainedSamplerCompletionPayload.from_output(payload)

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
        dispatch_key = self._lifecycle_dispatch_key(metadata)
        with self._runtime_lifecycle_lock:
            self._runtime_terminal_dispatch_status[dispatch_key] = status
            if (
                    self._runtime_live_dispatch_by_task_id.get(
                        metadata.task_id
                    )
                    == dispatch_key
            ):
                self._runtime_live_dispatch_by_task_id.pop(
                    metadata.task_id,
                    None,
                )

    def _terminal_dispatch_status(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> str | None:
        with self._runtime_lifecycle_lock:
            return self._runtime_terminal_dispatch_status.get(
                self._lifecycle_dispatch_key(metadata)
            )

    def _set_live_dispatch(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> None:
        with self._runtime_lifecycle_lock:
            self._runtime_live_dispatch_by_task_id[metadata.task_id] = (
                self._lifecycle_dispatch_key(metadata)
            )

    def _supersede_live_dispatch(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> None:
        dispatch_key = self._lifecycle_dispatch_key(metadata)
        with self._runtime_lifecycle_lock:
            if (
                    self._runtime_live_dispatch_by_task_id.get(
                        metadata.task_id
                    )
                    != dispatch_key
            ):
                return
            if dispatch_key not in self._runtime_terminal_dispatch_status:
                self._runtime_terminal_dispatch_status[
                    dispatch_key
                ] = "stale_task_result"
            self._runtime_live_dispatch_by_task_id.pop(
                metadata.task_id,
                None,
            )

    def _non_live_dispatch_status(
            self,
            metadata: _LifecycleRecordMetadata,
    ) -> str | None:
        dispatch_key = self._lifecycle_dispatch_key(metadata)
        with self._runtime_lifecycle_lock:
            terminal_status = self._runtime_terminal_dispatch_status.get(
                dispatch_key
            )
            if terminal_status is not None:
                return terminal_status
            live_key = self._runtime_live_dispatch_by_task_id.get(
                metadata.task_id
            )
            if live_key is not None and live_key != dispatch_key:
                return "stale_task_result"
            return None

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

    def likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        return self._state.likelihood_dispatch_diagnostics()

    def get_likelihood_dispatch_diagnostics(
            self,
    ) -> LikelihoodDispatchDiagnostics:
        return self.likelihood_dispatch_diagnostics()

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
