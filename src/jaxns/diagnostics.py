from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

_MISSING = object()


@dataclasses.dataclass(frozen=True, slots=True)
class DiagnosticConditionSummary:
    condition_name: str
    iteration: int
    num_samples: int
    value: int | float
    satisfied: bool


@dataclasses.dataclass(frozen=True, slots=True)
class AllocationDiagnostics:
    mode: str
    target_num_live_points: int
    shell_size: int
    target_summaries: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ParentSelectionDiagnostics:
    requested_parent_indices: object
    effective_parent_indices: object
    accepted_parent_indices: object
    sentinel_fallback_count: int
    sentinel_fallback_indices: object


@dataclasses.dataclass(frozen=True, slots=True)
class DepthDiagnostics:
    condition_summaries: tuple[DiagnosticConditionSummary, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class GoalDiagnostics:
    condition_summaries: tuple[DiagnosticConditionSummary, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
    mode: str
    direction_kernel_mode: str
    trajectory_mode: str
    phantom_burn_in: int
    retained_phantom_capacity: int
    retained_phantom_counts_per_sample: object
    likelihood_evaluations_per_classic_sample: object
    likelihood_evaluations_per_retained_phantom_cluster: object
    direction_adaptation_diagnostics: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ComputeSectorSummary:
    sector_id: str
    device_type: str
    device_id: str
    worker_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerRuntimeDiagnostics:
    worker_count: int
    compute_sector_summaries: tuple[object, ...]
    runner_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    requested_parent_indices: object
    effective_parent_indices: object
    accepted_parent_indices: object
    in_flight_parent_targets: tuple[int, ...]
    accepted_task_count: int
    retried_task_count: int
    revoked_task_count: int
    model_compilation_times: tuple[float, ...]
    async_identity_preserved: bool
    dispatch_records: tuple[object, ...] = ()

    @property
    def dispatch_trace(self) -> tuple[object, ...]:
        return self.dispatch_records


@dataclasses.dataclass(frozen=True, slots=True)
class LikelihoodDispatchDiagnostics:
    requested_worker_specs: tuple[str, ...]
    observed_worker_count: int
    observed_worker_device_classes: tuple[str, ...]
    dispatch_eval_count: int
    dispatch_latency_seconds: tuple[float, ...]
    dispatch_throughput_per_second: float
    compile_count: int
    cache_hit_count: int
    rejected_shape_cache_count: int
    distinct_compile_identity_count: int
    max_active_evals_per_worker: object
    max_active_evals_pool: int
    likelihood_eval_records: tuple[object, ...] = ()
    queued_eval_count: int = 0
    failed_eval_count: int = 0
    completed_eval_count_by_worker: object = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    allocation: AllocationDiagnostics
    parent_selection: ParentSelectionDiagnostics
    depth: DepthDiagnostics
    goal: GoalDiagnostics
    sampler: SamplerDiagnostics
    worker_runtime: WorkerRuntimeDiagnostics


def get_execution_diagnostics(results: object) -> ExecutionDiagnostics | None:
    return getattr(results, "execution_diagnostics", None)


def get_diagnostics(results: object) -> ExecutionDiagnostics | None:
    return get_execution_diagnostics(results)


def validate_execution_diagnostics(
        results: object,
        diagnostics: ExecutionDiagnostics,
) -> ExecutionDiagnostics:
    num_samples = int(np.asarray(getattr(results, "total_num_samples")))
    target_live = int(diagnostics.allocation.target_num_live_points)
    num_dispatches = max(0, num_samples - target_live)

    _require_leading_shape(
        diagnostics.parent_selection.requested_parent_indices,
        num_dispatches,
        "parent_selection.requested_parent_indices",
    )
    _require_leading_shape(
        diagnostics.parent_selection.effective_parent_indices,
        num_dispatches,
        "parent_selection.effective_parent_indices",
    )
    _require_leading_shape(
        diagnostics.parent_selection.accepted_parent_indices,
        num_dispatches,
        "parent_selection.accepted_parent_indices",
    )
    sentinel_fallback_indices = np.asarray(
        diagnostics.parent_selection.sentinel_fallback_indices
    )
    sentinel_fallback_count = int(
        diagnostics.parent_selection.sentinel_fallback_count
    )
    if sentinel_fallback_count != int(sentinel_fallback_indices.shape[0]):
        raise ValueError(
            "diagnostic sentinel align error: "
            "parent_selection.sentinel_fallback_count must match "
            "sentinel_fallback_indices length."
        )
    _require_leading_shape(
        diagnostics.sampler.retained_phantom_counts_per_sample,
        num_samples,
        "sampler.retained_phantom_counts_per_sample",
    )
    _require_leading_shape(
        diagnostics.sampler.likelihood_evaluations_per_classic_sample,
        num_samples,
        "sampler.likelihood_evaluations_per_classic_sample",
    )
    phantom_eval_shape = np.asarray(
        diagnostics.sampler
        .likelihood_evaluations_per_retained_phantom_cluster
    ).shape
    if len(phantom_eval_shape) < 1 or phantom_eval_shape[0] != num_samples:
        raise ValueError(
            "diagnostic shape align error: "
            "sampler.likelihood_evaluations_per_retained_phantom_cluster "
            f"has shape {phantom_eval_shape}, expected leading dimension "
            f"{num_samples}."
        )
    _validate_worker_runtime_diagnostics(diagnostics.worker_runtime)
    return diagnostics


def attach_execution_diagnostics(
        results: object,
        diagnostics: ExecutionDiagnostics,
):
    validate_execution_diagnostics(results, diagnostics)
    return dataclasses.replace(results, execution_diagnostics=diagnostics)


def with_execution_diagnostics(
        results: object,
        diagnostics: ExecutionDiagnostics,
):
    return attach_execution_diagnostics(results, diagnostics)


def _require_leading_shape(value: Any, expected: int, field_name: str) -> None:
    shape = np.asarray(value).shape
    if len(shape) == 0 or shape[0] != expected:
        raise ValueError(
            "diagnostic shape align error: "
            f"{field_name} has shape {shape}, expected leading dimension "
            f"{expected}."
        )


def _validate_worker_runtime_diagnostics(
        worker_runtime: WorkerRuntimeDiagnostics,
) -> None:
    records = tuple(worker_runtime.dispatch_records)
    runner_ids = tuple(worker_runtime.runner_ids)
    task_ids = tuple(worker_runtime.task_ids)
    requested = np.asarray(worker_runtime.requested_parent_indices)
    effective = np.asarray(worker_runtime.effective_parent_indices)
    accepted = np.asarray(worker_runtime.accepted_parent_indices)
    in_flight = tuple(worker_runtime.in_flight_parent_targets)
    _require_vector(requested, "worker_runtime.requested_parent_indices")
    _require_vector(effective, "worker_runtime.effective_parent_indices")
    _require_vector(accepted, "worker_runtime.accepted_parent_indices")
    _validate_model_compilation_times(
        worker_runtime.model_compilation_times
    )
    field_lengths = {
        "worker_runtime.runner_ids": len(runner_ids),
        "worker_runtime.task_ids": len(task_ids),
        "worker_runtime.requested_parent_indices": _leading_len(requested),
        "worker_runtime.effective_parent_indices": _leading_len(effective),
        "worker_runtime.accepted_parent_indices": _leading_len(accepted),
        "worker_runtime.in_flight_parent_targets": len(in_flight),
    }
    if records:
        field_lengths["worker_runtime.dispatch_records"] = len(records)
    lengths = set(field_lengths.values())
    if len(lengths) != 1:
        raise ValueError(
            "diagnostic worker-runtime align error: "
            f"misaligned field lengths {field_lengths}."
        )
    if not records:
        if any(field_lengths.values()):
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "worker runtime arrays require dispatch_records."
            )
        return

    normalised_statuses = []
    for idx, record in enumerate(records):
        _require_dispatch_record_field(
            record,
            "identity_owner",
            expected="coordinator",
        )
        runner_id = str(_require_dispatch_record_field(record, "runner_id"))
        task_id = str(_require_dispatch_record_field(record, "task_id"))
        attempt_id = str(_require_dispatch_record_field(record, "attempt_id"))
        transport_id = str(
            _require_dispatch_record_field(record, "transport_id")
        )
        if not runner_id or not task_id or not attempt_id or not transport_id:
            raise ValueError(
                "diagnostic dispatch record identity error: "
                "runner_id, task_id, attempt_id, and transport_id must be "
                "non-empty."
            )
        if runner_ids[idx] != runner_id or task_ids[idx] != task_id:
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "runner/task ids must align with dispatch_records."
            )
        _require_dispatch_record_field(
            record,
            "requested_parent_index",
            "requested_parent_idx",
        )
        _require_dispatch_record_field(
            record,
            "effective_parent_index",
            "effective_parent_idx",
        )
        _require_dispatch_record_field(
            record,
            "accepted_parent_index",
            "accepted_parent_idx",
        )
        _require_dispatch_record_field(record, "in_flight_parent_target")
        if int(requested[idx]) != int(
                _require_dispatch_record_field(
                    record,
                    "requested_parent_index",
                    "requested_parent_idx",
                )
        ):
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "requested parent indices must align with dispatch_records."
            )
        if int(effective[idx]) != int(
                _require_dispatch_record_field(
                    record,
                    "effective_parent_index",
                    "effective_parent_idx",
                )
        ):
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "effective parent indices must align with dispatch_records."
            )
        if int(accepted[idx]) != int(
                _require_dispatch_record_field(
                    record,
                    "accepted_parent_index",
                    "accepted_parent_idx",
                )
        ):
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "accepted parent indices must align with dispatch_records."
            )
        if int(in_flight[idx]) != int(
                _require_dispatch_record_field(
                    record,
                    "in_flight_parent_target",
                )
        ):
            raise ValueError(
                "diagnostic worker-runtime align error: "
                "in-flight parent targets must align with dispatch_records."
            )
        normalised_statuses.append(
            _normalise_runtime_status(
                _require_dispatch_record_field(record, "status")
            )
        )

    status_counts = {
        status: sum(item == status for item in normalised_statuses)
        for status in ("accepted", "retried", "revoked")
    }
    if int(worker_runtime.accepted_task_count) != status_counts["accepted"]:
        raise ValueError(
            "diagnostic worker-runtime align error: "
            "accepted_task_count must match dispatch_records."
        )
    if int(worker_runtime.retried_task_count) != status_counts["retried"]:
        raise ValueError(
            "diagnostic worker-runtime align error: "
            "retried_task_count must match dispatch_records."
        )
    if int(worker_runtime.revoked_task_count) != status_counts["revoked"]:
        raise ValueError(
            "diagnostic worker-runtime align error: "
            "revoked_task_count must match dispatch_records."
        )
    expected_identity = all(
        _dispatch_field(record, "identity_owner") == "coordinator"
        and bool(str(_dispatch_field(record, "runner_id")))
        and bool(str(_dispatch_field(record, "task_id")))
        and bool(str(_dispatch_field(record, "attempt_id")))
        and bool(str(_dispatch_field(record, "transport_id")))
        for record in records
    )
    if bool(worker_runtime.async_identity_preserved) != expected_identity:
        raise ValueError(
            "diagnostic worker-runtime identity error: "
            "async_identity_preserved must be derived from dispatch_records."
        )


def _leading_len(value: np.ndarray) -> int:
    if len(value.shape) == 0:
        return 0
    return int(value.shape[0])


def _require_vector(value: np.ndarray, field_name: str) -> None:
    if len(value.shape) != 1:
        raise ValueError(
            "diagnostic worker-runtime shape error: "
            f"{field_name} must be a one-dimensional array."
        )


def _validate_model_compilation_times(values: object) -> None:
    timings = np.asarray(tuple(values), dtype=np.float64)
    if timings.size == 0:
        return
    if not np.all(np.isfinite(timings)) or not np.all(timings > 0.0):
        raise ValueError(
            "diagnostic worker-runtime timing error: "
            "model_compilation_times must be finite positive measured "
            "durations, or an empty tuple when unavailable."
        )


def _dispatch_field(record: object, *field_names: str) -> object:
    for field_name in field_names:
        if isinstance(record, dict) and field_name in record:
            return record[field_name]
        if hasattr(record, field_name):
            return getattr(record, field_name)
    return _MISSING


def _require_dispatch_record_field(
        record: object,
        *field_names: str,
        expected: object = _MISSING,
) -> object:
    value = _dispatch_field(record, *field_names)
    if value is _MISSING:
        field_display = "/".join(
            repr(field_name)
            for field_name in field_names
        )
        raise ValueError(
            "diagnostic dispatch record completeness error: "
            f"missing {field_display}."
        )
    if expected is not _MISSING and value != expected:
        field_display = "/".join(
            repr(field_name)
            for field_name in field_names
        )
        raise ValueError(
            "diagnostic dispatch record identity error: "
            f"{field_display} must be {expected!r}."
        )
    return value


def _normalise_runtime_status(status: object) -> str:
    status = str(status).lower()
    if status in {"accept", "accepted", "complete", "completed"}:
        return "accepted"
    if status in {"retry", "retried"}:
        return "retried"
    if status in {"revoke", "revoked", "cancel", "cancelled"}:
        return "revoked"
    return status
