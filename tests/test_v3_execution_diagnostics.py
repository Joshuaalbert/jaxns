from __future__ import annotations

import dataclasses
import importlib
import inspect
import types
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.results import NestedSamplerResults
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.termination_condition import TerminationRegister


PUBLIC_DIAGNOSTIC_MODULE_NAMES = (
    "jaxns.diagnostics",
    "jaxns.results",
)
PUBLIC_DIAGNOSTIC_TYPE_NAMES = (
    "ExecutionDiagnostics",
    "AllocationDiagnostics",
    "ParentSelectionDiagnostics",
    "DepthDiagnostics",
    "GoalDiagnostics",
    "SamplerDiagnostics",
    "WorkerRuntimeDiagnostics",
    "DiagnosticConditionSummary",
)
PUBLIC_DIAGNOSTIC_ACCESSOR_NAMES = (
    "get_execution_diagnostics",
    "get_diagnostics",
)
PUBLIC_DIAGNOSTIC_BOUNDARY_NAMES = (
    "attach_execution_diagnostics",
    "with_execution_diagnostics",
    "validate_execution_diagnostics",
)
PUBLIC_DISPATCH_TRACE_CONSTRUCTOR_NAMES = (
    "dispatch_trace",
    "dispatch_ledger",
    "coordinator_dispatch_trace",
    "coordinator_dispatch_ledger",
)
PUBLIC_DISPATCH_TRACE_SETTER_NAMES = (
    "set_dispatch_trace",
    "set_dispatch_ledger",
    "set_coordinator_dispatch_trace",
    "set_coordinator_dispatch_ledger",
)
PUBLIC_DISPATCH_TRACE_ATTRIBUTE_NAMES = (
    "dispatch_trace",
    "dispatch_ledger",
    "coordinator_dispatch_trace",
    "coordinator_dispatch_ledger",
)
PUBLIC_WORKER_RUNTIME_DISPATCH_RECORD_NAMES = (
    "dispatch_trace",
    "dispatch_ledger",
    "dispatch_records",
    "task_trace",
    "task_ledger",
    "task_records",
)
STATISTICAL_RESULT_FIELD_NAMES = (
    "log_Z_mean",
    "log_Z_uncert",
    "ess",
    "H_mean",
    "log_dp",
    "log_X_mean",
    "log_posterior_density",
)
STATISTICAL_RESULT_SNAPSHOT_FIELD_NAMES = (
    "total_num_samples",
    "total_phantom_samples",
    "log_Z_mean",
    "log_Z_uncert",
    "ess",
    "H_mean",
    "log_L",
    "log_L_constraints",
    "log_L_phantom",
    "valid_phantom",
    "log_dp",
    "log_X_mean",
    "log_posterior_density",
    "num_live_points_per_sample",
    "num_likelihood_evaluations_per_sample",
)


@dataclasses.dataclass(frozen=True, slots=True)
class ToyModel(PureDataclassPytree):
    centre: float = 0.25

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        return -jnp.square(jnp.asarray(U) - self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


ToyModel.register_pytree()


class GoalObservation(NamedTuple):
    iteration: int
    state_num_samples: int
    satisfied: bool


class DepthConditionObservation(NamedTuple):
    condition_name: str
    iteration: int
    num_samples: int
    value: int
    satisfied: bool


class DispatchRecord(NamedTuple):
    dispatch_sequence: int
    runner_id: str
    task_id: str
    attempt_id: str
    transport_id: str
    requested_parent_index: int
    in_flight_parent_target: int
    effective_parent_index: int
    accepted_parent_index: int
    status: str
    identity_owner: str


class AllocationDiagnosticsCase(NamedTuple):
    allocation_target: str
    collect_phantoms: bool


class DispatchTraceProbe:
    """Public fixture used to prove dispatch ids come from the coordinator."""

    def __init__(self):
        self.records: list[object] = []

    def record_dispatch(self, record=None, **fields) -> None:
        if record is not None and fields:
            raise ValueError(
                "Provide either a record object or keyword fields."
            )
        self.records.append(fields if record is None else record)

    def append(self, record) -> None:
        self.records.append(record)

    def extend(self, records) -> None:
        self.records.extend(records)


def _available_public_diagnostics_modules() -> tuple[object, ...]:
    modules = []
    for module_name in PUBLIC_DIAGNOSTIC_MODULE_NAMES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        modules.append(module)
    return tuple(modules)


def _assert_public_diagnostics_symbol(symbol, symbol_name: str) -> None:
    module_name = getattr(symbol, "__module__", "")
    assert module_name.startswith("jaxns."), symbol_name
    assert "._" not in module_name, symbol_name


def _public_diagnostics_type(type_name: str):
    missing_modules = []
    for module in _available_public_diagnostics_modules():
        public_type = getattr(module, type_name, None)
        if public_type is None:
            missing_modules.append(module.__name__)
            continue
        _assert_public_diagnostics_symbol(public_type, type_name)
        return public_type

    if not missing_modules:
        missing_modules = list(PUBLIC_DIAGNOSTIC_MODULE_NAMES)
    pytest.fail(
        f"{type_name} must be reachable from a public diagnostics/reporting "
        f"module; checked modules: {tuple(missing_modules)!r}."
    )


def _public_diagnostics_types():
    return types.SimpleNamespace(
        **{
            type_name: _public_diagnostics_type(type_name)
            for type_name in PUBLIC_DIAGNOSTIC_TYPE_NAMES
        }
    )


def _make_nested_sampler(
        *,
        collect_phantoms: bool,
        phantom_burn_in: int = 1,
        max_samples: int = 8,
) -> NestedSampler:
    model = ToyModel()
    return NestedSampler(
        model=model,
        sampler=UniDimSliceSampler(
            model=model,
            num_slices=3,
            no_step_out=True,
            gradient_guided=False,
            collect_phantom_samples=collect_phantoms,
            phantom_burn_in=phantom_burn_in,
        ),
        target_num_live_points=2,
        max_samples=max_samples,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=max_samples),
        batch_size=None,
    )


def _goal_at_min_sample_count(
        observations: list[GoalObservation],
        min_num_samples: int,
) -> Callable[[State], bool]:
    def goal_cond(state: State) -> bool:
        num_samples = int(state.num_samples)
        satisfied = num_samples >= min_num_samples
        observations.append(
            GoalObservation(
                iteration=len(observations),
                state_num_samples=num_samples,
                satisfied=satisfied,
            )
        )
        return satisfied

    return goal_cond


def _run_result_with_observations(
        ns: NestedSampler,
        *,
        allocation_target: str,
        depth_max_samples: int = 4,
        depth_cond: TerminationCondition | None = None,
) -> tuple[State, NestedSamplerResults, list[GoalObservation]]:
    observations: list[GoalObservation] = []
    if depth_cond is None:
        depth_cond = TerminationCondition(max_samples=depth_max_samples)
    state = ns.run_until_goal(
        goal_cond=_goal_at_min_sample_count(
            observations,
            min_num_samples=depth_max_samples,
        ),
        depth_cond=depth_cond,
        allocation_target=allocation_target,
        key=jax.random.PRNGKey(17),
        max_goal_iterations=3,
    )
    return state, state.to_result().trim(), observations


def _assert_diagnostics_sections(diagnostics):
    for section_name in (
            "allocation",
            "parent_selection",
            "depth",
            "goal",
            "sampler",
            "worker_runtime",
    ):
        assert hasattr(diagnostics, section_name), section_name


def _public_diagnostics_from_results(results: NestedSamplerResults):
    for module in _available_public_diagnostics_modules():
        for accessor_name in PUBLIC_DIAGNOSTIC_ACCESSOR_NAMES:
            accessor = getattr(module, accessor_name, None)
            if callable(accessor):
                diagnostics = accessor(results)
                if diagnostics is not None:
                    return diagnostics

    for method_name in PUBLIC_DIAGNOSTIC_ACCESSOR_NAMES:
        method = getattr(results, method_name, None)
        if callable(method):
            diagnostics = method()
            if diagnostics is not None:
                return diagnostics

    diagnostics = getattr(results, "diagnostics", None)
    if diagnostics is not None:
        return diagnostics

    pytest.fail(
        "Execution diagnostics must be reachable from public results through "
        "a public accessor or attached diagnostics object, without requiring "
        "private execution/runtime imports."
    )


def _require_diagnostics(results: NestedSamplerResults):
    diagnostics = _public_diagnostics_from_results(results)
    _assert_diagnostics_sections(diagnostics)
    return diagnostics


def _apply_public_diagnostics_boundary(
        results: NestedSamplerResults,
        diagnostics,
):
    for module in _available_public_diagnostics_modules():
        for boundary_name in PUBLIC_DIAGNOSTIC_BOUNDARY_NAMES:
            boundary = getattr(module, boundary_name, None)
            if callable(boundary):
                return boundary(results, diagnostics)

    for method_name in PUBLIC_DIAGNOSTIC_BOUNDARY_NAMES:
        method = getattr(results, method_name, None)
        if callable(method):
            return method(diagnostics)

    pytest.fail(
        "Malformed diagnostics must be validated by a public "
        "diagnostics/result boundary instead of by private hooks or the "
        "NestedSamplerResults constructor."
    )


def _assert_empty_worker_runtime(worker_runtime) -> None:
    explicit_empty_fields = (
        "compute_sector_summaries",
        "runner_ids",
        "task_ids",
        "requested_parent_indices",
        "effective_parent_indices",
        "accepted_parent_indices",
        "in_flight_parent_targets",
        "model_compilation_times",
    )
    for field_name in explicit_empty_fields:
        value = getattr(worker_runtime, field_name)
        assert value is not None, field_name
        assert len(value) == 0, field_name

    saw_dispatch_records = False
    for field_name in PUBLIC_WORKER_RUNTIME_DISPATCH_RECORD_NAMES:
        if hasattr(worker_runtime, field_name):
            saw_dispatch_records = True
            dispatch_records = _coerce_dispatch_records(
                getattr(worker_runtime, field_name)
            )
            assert len(dispatch_records) == 0
    assert saw_dispatch_records, (
        "WorkerRuntimeDiagnostics must expose an explicit empty public "
        "dispatch ledger/trace for local runs."
    )

    assert int(worker_runtime.worker_count) == 0
    assert int(worker_runtime.accepted_task_count) == 0
    assert int(worker_runtime.retried_task_count) == 0
    assert int(worker_runtime.revoked_task_count) == 0
    assert bool(worker_runtime.async_identity_preserved)


def _as_tuple(value) -> tuple:
    array = np.asarray(value)
    if array.shape == ():
        return (array.item(),)
    return tuple(array.tolist())


def _scalar_int(value, field_name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        pytest.fail(f"{field_name} must be scalar, got shape {array.shape}.")
    return int(array.item())


def _record_depth_condition_observations(
        monkeypatch,
        observations: list[DepthConditionObservation],
) -> None:
    original_is_done = TerminationRegister.is_done

    def append_observation(
            condition_name: str,
            num_active_conditions: int,
            num_samples,
            value,
            satisfied,
    ) -> None:
        observations.append(
            DepthConditionObservation(
                condition_name=condition_name,
                iteration=len(observations) // num_active_conditions,
                num_samples=_scalar_int(num_samples, "depth.num_samples"),
                value=_scalar_int(value, "depth.value"),
                satisfied=bool(np.asarray(satisfied).item()),
            )
        )

    def recording_is_done(self, term_cond):
        result = original_is_done(self, term_cond)
        active_conditions = sum(
            condition is not None
            for condition in (
                term_cond.max_samples,
                term_cond.max_num_likelihood_evaluations,
            )
        )

        def callback_for(condition_name: str):
            def callback(num_samples, value, satisfied) -> None:
                append_observation(
                    condition_name,
                    active_conditions,
                    num_samples,
                    value,
                    satisfied,
                )

            return callback

        if term_cond.max_samples is not None:
            jax.debug.callback(
                callback_for("max_samples"),
                self.num_samples_used,
                self.num_samples_used,
                self.num_samples_used >= term_cond.max_samples,
                ordered=True,
            )

        if term_cond.max_num_likelihood_evaluations is not None:
            jax.debug.callback(
                callback_for("max_num_likelihood_evaluations"),
                self.num_samples_used,
                self.num_likelihood_evaluations,
                (
                    self.num_likelihood_evaluations
                    >= term_cond.max_num_likelihood_evaluations
                ),
                ordered=True,
            )

        return result

    monkeypatch.setattr(TerminationRegister, "is_done", recording_is_done)
    jax.clear_caches()


def _normalise_depth_summary(summary) -> DepthConditionObservation:
    return DepthConditionObservation(
        condition_name=str(summary.condition_name),
        iteration=_scalar_int(summary.iteration, "depth.summary.iteration"),
        num_samples=_scalar_int(
            summary.num_samples,
            "depth.summary.num_samples",
        ),
        value=_scalar_int(summary.value, "depth.summary.value"),
        satisfied=bool(summary.satisfied),
    )


def _record_mapping(record) -> dict[str, object]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "_asdict"):
        return dict(record._asdict())
    if dataclasses.is_dataclass(record):
        return {
            field.name: getattr(record, field.name)
            for field in dataclasses.fields(record)
        }
    return {}


def _record_field(
        record,
        *field_names: str,
        default=dataclasses.MISSING,
) -> object:
    mapping = _record_mapping(record)
    for field_name in field_names:
        if field_name in mapping:
            return mapping[field_name]
        if hasattr(record, field_name):
            return getattr(record, field_name)
    if default is not dataclasses.MISSING:
        return default
    pytest.fail(
        "Dispatch ledger records must expose one of "
        f"{field_names!r}; got {record!r}."
    )


def _allocation_summary_mode_matches(value, allocation_target: str) -> bool:
    normalised = str(value).strip().lower().replace("-", "_")
    return normalised == allocation_target


def _is_numeric_payload(value) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.size > 0 and array.dtype.kind in "iuf"


def _numeric_payloads_by_name(
        summaries: tuple[object, ...],
        name_predicate: Callable[[str], bool],
) -> list[tuple[str, np.ndarray]]:
    payloads = []
    for summary in summaries:
        mapping = _record_mapping(summary)
        for field_name, value in mapping.items():
            if not name_predicate(field_name.lower()):
                continue
            if not _is_numeric_payload(value):
                continue
            payloads.append((field_name, np.asarray(value)))
    return payloads


def _assert_numeric_payloads_are_finite(
        payloads: list[tuple[str, np.ndarray]],
        context: str,
) -> None:
    assert payloads, context
    for field_name, array in payloads:
        numeric = np.asarray(array, dtype=np.float64)
        assert np.all(np.isfinite(numeric)), field_name


def _assert_allocation_target_summaries_are_meaningful(
        diagnostics,
        allocation_target: str,
) -> None:
    assert diagnostics.allocation.mode == allocation_target
    summaries = tuple(diagnostics.allocation.target_summaries)
    assert summaries, (
        "Allocation diagnostics must expose at least one target summary for "
        f"{allocation_target!r} runs."
    )

    summary_mappings = []
    for summary in summaries:
        mapping = _record_mapping(summary)
        assert mapping, (
            "Allocation target summaries must be structured public records, "
            f"got {summary!r}."
        )
        summary_mappings.append(mapping)

    mode_field_names = (
        "mode",
        "allocation_mode",
        "allocation_target",
        "target_mode",
        "utility_mode",
        "strategy",
    )
    mode_values = [
        mapping[field_name]
        for mapping in summary_mappings
        for field_name in mode_field_names
        if field_name in mapping
    ]
    assert any(
        _allocation_summary_mode_matches(value, allocation_target)
        for value in mode_values
    ), (
        "Allocation target summaries must identify the allocation mode that "
        f"produced them; saw {mode_values!r}."
    )

    ignored_target_fields = {
        "allocation_target",
        "mode",
        "target_mode",
        "target_num_live_points",
    }
    target_payloads = _numeric_payloads_by_name(
        summaries,
        lambda name: (
            name not in ignored_target_fields
            and (
                "target" in name
                or "allocated" in name
                or "lineage" in name
                or "live_point" in name
            )
        ),
    )
    _assert_numeric_payloads_are_finite(
        target_payloads,
        "Allocation target summaries must include finite numeric target or "
        f"lineage payloads for {allocation_target!r}.",
    )
    assert any(
        np.any(np.asarray(array, dtype=np.float64) > 0.0)
        for _, array in target_payloads
    ), (
        "Allocation target summaries must include at least one positive "
        f"target/lineage value for {allocation_target!r}."
    )

    if allocation_target == "uniform":
        return

    mode_token = allocation_target.split("_", maxsplit=1)[0]
    utility_payloads = _numeric_payloads_by_name(
        summaries,
        lambda name: (
            "utility" in name
            or "priority" in name
            or "improvement" in name
            or mode_token in name
        ),
    )
    _assert_numeric_payloads_are_finite(
        utility_payloads,
        "Evidence/posterior allocation summaries must include finite "
        f"mode-specific utility payloads for {allocation_target!r}.",
    )


def _normalise_dispatch_status(status) -> str:
    status_name = str(status).lower()
    if status_name in {"accept", "accepted", "complete", "completed"}:
        return "accepted"
    if status_name in {"retry", "retried"}:
        return "retried"
    if status_name in {"revoke", "revoked", "cancel", "cancelled"}:
        return "revoked"
    return status_name


def _normalise_dispatch_record(
        record,
        *,
        fallback_sequence: int | None = None,
) -> DispatchRecord:
    dispatch_sequence = _record_field(
        record,
        "dispatch_sequence",
        "coordinator_sequence",
        "sequence",
        default=fallback_sequence,
    )
    if dispatch_sequence is None:
        pytest.fail(
            "Dispatch ledger records must expose coordinator dispatch order."
        )
    identity_owner = str(
        _record_field(
            record,
            "identity_owner",
            "issued_by",
            "source",
            "owner",
        )
    ).lower()
    if identity_owner != "coordinator":
        pytest.fail(
            "Dispatch ledger identity must be issued by the coordinator, "
            f"got {identity_owner!r}."
        )

    return DispatchRecord(
        dispatch_sequence=_scalar_int(
            dispatch_sequence,
            "dispatch_record.dispatch_sequence",
        ),
        runner_id=str(
            _record_field(record, "runner_id", "coordinator_runner_id")
        ),
        task_id=str(_record_field(record, "task_id", "coordinator_task_id")),
        attempt_id=str(
            _record_field(record, "attempt_id", "coordinator_attempt_id")
        ),
        transport_id=str(
            _record_field(
                record,
                "transport_id",
                "delivery_id",
                "coordinator_transport_id",
            )
        ),
        requested_parent_index=_scalar_int(
            _record_field(
                record,
                "requested_parent_index",
                "requested_parent_idx",
            ),
            "dispatch_record.requested_parent_index",
        ),
        in_flight_parent_target=_scalar_int(
            _record_field(
                record,
                "in_flight_parent_target",
                "parent_target",
                "target_parent_index",
            ),
            "dispatch_record.in_flight_parent_target",
        ),
        effective_parent_index=_scalar_int(
            _record_field(
                record,
                "effective_parent_index",
                "effective_parent_idx",
            ),
            "dispatch_record.effective_parent_index",
        ),
        accepted_parent_index=_scalar_int(
            _record_field(
                record,
                "accepted_parent_index",
                "accepted_parent_idx",
            ),
            "dispatch_record.accepted_parent_index",
        ),
        status=_normalise_dispatch_status(
            _record_field(record, "status", "task_status")
        ),
        identity_owner=identity_owner,
    )


def _coerce_dispatch_records(ledger) -> tuple[object, ...]:
    if callable(ledger):
        ledger = ledger()
    if ledger is None:
        return ()
    if isinstance(ledger, dict):
        for field_name in ("records", "events", "entries", "items"):
            if field_name in ledger:
                return tuple(ledger[field_name])
    for field_name in ("records", "events", "entries", "items"):
        if hasattr(ledger, field_name):
            return tuple(getattr(ledger, field_name))
    return tuple(ledger)


def _load_balancer_with_dispatch_trace(LoadBalancerClient, trace_probe):
    signature = inspect.signature(LoadBalancerClient)
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    for field_name in PUBLIC_DISPATCH_TRACE_CONSTRUCTOR_NAMES:
        if field_name not in signature.parameters and not has_var_kwargs:
            continue
        try:
            return LoadBalancerClient(
                address="local",
                **{field_name: trace_probe},
            )
        except TypeError:
            continue

    lb = LoadBalancerClient(address="local")
    for setter_name in PUBLIC_DISPATCH_TRACE_SETTER_NAMES:
        setter = getattr(lb, setter_name, None)
        if callable(setter):
            setter(trace_probe)
            return lb
    for attribute_name in PUBLIC_DISPATCH_TRACE_ATTRIBUTE_NAMES:
        if hasattr(lb, attribute_name):
            try:
                setattr(lb, attribute_name, trace_probe)
            except AttributeError:
                continue
            return lb
    return lb


def _public_coordinator_dispatch_records(
        lb,
        trace_probe,
) -> tuple[object, ...]:
    del lb
    if trace_probe.records:
        return tuple(trace_probe.records)
    pytest.fail(
        "Load-balanced execution must append coordinator-owned dispatch "
        "events to the injected public dispatch trace/probe during execution; "
        "records read back from load-balancer attributes after the fact are "
        "not an independent source of truth."
    )


def _worker_runtime_dispatch_records(worker_runtime) -> tuple[object, ...]:
    for field_name in PUBLIC_WORKER_RUNTIME_DISPATCH_RECORD_NAMES:
        if hasattr(worker_runtime, field_name):
            records = _coerce_dispatch_records(
                getattr(worker_runtime, field_name)
            )
            if records:
                return records
    pytest.fail(
        "WorkerRuntimeDiagnostics must expose public dispatch ledger/trace "
        "records, not only post-hoc task summary arrays."
    )


def _normalise_ordered_dispatch_records(
        records,
) -> tuple[DispatchRecord, ...]:
    normalised = tuple(
        _normalise_dispatch_record(record, fallback_sequence=sequence)
        for sequence, record in enumerate(records)
    )
    return tuple(
        sorted(normalised, key=lambda record: record.dispatch_sequence)
    )


def _assert_worker_runtime_matches_dispatch_records(
        worker_runtime,
        records: tuple[DispatchRecord, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    assert records
    dispatch_sequences = tuple(record.dispatch_sequence for record in records)
    assert dispatch_sequences == tuple(sorted(dispatch_sequences))
    assert len(set(dispatch_sequences)) == len(dispatch_sequences)
    assert {
        record.identity_owner for record in records
    } == {"coordinator"}

    task_ids = tuple(
        str(task_id) for task_id in _as_tuple(worker_runtime.task_ids)
    )
    runner_ids = tuple(
        str(runner_id) for runner_id in _as_tuple(worker_runtime.runner_ids)
    )
    requested_parent_indices = tuple(
        int(parent_idx)
        for parent_idx in _as_tuple(worker_runtime.requested_parent_indices)
    )
    effective_parent_indices = tuple(
        int(parent_idx)
        for parent_idx in _as_tuple(worker_runtime.effective_parent_indices)
    )
    accepted_parent_indices = tuple(
        int(parent_idx)
        for parent_idx in _as_tuple(worker_runtime.accepted_parent_indices)
    )
    in_flight_parent_targets = tuple(
        int(parent_target)
        for parent_target in _as_tuple(worker_runtime.in_flight_parent_targets)
    )

    assert task_ids == tuple(record.task_id for record in records)
    assert runner_ids == tuple(record.runner_id for record in records)
    assert requested_parent_indices == tuple(
        record.requested_parent_index for record in records
    )
    assert effective_parent_indices == tuple(
        record.effective_parent_index for record in records
    )
    assert accepted_parent_indices == tuple(
        record.accepted_parent_index for record in records
    )
    assert in_flight_parent_targets == tuple(
        record.in_flight_parent_target for record in records
    )

    status_counts = {
        status: sum(record.status == status for record in records)
        for status in ("accepted", "retried", "revoked")
    }
    assert int(worker_runtime.accepted_task_count) == status_counts["accepted"]
    assert int(worker_runtime.retried_task_count) == status_counts["retried"]
    assert int(worker_runtime.revoked_task_count) == status_counts["revoked"]
    assert all(task_id not in ("", "None") for task_id in task_ids)
    assert all(runner_id not in ("", "None") for runner_id in runner_ids)

    runner_task_ids = tuple(zip(runner_ids, task_ids, strict=True))
    dispatch_identities = tuple(
        (
            record.runner_id,
            record.task_id,
            record.attempt_id,
            record.transport_id,
        )
        for record in records
    )
    assert all(
        all(identity_part not in ("", "None") for identity_part in identity)
        for identity in dispatch_identities
    )

    return task_ids, runner_ids, runner_task_ids


def _assert_statistical_results_match(
        left: NestedSamplerResults,
        right: NestedSamplerResults,
) -> None:
    for field_name in STATISTICAL_RESULT_SNAPSHOT_FIELD_NAMES:
        np.testing.assert_allclose(
            np.asarray(getattr(left, field_name)),
            np.asarray(getattr(right, field_name)),
            err_msg=field_name,
        )


def _assert_diagnostics_do_not_expose_statistical_estimates(diagnostics):
    for field_name in STATISTICAL_RESULT_FIELD_NAMES:
        assert not hasattr(diagnostics, field_name), field_name
        for section_name in (
                "allocation",
                "parent_selection",
                "depth",
                "goal",
                "sampler",
                "worker_runtime",
        ):
            assert not hasattr(
                getattr(diagnostics, section_name),
                field_name,
            ), (section_name, field_name)


def _make_base_results() -> NestedSamplerResults:
    log_l = jnp.asarray([0.0, 1.0])
    return NestedSamplerResults(
        log_Z_mean=jnp.asarray(0.0),
        log_Z_uncert=jnp.asarray(0.1),
        ess=jnp.asarray(2.0),
        H_mean=jnp.asarray(0.0),
        total_num_samples=jnp.asarray(2),
        total_phantom_samples=jnp.asarray(0),
        total_num_likelihood_evaluations=jnp.asarray(4),
        log_efficiency=jnp.log(jnp.asarray(0.5)),
        termination_reason=jnp.asarray(0),
        U_samples=log_l,
        X_samples=log_l,
        log_L_constraints=jnp.asarray([-jnp.inf, 0.0]),
        log_L_phantom=jnp.full((2, 0), -jnp.inf),
        valid_phantom=jnp.asarray([False, False]),
        log_L=log_l,
        log_L_blocks=log_l,
        log_dp=jnp.asarray([-0.7, -0.7]),
        log_X_mean=jnp.asarray([0.0, -0.5]),
        log_posterior_density=log_l,
        num_live_points_per_sample=jnp.asarray([2, 1]),
        num_likelihood_evaluations_per_sample=jnp.asarray([1, 3]),
        log_L_supremum=log_l[-1],
        U_supremum=log_l[-1],
        X_supremum=log_l[-1],
        log_L_map=log_l[-1],
        U_map=log_l[-1],
        X_map=log_l[-1],
    )


def _make_result_diagnostics(diagnostics_types, *, num_samples: int):
    num_dispatches = max(0, num_samples - 2)
    classic_likelihood_evaluations = [1, 3]
    classic_likelihood_evaluations.extend([1] * num_dispatches)
    classic_likelihood_evaluations = classic_likelihood_evaluations[
        :num_samples
    ]
    total_likelihood_evaluations = sum(classic_likelihood_evaluations)

    return diagnostics_types.ExecutionDiagnostics(
        allocation=diagnostics_types.AllocationDiagnostics(
            mode="uniform",
            target_num_live_points=2,
            shell_size=1,
            target_summaries=(),
        ),
        parent_selection=diagnostics_types.ParentSelectionDiagnostics(
            requested_parent_indices=jnp.zeros(
                (num_dispatches,),
                dtype=jnp.int32,
            ),
            effective_parent_indices=jnp.zeros(
                (num_dispatches,),
                dtype=jnp.int32,
            ),
            accepted_parent_indices=jnp.zeros(
                (num_dispatches,),
                dtype=jnp.int32,
            ),
            sentinel_fallback_count=0,
            sentinel_fallback_indices=jnp.asarray([], dtype=jnp.int32),
        ),
        depth=diagnostics_types.DepthDiagnostics(
            condition_summaries=(
                diagnostics_types.DiagnosticConditionSummary(
                    condition_name="max_samples",
                    iteration=0,
                    num_samples=num_samples,
                    value=num_samples,
                    satisfied=True,
                ),
                diagnostics_types.DiagnosticConditionSummary(
                    condition_name="max_num_likelihood_evaluations",
                    iteration=0,
                    num_samples=num_samples,
                    value=total_likelihood_evaluations,
                    satisfied=False,
                ),
            ),
        ),
        goal=diagnostics_types.GoalDiagnostics(
            condition_summaries=(
                diagnostics_types.DiagnosticConditionSummary(
                    condition_name="goal_cond",
                    iteration=0,
                    num_samples=num_samples,
                    value=num_samples,
                    satisfied=True,
                ),
            ),
        ),
        sampler=diagnostics_types.SamplerDiagnostics(
            mode="slice",
            direction_kernel_mode="isotropic",
            trajectory_mode="straight_line_perfect_bracketing",
            phantom_burn_in=1,
            retained_phantom_capacity=0,
            retained_phantom_counts_per_sample=jnp.zeros(
                (num_samples,),
                dtype=jnp.int32,
            ),
            likelihood_evaluations_per_classic_sample=jnp.asarray(
                classic_likelihood_evaluations,
                dtype=jnp.int32,
            ),
            likelihood_evaluations_per_retained_phantom_cluster=(
                jnp.zeros((num_samples, 0), dtype=jnp.int32)
            ),
        ),
        worker_runtime=diagnostics_types.WorkerRuntimeDiagnostics(
            worker_count=0,
            compute_sector_summaries=(),
            runner_ids=(),
            task_ids=(),
            requested_parent_indices=jnp.asarray([], dtype=jnp.int32),
            effective_parent_indices=jnp.asarray([], dtype=jnp.int32),
            accepted_parent_indices=jnp.asarray([], dtype=jnp.int32),
            in_flight_parent_targets=(),
            accepted_task_count=0,
            retried_task_count=0,
            revoked_task_count=0,
            model_compilation_times=(),
            async_identity_preserved=True,
        ),
    )


def _make_result_aligned_diagnostics(diagnostics_types):
    return _make_result_diagnostics(diagnostics_types, num_samples=2)


def _make_result_misaligned_diagnostics(diagnostics_types):
    return _make_result_diagnostics(diagnostics_types, num_samples=3)


def test_public_execution_diagnostics_schema_types_are_reachable():
    diagnostics_types = _public_diagnostics_types()

    for type_name in PUBLIC_DIAGNOSTIC_TYPE_NAMES:
        public_type = getattr(diagnostics_types, type_name)
        _assert_public_diagnostics_symbol(public_type, type_name)


@pytest.mark.parametrize(
    "case",
    (
        AllocationDiagnosticsCase(
            allocation_target="uniform",
            collect_phantoms=False,
        ),
        AllocationDiagnosticsCase(
            allocation_target="evidence_improving",
            collect_phantoms=False,
        ),
        AllocationDiagnosticsCase(
            allocation_target="posterior_improving",
            collect_phantoms=True,
        ),
    ),
    ids=lambda case: case.allocation_target,
)
def test_allocation_target_summaries_record_mode_specific_content(
        case: AllocationDiagnosticsCase,
):
    ns = _make_nested_sampler(collect_phantoms=case.collect_phantoms)
    _, results, _ = _run_result_with_observations(
        ns,
        allocation_target=case.allocation_target,
    )
    diagnostics = _require_diagnostics(results)

    _assert_allocation_target_summaries_are_meaningful(
        diagnostics,
        case.allocation_target,
    )


def test_local_no_phantom_result_has_schema_and_explicit_empty_values():
    ns = _make_nested_sampler(collect_phantoms=False)
    state, results, observations = _run_result_with_observations(
        ns,
        allocation_target="uniform",
    )
    diagnostics = _require_diagnostics(results)

    assert int(state.num_samples) == int(results.total_num_samples)
    assert observations
    assert [observation.iteration for observation in observations] == list(
        range(len(observations))
    )
    assert observations[-1] == GoalObservation(
        iteration=len(observations) - 1,
        state_num_samples=int(state.num_samples),
        satisfied=True,
    )
    assert diagnostics.allocation.mode == "uniform"
    assert int(diagnostics.allocation.target_num_live_points) == 2
    assert int(diagnostics.allocation.shell_size) == 1
    _assert_allocation_target_summaries_are_meaningful(
        diagnostics,
        "uniform",
    )

    assert diagnostics.sampler.mode == "slice"
    assert diagnostics.sampler.direction_kernel_mode == "isotropic"
    assert (
        diagnostics.sampler.trajectory_mode
        == "straight_line_perfect_bracketing"
    )
    assert int(diagnostics.sampler.phantom_burn_in) == 1

    num_samples = int(results.total_num_samples)
    np.testing.assert_array_equal(
        np.asarray(diagnostics.sampler.retained_phantom_counts_per_sample),
        np.zeros((num_samples,), dtype=np.int64),
    )
    assert (
        np.asarray(
            diagnostics.sampler
            .likelihood_evaluations_per_retained_phantom_cluster
        ).shape
        == (num_samples, 0)
    )
    assert results.log_L_phantom.shape == (num_samples, 0)
    np.testing.assert_array_equal(
        np.asarray(results.valid_phantom),
        np.zeros((num_samples,), dtype=bool),
    )

    _assert_empty_worker_runtime(diagnostics.worker_runtime)


def test_phantom_run_records_sampler_and_run_allocation_fields():
    ns = _make_nested_sampler(collect_phantoms=True, phantom_burn_in=1)
    _, results, _ = _run_result_with_observations(
        ns,
        allocation_target="posterior_improving",
    )
    diagnostics = _require_diagnostics(results)

    assert diagnostics.allocation.mode == "posterior_improving"
    _assert_allocation_target_summaries_are_meaningful(
        diagnostics,
        "posterior_improving",
    )
    assert int(diagnostics.sampler.phantom_burn_in) == 1
    assert int(diagnostics.sampler.retained_phantom_capacity) == 1

    retained_counts = np.asarray(
        diagnostics.sampler.retained_phantom_counts_per_sample
    )
    np.testing.assert_array_equal(
        retained_counts,
        np.asarray(results.valid_phantom, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(
            diagnostics.sampler.likelihood_evaluations_per_classic_sample
        ),
        np.asarray(results.num_likelihood_evaluations_per_sample),
    )
    assert (
        np.asarray(
            diagnostics.sampler
            .likelihood_evaluations_per_retained_phantom_cluster
        ).shape
        == results.log_L_phantom.shape
    )

    expected_dispatches = int(results.total_num_samples) - 2
    requested = np.asarray(
        diagnostics.parent_selection.requested_parent_indices
    )
    effective = np.asarray(
        diagnostics.parent_selection.effective_parent_indices
    )
    accepted = np.asarray(
        diagnostics.parent_selection.accepted_parent_indices
    )
    assert requested.shape == (expected_dispatches,)
    assert effective.shape == requested.shape
    assert accepted.shape == requested.shape
    assert int(diagnostics.parent_selection.sentinel_fallback_count) >= 0
    _assert_empty_worker_runtime(diagnostics.worker_runtime)


def test_depth_and_goal_summaries_match_values_used_by_execution(monkeypatch):
    ns = _make_nested_sampler(collect_phantoms=False)
    depth_cond = TerminationCondition(
        max_samples=4,
        max_num_likelihood_evaluations=10_000,
    )
    depth_observations: list[DepthConditionObservation] = []
    _record_depth_condition_observations(monkeypatch, depth_observations)

    state, results, observations = _run_result_with_observations(
        ns,
        allocation_target="evidence_improving",
        depth_max_samples=4,
        depth_cond=depth_cond,
    )
    jax.block_until_ready(state.num_samples)
    diagnostics = _require_diagnostics(results)

    _assert_allocation_target_summaries_are_meaningful(
        diagnostics,
        "evidence_improving",
    )
    assert observations
    assert int(state.num_samples) == observations[-1].state_num_samples

    goal_summaries = tuple(diagnostics.goal.condition_summaries)
    assert len(goal_summaries) == len(observations)
    for summary, observation in zip(
            goal_summaries,
            observations,
            strict=True,
    ):
        assert summary.condition_name == "goal_cond"
        assert int(summary.iteration) == observation.iteration
        assert int(summary.num_samples) == observation.state_num_samples
        assert int(summary.value) == observation.state_num_samples
        assert bool(summary.satisfied) == observation.satisfied

    assert depth_observations
    assert (
        len({observation.iteration for observation in depth_observations}) > 1
    )
    assert any(
        not observation.satisfied
        for observation in depth_observations
        if observation.condition_name == "max_samples"
    )
    assert any(
        observation.satisfied
        for observation in depth_observations
        if observation.condition_name == "max_samples"
    )

    observed_condition_names = {
        observation.condition_name
        for observation in depth_observations
    }
    depth_summaries = tuple(
        _normalise_depth_summary(summary)
        for summary in diagnostics.depth.condition_summaries
        if summary.condition_name in observed_condition_names
    )
    assert depth_summaries == tuple(depth_observations)


def test_worker_runtime_assertions_allow_runner_scoped_task_ids():
    records = (
        DispatchRecord(
            dispatch_sequence=0,
            runner_id="runner-a",
            task_id="task-0",
            attempt_id="attempt-0",
            transport_id="transport-0",
            requested_parent_index=0,
            in_flight_parent_target=4,
            effective_parent_index=2,
            accepted_parent_index=2,
            status="accepted",
            identity_owner="coordinator",
        ),
        DispatchRecord(
            dispatch_sequence=1,
            runner_id="runner-b",
            task_id="task-0",
            attempt_id="attempt-1",
            transport_id="transport-1",
            requested_parent_index=1,
            in_flight_parent_target=5,
            effective_parent_index=3,
            accepted_parent_index=4,
            status="retried",
            identity_owner="coordinator",
        ),
    )
    worker_runtime = types.SimpleNamespace(
        runner_ids=("runner-a", "runner-b"),
        task_ids=("task-0", "task-0"),
        requested_parent_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        effective_parent_indices=jnp.asarray([2, 3], dtype=jnp.int32),
        accepted_parent_indices=jnp.asarray([2, 4], dtype=jnp.int32),
        in_flight_parent_targets=(4, 5),
        accepted_task_count=1,
        retried_task_count=1,
        revoked_task_count=0,
    )

    task_ids, runner_ids, runner_task_ids = (
        _assert_worker_runtime_matches_dispatch_records(
            worker_runtime,
            records,
        )
    )

    assert len(set(task_ids)) == 1
    assert len(set(runner_ids)) == 2
    assert len(set(runner_task_ids)) == 2


def test_load_balanced_worker_runtime_identity_fields_are_coordinator_owned():
    runtime = importlib.import_module("jaxns.runtime")
    LoadBalancerClient = runtime.LoadBalancerClient
    trace_probe = DispatchTraceProbe()

    with _load_balancer_with_dispatch_trace(
            LoadBalancerClient,
            trace_probe,
    ) as lb:
        lb.add_workers(["cpu:*:2"])
        results_by_runner = []
        for _ in range(2):
            runner_model = ToyModel(centre=0.25)
            runner = lb.get_nested_sampler(
                model=runner_model,
                collect_phantoms=True,
                sampler=UniDimSliceSampler(
                    model=runner_model,
                    num_slices=3,
                    no_step_out=True,
                    collect_phantom_samples=True,
                    phantom_burn_in=1,
                ),
                target_num_live_points=2,
                max_samples=8,
                shell_size=1,
                termination_condition=TerminationCondition(max_samples=8),
                batch_size=None,
            )
            _, results, _ = _run_result_with_observations(
                runner,
                allocation_target="uniform",
            )
            results_by_runner.append(results)
        coordinator_records = _normalise_ordered_dispatch_records(
            _public_coordinator_dispatch_records(lb, trace_probe)
        )

    # The same model and PRNG seed make post-hoc statistical reconstruction
    # collide; async identities must still be distinct per coordinator runner.
    _assert_statistical_results_match(
        results_by_runner[0],
        results_by_runner[1],
    )
    diagnostics_by_runner = [
        _require_diagnostics(results)
        for results in results_by_runner
    ]
    worker_runtime = diagnostics_by_runner[0].worker_runtime
    assert int(worker_runtime.worker_count) == 2
    assert len(worker_runtime.compute_sector_summaries) == 1

    sector = worker_runtime.compute_sector_summaries[0]
    assert sector.device_type == "cpu"
    assert int(sector.worker_count) == 2

    coordinator_records_by_runner: dict[str, list[DispatchRecord]] = {}
    for record in coordinator_records:
        coordinator_records_by_runner.setdefault(record.runner_id, []).append(
            record
        )

    runner_ids_by_runner = []
    runner_task_ids_by_runner = []
    for diagnostics, results in zip(
            diagnostics_by_runner,
            results_by_runner,
            strict=True,
    ):
        diagnostic_runner_ids = tuple(
            str(runner_id)
            for runner_id in _as_tuple(diagnostics.worker_runtime.runner_ids)
        )
        assert len(set(diagnostic_runner_ids)) == 1
        runner_id = diagnostic_runner_ids[0]
        assert runner_id in coordinator_records_by_runner

        observed_records = tuple(coordinator_records_by_runner[runner_id])
        _task_ids, runner_ids, runner_task_ids = (
            _assert_worker_runtime_matches_dispatch_records(
                diagnostics.worker_runtime,
                observed_records,
            )
        )
        runner_ids_by_runner.append(set(runner_ids))
        runner_task_ids_by_runner.append(set(runner_task_ids))
        assert len(set(runner_ids)) == 1

        diagnostic_records = _normalise_ordered_dispatch_records(
            _worker_runtime_dispatch_records(diagnostics.worker_runtime)
        )
        assert tuple(diagnostic_records) == tuple(
            observed_records
        )

        assert int(diagnostics.worker_runtime.accepted_task_count) >= (
            int(results.total_num_samples)
            - int(diagnostics.allocation.target_num_live_points)
        )
        compilation_times = tuple(
            diagnostics.worker_runtime.model_compilation_times
        )
        if compilation_times:
            compilation_times_array = np.asarray(
                compilation_times,
                dtype=np.float64,
            )
            assert np.all(np.isfinite(compilation_times_array))
            assert np.all(compilation_times_array > 0.0)
        assert bool(diagnostics.worker_runtime.async_identity_preserved)

    assert runner_task_ids_by_runner[0].isdisjoint(
        runner_task_ids_by_runner[1]
    )
    assert runner_ids_by_runner[0].isdisjoint(runner_ids_by_runner[1])


def test_diagnostics_boundary_keeps_runtime_audit_separate_from_statistics():
    diagnostics_types = _public_diagnostics_types()
    results = _make_base_results()
    diagnostics = _make_result_aligned_diagnostics(diagnostics_types)
    statistical_snapshot = {
        field_name: np.asarray(getattr(results, field_name)).copy()
        for field_name in STATISTICAL_RESULT_FIELD_NAMES
    }

    results_with_diagnostics = _apply_public_diagnostics_boundary(
        results,
        diagnostics,
    )
    attached_diagnostics = _require_diagnostics(results_with_diagnostics)

    for field_name, expected_value in statistical_snapshot.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(results_with_diagnostics, field_name)),
            expected_value,
            err_msg=field_name,
        )
    _assert_diagnostics_do_not_expose_statistical_estimates(
        attached_diagnostics
    )


def test_malformed_diagnostics_fail_at_public_result_boundary():
    diagnostics_types = _public_diagnostics_types()
    results = _make_base_results()
    diagnostics = _make_result_misaligned_diagnostics(diagnostics_types)

    with pytest.raises(ValueError, match=r"(?i)diagnostic|shape|align"):
        _apply_public_diagnostics_boundary(results, diagnostics)
