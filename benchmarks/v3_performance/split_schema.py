"""Schema checks for Ticket 0020 split v3 performance benchmark records."""

from __future__ import annotations

import datetime as _datetime
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from benchmarks.v3_performance.feature_manifest import row_by_id


SCHEMA_VERSION = "v3_performance_split_v1"
METRIC_FAMILY = "v3_performance_split"
EXECUTION_MODES = ("pure_core", "distributed")
REQUIRED_TIMING_FIELDS = (
    "setup_seconds",
    "compile_seconds",
    "run_seconds",
    "result_conversion_seconds",
    "mc_shrinkage_seconds",
    "total_seconds",
)
REQUIRED_RESULT_FIELDS = (
    "likelihood_evaluations",
    "total_samples",
    "log_Z_mean",
    "log_Z_uncert",
    "logZ_ref",
    "accuracy_passed",
)
REQUIRED_METADATA_FIELDS = (
    "feature_row_id",
    "problem_fixture",
    "dimension",
    "allocation_target",
    "seed",
    "target_num_live_points",
    "max_samples",
    "shell_size",
    "num_slices",
    "phantom_burn_in",
    "mc_sample_count",
    "depth_condition",
    "goal_condition",
    "phantom_enabled",
    "c_min",
    "direction_kernel",
    "trajectory_mode",
    "resume_pattern",
)
REQUIRED_PURE_CORE_DIAGNOSTICS = (
    "jax_cache_hit_count",
    "jax_cache_miss_count",
    "jit_compile_count",
    "static_shape_signature",
    "static_shape_cache_key",
    "rejected_dynamic_shape_count",
)
REQUIRED_DISTRIBUTED_DIAGNOSTICS = (
    "requested_worker_spec",
    "observed_node_count",
    "live_node_ingress_process_count",
    "live_node_coordinator_count",
    "observed_worker_process_count",
    "worker_utilization",
    "queue_diagnostics",
    "compile_cache_diagnostics",
)


def assert_split_benchmark_record(record: Mapping[str, Any]) -> None:
    """Assert that one split benchmark record follows the Ticket 0020 schema."""
    _require_mapping(record, "record")
    for key in (
            "schema_version",
            "metric_family",
            "execution_mode",
            "metadata",
            "timings",
            "diagnostics",
            "results",
    ):
        if key not in record:
            raise AssertionError(f"record missing required field {key!r}.")
    if record["schema_version"] != SCHEMA_VERSION:
        raise AssertionError("record.schema_version is not supported.")
    if record["metric_family"] != METRIC_FAMILY:
        raise AssertionError("record.metric_family is not supported.")
    execution_mode = str(record["execution_mode"])
    if execution_mode not in EXECUTION_MODES:
        raise AssertionError("execution_mode must be pure_core or distributed.")

    metadata = _require_mapping(record["metadata"], "metadata")
    timings = _require_mapping(record["timings"], "timings")
    diagnostics = _require_mapping(record["diagnostics"], "diagnostics")
    results = _require_mapping(record["results"], "results")
    _assert_required_keys(metadata, REQUIRED_METADATA_FIELDS, "metadata")
    _assert_required_keys(timings, REQUIRED_TIMING_FIELDS, "timings")
    _assert_required_keys(results, REQUIRED_RESULT_FIELDS, "results")

    feature_row = row_by_id(str(metadata["feature_row_id"]))
    if metadata["problem_fixture"] != feature_row.problem_fixture:
        raise AssertionError("metadata.problem_fixture must match manifest row.")
    if int(metadata["dimension"]) != feature_row.dimension:
        raise AssertionError("metadata.dimension must match manifest row.")
    if metadata["allocation_target"] != feature_row.allocation_target:
        raise AssertionError("metadata.allocation_target must match manifest row.")
    for field_name in (
            "depth_condition",
            "goal_condition",
            "direction_kernel",
            "trajectory_mode",
            "resume_pattern",
    ):
        if metadata[field_name] != getattr(feature_row, field_name):
            raise AssertionError(f"metadata.{field_name} must match manifest row.")
    if bool(metadata["phantom_enabled"]) != feature_row.phantom_enabled:
        raise AssertionError("metadata.phantom_enabled must match manifest row.")
    if metadata["c_min"] != feature_row.c_min:
        raise AssertionError("metadata.c_min must match manifest row.")
    if not isinstance(metadata["seed"], Integral):
        raise AssertionError("metadata.seed must be an integer.")
    if int(metadata["seed"]) not in feature_row.seeds:
        raise AssertionError("metadata.seed must come from the manifest row.")
    for field_name in (
            "target_num_live_points",
            "max_samples",
            "shell_size",
            "num_slices",
            "mc_sample_count",
    ):
        value = metadata[field_name]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(f"metadata.{field_name} must be positive.")
    phantom_burn_in = metadata["phantom_burn_in"]
    if (
            not isinstance(phantom_burn_in, Integral)
            or int(phantom_burn_in) < 0
    ):
        raise AssertionError("metadata.phantom_burn_in must be non-negative.")

    _assert_timing_fields(timings)
    _assert_result_fields(results)
    if execution_mode == "pure_core":
        _assert_pure_core_diagnostics(diagnostics)
    else:
        _assert_distributed_diagnostics(diagnostics)


def assert_split_benchmark_coverage(
        pure_core_feature_row_ids: Sequence[str],
        distributed_feature_row_ids: Sequence[str],
) -> None:
    """Assert pure-core and distributed suites cover identical feature rows."""
    pure_ids = _row_id_set(pure_core_feature_row_ids, "pure-core")
    distributed_ids = _row_id_set(distributed_feature_row_ids, "distributed")
    if pure_ids != distributed_ids:
        raise AssertionError(
            "pure-core and distributed benchmark coverage must match. "
            f"pure_only={sorted(pure_ids - distributed_ids)}, "
            f"distributed_only={sorted(distributed_ids - pure_ids)}."
        )


def assert_split_benchmark_records(
        records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate a mixed pure-core/distributed split benchmark record set."""
    if not records:
        raise AssertionError("split benchmark records must be non-empty.")
    pure_ids = []
    distributed_ids = []
    for record in records:
        assert_split_benchmark_record(record)
        feature_row_id = str(record["metadata"]["feature_row_id"])
        if record["execution_mode"] == "pure_core":
            pure_ids.append(feature_row_id)
        else:
            distributed_ids.append(feature_row_id)
    assert_split_benchmark_coverage(pure_ids, distributed_ids)


def assert_split_benchmark_markdown_report(
        *,
        path: Path | str,
        text: str,
        records: Sequence[Mapping[str, Any]],
) -> None:
    """Assert dated Markdown report expectations for split benchmark output."""
    report_path = Path(path)
    match = re.fullmatch(
        r"v3_performance_split_(\d{4}-\d{2}-\d{2})\.md",
        report_path.name,
    )
    if match is None:
        raise AssertionError(
            "split benchmark report path must be dated as "
            "v3_performance_split_YYYY-MM-DD.md."
        )
    report_date = _datetime.date.fromisoformat(match.group(1))
    if f"Date: {report_date.isoformat()}" not in text:
        raise AssertionError("split benchmark report must include its date.")
    required_snippets = (
        "# V3 Performance Split Benchmark Report",
        "## Pure-Core Results",
        "## Distributed Results",
        "setup s",
        "compile s",
        "run s",
        "result s",
        "MC s",
        "likelihood evals",
        "## Diagnostics",
        "JAX cache/static-shape",
        "worker topology",
    )
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        raise AssertionError(
            "split benchmark report missing required text: "
            f"{', '.join(missing)}."
        )
    for record in records:
        assert_split_benchmark_record(record)
        feature_row_id = str(record["metadata"]["feature_row_id"])
        execution_mode = str(record["execution_mode"])
        if feature_row_id not in text:
            raise AssertionError(
                f"split benchmark report missing feature row {feature_row_id!r}."
            )
        if execution_mode not in text:
            raise AssertionError(
                f"split benchmark report missing mode {execution_mode!r}."
            )


def _assert_timing_fields(timings: Mapping[str, Any]) -> None:
    measured_total = 0.0
    for field_name in REQUIRED_TIMING_FIELDS:
        value = _finite_non_negative(timings[field_name], field_name)
        if field_name != "total_seconds":
            measured_total += value
    if not math.isclose(
            float(timings["total_seconds"]),
            measured_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
    ):
        raise AssertionError(
            "total_seconds must equal setup + compile + run + result + MC."
        )


def _assert_result_fields(results: Mapping[str, Any]) -> None:
    for field_name in ("likelihood_evaluations", "total_samples"):
        value = results[field_name]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(f"results.{field_name} must be positive.")
    for field_name in ("log_Z_mean", "log_Z_uncert", "logZ_ref"):
        _finite(results[field_name], f"results.{field_name}")
    if not isinstance(results["accuracy_passed"], bool):
        raise AssertionError("results.accuracy_passed must be boolean.")
    if not results["accuracy_passed"]:
        raise AssertionError(
            "results.accuracy_passed must be True before benchmark timing "
            "numbers are accepted."
        )


def _assert_pure_core_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    _assert_required_keys(
        diagnostics,
        REQUIRED_PURE_CORE_DIAGNOSTICS,
        "pure-core diagnostics",
    )
    for field_name in (
            "jax_cache_hit_count",
            "jax_cache_miss_count",
            "jit_compile_count",
            "rejected_dynamic_shape_count",
    ):
        value = diagnostics[field_name]
        if not isinstance(value, Integral) or int(value) < 0:
            raise AssertionError(
                f"pure-core diagnostics.{field_name} must be non-negative."
            )
    for field_name in ("static_shape_signature", "static_shape_cache_key"):
        value = diagnostics[field_name]
        if not isinstance(value, str) or not value:
            raise AssertionError(
                f"pure-core diagnostics.{field_name} must be a non-empty string."
            )


def _assert_distributed_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    _assert_required_keys(
        diagnostics,
        REQUIRED_DISTRIBUTED_DIAGNOSTICS,
        "distributed diagnostics",
    )
    requested_worker_spec = diagnostics["requested_worker_spec"]
    if isinstance(requested_worker_spec, (str, bytes)) or not isinstance(
            requested_worker_spec,
            Sequence,
    ):
        raise AssertionError(
            "distributed diagnostics.requested_worker_spec must be a sequence."
        )
    if not requested_worker_spec or not all(
            isinstance(item, str) and item
            for item in requested_worker_spec
    ):
        raise AssertionError(
            "distributed diagnostics.requested_worker_spec entries must be "
            "non-empty strings."
        )
    for field_name in (
            "observed_node_count",
            "live_node_ingress_process_count",
            "live_node_coordinator_count",
            "observed_worker_process_count",
    ):
        value = diagnostics[field_name]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(
                f"distributed diagnostics.{field_name} must be positive."
            )
    if int(diagnostics["live_node_ingress_process_count"]) != int(
            diagnostics["observed_node_count"]
    ):
        raise AssertionError(
            "distributed diagnostics must report one live ingress per node."
        )
    if int(diagnostics["live_node_coordinator_count"]) != int(
            diagnostics["observed_node_count"]
    ):
        raise AssertionError(
            "distributed diagnostics must report one live coordinator per node."
        )
    _assert_fraction(
        diagnostics["worker_utilization"],
        "distributed diagnostics.worker_utilization",
    )
    for field_name in ("queue_diagnostics", "compile_cache_diagnostics"):
        _require_mapping(diagnostics[field_name], field_name)


def _row_id_set(row_ids: Sequence[str], suite_name: str) -> set[str]:
    if isinstance(row_ids, (str, bytes)):
        raise AssertionError(f"{suite_name} row ids must be a sequence.")
    normalized = {str(row_id) for row_id in row_ids}
    if not normalized:
        raise AssertionError(f"{suite_name} row ids must be non-empty.")
    for row_id in normalized:
        row_by_id(row_id)
    return normalized


def _assert_required_keys(
        mapping: Mapping[str, Any],
        required_keys: Sequence[str],
        label: str,
) -> None:
    missing = [field_name for field_name in required_keys
               if field_name not in mapping]
    if missing:
        raise AssertionError(
            f"{label} missing required field(s): {', '.join(missing)}."
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be a mapping.")
    return value


def _finite_non_negative(value: Any, name: str) -> float:
    numeric = _finite(value, name)
    if numeric < 0.0:
        raise AssertionError(f"{name} must be non-negative.")
    return numeric


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, Real):
        raise AssertionError(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AssertionError(f"{name} must be finite.")
    return numeric


def _assert_fraction(value: Any, name: str) -> None:
    numeric = _finite(value, name)
    if numeric < 0.0 or numeric > 1.0:
        raise AssertionError(f"{name} must be between 0 and 1.")
