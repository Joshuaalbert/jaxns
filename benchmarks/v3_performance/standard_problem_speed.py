"""Speed benchmark for the v3 8D ``basic_mvn`` local runtime."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any

# Support ``python -m benchmarks...`` from a source checkout without requiring
# an editable install first.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.model import Model
from jaxns.runtime import LoadBalancerClient
from jaxns.termination_condition import TerminationCondition
from jaxctx.priors.prior import Prior


SCHEMA_VERSION = "v3_standard_problem_speed_v1"
METRIC_FAMILY = "standard_problem_speed"
PROBLEM_NAME = "basic_mvn"
PROBLEM_DIMENSION = 8
DEFAULT_ALLOCATION_TARGETS = (
    "uniform",
    "evidence_improving",
    "posterior_improving",
)
REQUIRED_TIMING_FIELDS = (
    "setup_seconds",
    "run_seconds",
    "result_conversion_seconds",
    "mc_shrinkage_seconds",
    "total_seconds",
)
REQUIRED_FRACTION_FIELDS = (
    "setup_fraction",
    "run_fraction",
    "result_conversion_fraction",
    "mc_shrinkage_fraction",
)
REQUIRED_METADATA_FIELDS = (
    "problem",
    "dimension",
    "allocation_target",
    "seed",
    "worker_specs",
    "target_num_live_points",
    "live_points_per_dimension",
    "live_point_policy",
    "max_samples",
    "shell_size",
    "num_slices",
    "phantom_burn_in",
    "direction_kernel",
    "mc_sample_count",
    "software_version",
)
REQUIRED_RESULT_FIELDS = (
    "likelihood_evaluations",
    "total_samples",
    "log_Z_mean",
    "log_Z_uncert",
    "logZ_ref",
)
REQUIRED_DIAGNOSTIC_FIELDS = (
    "actual_worker_count",
    "worker_sampler_latency_seconds",
    "observed_worker_device_classes",
    "dispatch_eval_count",
    "dispatch_latency_seconds_total",
    "dispatch_latency_seconds_mean",
    "dispatch_throughput_per_second",
    "compile_count",
    "cache_hit_count",
    "rejected_shape_cache_count",
    "distinct_compile_identity_count",
    "max_active_evals_per_worker",
    "max_active_evals_pool",
    "completed_eval_count_by_worker",
    "queued_eval_count",
    "failed_eval_count",
)
LIKELIHOOD_DISPATCH_LATENCY_TAIL_FIELDS = (
    "dispatch_latency_seconds_max",
    "dispatch_latency_seconds_p95",
)

tfpd = tfp.distributions
perf_counter = time.perf_counter
_MISSING = object()


@dataclasses.dataclass(frozen=True, slots=True)
class StandardProblemSpeedConfig:
    """Configuration for the public v3 standard-problem speed benchmark."""

    problem: str = PROBLEM_NAME
    dimension: int = PROBLEM_DIMENSION
    allocation_targets: tuple[str, ...] = DEFAULT_ALLOCATION_TARGETS
    seed: int = 0
    worker_specs: tuple[str, ...] = ("cpu:*:2",)
    target_num_live_points: int = 30
    live_points_per_dimension: float | None = None
    live_point_policy: str | None = None
    max_samples: int = 1200
    shell_size: int = 15
    num_slices: int = 24
    phantom_burn_in: int = 4
    direction_kernel: str = "ellipsoidal"
    mc_sample_count: int = 1000


def default_standard_problem_speed_config(
        *,
        allocation_targets: Sequence[str] = DEFAULT_ALLOCATION_TARGETS,
        seed: int = 0,
        worker_specs: Sequence[str] = ("cpu:*:2",),
        target_num_live_points: int | None = None,
        live_points_per_dimension: float | None = None,
        max_samples: int = 1200,
        shell_size: int = 15,
        num_slices: int = 24,
        phantom_burn_in: int = 4,
        direction_kernel: str = "ellipsoidal",
        mc_sample_count: int = 1000,
) -> StandardProblemSpeedConfig:
    """Build an explicit benchmark configuration."""
    if target_num_live_points is None:
        if live_points_per_dimension is None:
            target_num_live_points = 30
        else:
            target_num_live_points = int(
                round(float(live_points_per_dimension) * PROBLEM_DIMENSION)
            )
    if live_points_per_dimension is None:
        live_points_per_dimension = (
            float(target_num_live_points) / float(PROBLEM_DIMENSION)
        )

    if math.isclose(
            float(live_points_per_dimension),
            50.0,
            rel_tol=0.0,
            abs_tol=1e-12,
    ):
        live_point_policy = "50_per_dimension"
    elif int(target_num_live_points) == 30:
        live_point_policy = "accepted_standard_gate"
    else:
        live_point_policy = "custom"

    config = StandardProblemSpeedConfig(
        allocation_targets=tuple(allocation_targets),
        seed=int(seed),
        worker_specs=_normalize_worker_specs(worker_specs),
        target_num_live_points=int(target_num_live_points),
        live_points_per_dimension=float(live_points_per_dimension),
        live_point_policy=live_point_policy,
        max_samples=int(max_samples),
        shell_size=int(shell_size),
        num_slices=int(num_slices),
        phantom_burn_in=int(phantom_burn_in),
        direction_kernel=str(direction_kernel),
        mc_sample_count=int(mc_sample_count),
    )
    _validate_config(config)
    return config


def collect_standard_problem_speed_records(
        config: StandardProblemSpeedConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the configured local-LB speed benchmarks and return JSON records."""
    config = (
        default_standard_problem_speed_config()
        if config is None
        else _coerce_config(config)
    )

    records: list[dict[str, Any]] = []
    for allocation_target in config.allocation_targets:
        setup_started = perf_counter()
        model, log_z_ref = build_basic_mvn_problem()
        sampler = UniDimSliceSampler(
            model=model,
            num_slices=config.num_slices,
            phantom_burn_in=config.phantom_burn_in,
            collect_phantom_samples=True,
            direction_kernel=config.direction_kernel,
        )

        with LoadBalancerClient(address="local") as lb:
            lb.add_workers(list(config.worker_specs))
            actual_worker_count = sum(
                int(getattr(sector, "num_workers", 0))
                for sector in getattr(lb, "compute_sectors", ())
            )
            runner = lb.get_nested_sampler(
                model=model,
                collect_phantoms=True,
                sampler=sampler,
                target_num_live_points=config.target_num_live_points,
                max_samples=config.max_samples,
                shell_size=config.shell_size,
            )
            setup_seconds = perf_counter() - setup_started

            run_started = perf_counter()
            state = runner.run_until_goal(
                goal_cond=lambda state: False,
                depth_cond=TerminationCondition(max_samples=runner.max_samples),
                allocation_target=allocation_target,
                key=jax.random.PRNGKey(config.seed),
            )
            run_seconds = perf_counter() - run_started

            result_started = perf_counter()
            result = state.to_result().trim()
            result_conversion_seconds = perf_counter() - result_started

            diagnostics = (
                result.get_diagnostics()
                if hasattr(result, "get_diagnostics")
                else getattr(result, "execution_diagnostics", None)
            )
            worker_runtime = getattr(diagnostics, "worker_runtime", None)
            dispatch_records = getattr(worker_runtime, "dispatch_records", ())
            worker_sampler_latency_seconds = 0.0
            for dispatch_record in dispatch_records:
                if isinstance(dispatch_record, Mapping):
                    value = dispatch_record.get(
                        "sampler_execution_latency_seconds",
                        0.0,
                    )
                else:
                    value = getattr(
                        dispatch_record,
                        "sampler_execution_latency_seconds",
                        0.0,
                    )
                if value is not None and math.isfinite(float(value)):
                    worker_sampler_latency_seconds += max(float(value), 0.0)

            if hasattr(lb, "likelihood_dispatch_diagnostics"):
                dispatch_diagnostics = lb.likelihood_dispatch_diagnostics()
            elif hasattr(lb, "get_likelihood_dispatch_diagnostics"):
                dispatch_diagnostics = lb.get_likelihood_dispatch_diagnostics()
            else:
                raise AssertionError(
                    "load balancer must expose likelihood dispatch diagnostics."
                )

            dispatch_latencies = [
                _finite_non_negative(value, "dispatch_latency_seconds")
                for value in _read_diagnostic(
                    dispatch_diagnostics,
                    "dispatch_latency_seconds",
                    default=(),
                )
            ]
            dispatch_eval_count = int(
                _read_diagnostic(
                    dispatch_diagnostics,
                    "dispatch_eval_count",
                    default=len(dispatch_latencies),
                )
            )
            total_dispatch_latency = float(sum(dispatch_latencies))
            if dispatch_latencies and dispatch_eval_count > 0:
                mean_dispatch_latency = (
                    total_dispatch_latency / float(dispatch_eval_count)
                )
                max_dispatch_latency = float(max(dispatch_latencies))
                p95_dispatch_latency = float(
                    np.percentile(np.asarray(dispatch_latencies), 95.0)
                )
            else:
                mean_dispatch_latency = 0.0
                max_dispatch_latency = 0.0
                p95_dispatch_latency = 0.0

            mc_started = perf_counter()
            mc_shrinkage_samples = result.sample_mc_shrinkage(
                num_samples=config.mc_sample_count
            )
            _block_until_ready(mc_shrinkage_samples)
            for attr_name in (
                    "log_Z_samples",
                    "log_L_blocks",
                    "log_L_samples",
                    "sample_indices",
            ):
                if hasattr(mc_shrinkage_samples, attr_name):
                    _block_until_ready(getattr(mc_shrinkage_samples, attr_name))
            mc_shrinkage_seconds = perf_counter() - mc_started

        timings = {
            "setup_seconds": float(setup_seconds),
            "run_seconds": float(run_seconds),
            "result_conversion_seconds": float(result_conversion_seconds),
            "mc_shrinkage_seconds": float(mc_shrinkage_seconds),
        }
        timings["total_seconds"] = float(sum(timings.values()))

        record = {
            "schema_version": SCHEMA_VERSION,
            "metric_family": METRIC_FAMILY,
            "metadata": {
                "problem": config.problem,
                "dimension": int(config.dimension),
                "allocation_target": allocation_target,
                "seed": int(config.seed),
                "worker_specs": list(config.worker_specs),
                "target_num_live_points": int(config.target_num_live_points),
                "live_points_per_dimension": float(
                    config.live_points_per_dimension
                ),
                "live_point_policy": str(config.live_point_policy),
                "max_samples": int(config.max_samples),
                "shell_size": int(config.shell_size),
                "num_slices": int(config.num_slices),
                "phantom_burn_in": int(config.phantom_burn_in),
                "direction_kernel": config.direction_kernel,
                "mc_sample_count": int(config.mc_sample_count),
                "software_version": _software_version(),
            },
            "timings": timings,
            "timing_fractions": compute_timing_fractions(timings),
            "diagnostics": {
                "actual_worker_count": int(actual_worker_count),
                "worker_sampler_latency_seconds": float(
                    worker_sampler_latency_seconds
                ),
                "observed_worker_device_classes": list(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "observed_worker_device_classes",
                        default=(),
                    )
                ),
                "dispatch_eval_count": dispatch_eval_count,
                "dispatch_latency_seconds_total": total_dispatch_latency,
                "dispatch_latency_seconds_mean": float(mean_dispatch_latency),
                "dispatch_latency_seconds_max": max_dispatch_latency,
                "dispatch_latency_seconds_p95": p95_dispatch_latency,
                "dispatch_throughput_per_second": float(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "dispatch_throughput_per_second",
                        default=0.0,
                    )
                ),
                "compile_count": int(
                    _read_diagnostic(dispatch_diagnostics, "compile_count")
                ),
                "cache_hit_count": int(
                    _read_diagnostic(dispatch_diagnostics, "cache_hit_count")
                ),
                "rejected_shape_cache_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "rejected_shape_cache_count",
                    )
                ),
                "distinct_compile_identity_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "distinct_compile_identity_count",
                    )
                ),
                "max_active_evals_per_worker": _json_safe_mapping(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "max_active_evals_per_worker",
                    ),
                    "max_active_evals_per_worker",
                ),
                "max_active_evals_pool": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "max_active_evals_pool",
                    )
                ),
                "completed_eval_count_by_worker": _json_safe_mapping(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "completed_eval_count_by_worker",
                    ),
                    "completed_eval_count_by_worker",
                ),
                "queued_eval_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "queued_eval_count",
                        default=0,
                    )
                ),
                "failed_eval_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "failed_eval_count",
                        default=0,
                    )
                ),
            },
            "results": {
                "likelihood_evaluations": int(
                    getattr(result, "total_num_likelihood_evaluations")
                ),
                "total_samples": int(getattr(result, "total_num_samples")),
                "log_Z_mean": float(getattr(result, "log_Z_mean")),
                "log_Z_uncert": float(getattr(result, "log_Z_uncert")),
                "logZ_ref": float(log_z_ref),
            },
        }
        assert_standard_problem_speed_record(record)
        records.append(json.loads(json.dumps(record, sort_keys=True)))
    return records


def collect_worker_scaling_speed_records(
        config: StandardProblemSpeedConfig | Mapping[str, Any] | None = None,
        worker_specs_grid: Sequence[Sequence[str] | str] | None = None,
) -> list[dict[str, Any]]:
    """Run the same benchmark for each requested local worker specification."""
    base_config = (
        default_standard_problem_speed_config()
        if config is None
        else _coerce_config(config)
    )
    if worker_specs_grid is None:
        worker_specs_grid = (base_config.worker_specs,)

    records: list[dict[str, Any]] = []
    for worker_specs in worker_specs_grid:
        if isinstance(worker_specs, str):
            normalized_specs = (worker_specs,)
        else:
            normalized_specs = _normalize_worker_specs(worker_specs)
        records.extend(
            collect_standard_problem_speed_records(
                dataclasses.replace(
                    base_config,
                    worker_specs=normalized_specs,
                )
            )
        )
    return records


def compute_timing_fractions(timings: Mapping[str, Any]) -> dict[str, float]:
    """Return timing fractions derived from measured seconds."""
    measured_names = tuple(
        name for name in REQUIRED_TIMING_FIELDS
        if name != "total_seconds"
    )
    measured_total = sum(
        _finite_non_negative(timings[name], name)
        for name in measured_names
    )
    if measured_total <= 0.0:
        raise ValueError("timing total must be positive.")
    return {
        name.removesuffix("_seconds") + "_fraction": (
            _finite_non_negative(timings[name], name) / measured_total
        )
        for name in measured_names
    }


def assert_standard_problem_speed_record(record: Mapping[str, Any]) -> None:
    """Assert that a speed record matches the public benchmark schema."""
    _require_mapping(record, "record")
    for key in (
            "schema_version",
            "metric_family",
            "metadata",
            "timings",
            "timing_fractions",
            "diagnostics",
            "results",
    ):
        if key not in record:
            raise AssertionError(f"record missing required field {key!r}.")
    if record["schema_version"] != SCHEMA_VERSION:
        raise AssertionError("record.schema_version is not supported.")
    if record["metric_family"] != METRIC_FAMILY:
        raise AssertionError("record.metric_family is not supported.")

    metadata_map = _require_mapping(record["metadata"], "metadata")
    timings_map = _require_mapping(record["timings"], "timings")
    fractions_map = _require_mapping(
        record["timing_fractions"],
        "timing_fractions",
    )
    diagnostics_map = _require_mapping(record["diagnostics"], "diagnostics")
    results_map = _require_mapping(record["results"], "results")

    _assert_required_keys(metadata_map, REQUIRED_METADATA_FIELDS, "metadata")
    _assert_required_keys(timings_map, REQUIRED_TIMING_FIELDS, "timing")
    _assert_required_keys(
        fractions_map,
        REQUIRED_FRACTION_FIELDS,
        "timing_fractions",
    )
    _assert_required_keys(
        diagnostics_map,
        REQUIRED_DIAGNOSTIC_FIELDS,
        "diagnostics",
    )
    _assert_required_keys(results_map, REQUIRED_RESULT_FIELDS, "results")

    for name in REQUIRED_TIMING_FIELDS:
        _finite_non_negative(timings_map[name], name)
    timing_total = sum(
        float(timings_map[name])
        for name in REQUIRED_TIMING_FIELDS
        if name != "total_seconds"
    )
    if not math.isclose(
            float(timings_map["total_seconds"]),
            timing_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
    ):
        raise AssertionError("total_seconds must equal measured timing sum.")

    expected_fractions = compute_timing_fractions(timings_map)
    for name in REQUIRED_FRACTION_FIELDS:
        actual = _finite_non_negative(fractions_map[name], name)
        expected = expected_fractions[name]
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(
                f"{name} fraction must be derived from measured timing."
            )

    if metadata_map["problem"] != PROBLEM_NAME:
        raise AssertionError("metadata.problem must be basic_mvn.")
    if int(metadata_map["dimension"]) != PROBLEM_DIMENSION:
        raise AssertionError("metadata.dimension must be 8.")
    if metadata_map["allocation_target"] not in DEFAULT_ALLOCATION_TARGETS:
        raise AssertionError("metadata.allocation_target is not supported.")
    for name in (
            "seed",
            "target_num_live_points",
            "max_samples",
            "shell_size",
            "num_slices",
            "phantom_burn_in",
            "mc_sample_count",
    ):
        minimum = 0 if name in {"seed", "phantom_burn_in"} else 1
        value = metadata_map[name]
        if not isinstance(value, Integral) or int(value) < minimum:
            raise AssertionError(f"{name} must be >= {minimum}.")

    live_points_per_dimension = float(metadata_map["live_points_per_dimension"])
    expected_lppd = (
        float(metadata_map["target_num_live_points"]) / PROBLEM_DIMENSION
    )
    if (
            not math.isfinite(live_points_per_dimension)
            or live_points_per_dimension <= 0.0
            or not math.isclose(
                live_points_per_dimension,
                expected_lppd,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
    ):
        raise AssertionError(
            "live_points_per_dimension must match target live points."
        )
    if metadata_map["live_point_policy"] not in {
        "accepted_standard_gate",
        "50_per_dimension",
        "custom",
    }:
        raise AssertionError("metadata.live_point_policy is not supported.")
    _assert_worker_specs(metadata_map["worker_specs"])
    if not str(metadata_map["direction_kernel"]):
        raise AssertionError("metadata.direction_kernel must be non-empty.")
    if not str(metadata_map["software_version"]):
        raise AssertionError("metadata.software_version must be non-empty.")

    for name in ("log_Z_mean", "log_Z_uncert", "logZ_ref"):
        value = float(results_map[name])
        if not math.isfinite(value):
            raise AssertionError(f"results.{name} must be finite.")
    for name in ("likelihood_evaluations", "total_samples"):
        value = results_map[name]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(f"results.{name} must be positive.")

    _finite_non_negative(
        diagnostics_map["worker_sampler_latency_seconds"],
        "worker_sampler_latency_seconds",
    )
    actual_worker_count = diagnostics_map["actual_worker_count"]
    if not isinstance(actual_worker_count, Integral) or int(actual_worker_count) <= 0:
        raise AssertionError("diagnostics.actual_worker_count must be positive.")

    device_classes = diagnostics_map["observed_worker_device_classes"]
    if isinstance(device_classes, (str, bytes)) or not isinstance(
            device_classes,
            Sequence,
    ):
        raise AssertionError("observed_worker_device_classes must be a sequence.")
    if (
            not device_classes
            or not all(isinstance(item, str) and item for item in device_classes)
    ):
        raise AssertionError(
            "observed_worker_device_classes entries must be non-empty."
        )

    for name in (
            "dispatch_eval_count",
            "compile_count",
            "cache_hit_count",
            "rejected_shape_cache_count",
            "distinct_compile_identity_count",
            "max_active_evals_pool",
            "queued_eval_count",
            "failed_eval_count",
    ):
        value = diagnostics_map[name]
        if not isinstance(value, Integral) or int(value) < 0:
            raise AssertionError(f"{name} must be a non-negative integer.")
    for name in (
            "dispatch_latency_seconds_total",
            "dispatch_latency_seconds_mean",
            "dispatch_throughput_per_second",
    ):
        _finite_non_negative(diagnostics_map[name], name)
    if not any(
            name in diagnostics_map
            for name in LIKELIHOOD_DISPATCH_LATENCY_TAIL_FIELDS
    ):
        raise AssertionError("dispatch_latency_seconds tail summary is required.")
    for name in LIKELIHOOD_DISPATCH_LATENCY_TAIL_FIELDS:
        if name in diagnostics_map:
            _finite_non_negative(diagnostics_map[name], name)

    dispatch_eval_count = int(diagnostics_map["dispatch_eval_count"])
    total_latency = float(diagnostics_map["dispatch_latency_seconds_total"])
    mean_latency = float(diagnostics_map["dispatch_latency_seconds_mean"])
    if dispatch_eval_count == 0:
        if total_latency != 0.0 or mean_latency != 0.0:
            raise AssertionError(
                "zero dispatch_eval_count requires zero dispatch latency summary."
            )
    elif total_latency > 0.0:
        expected_mean_latency = total_latency / float(dispatch_eval_count)
        if not math.isclose(
                mean_latency,
                expected_mean_latency,
                rel_tol=1e-9,
                abs_tol=1e-12,
        ):
            raise AssertionError(
                "dispatch_latency_seconds_mean must equal total latency "
                "divided by dispatch_eval_count."
            )
        expected_throughput = dispatch_eval_count / total_latency
        actual_throughput = float(
            diagnostics_map["dispatch_throughput_per_second"]
        )
        if not math.isclose(
                actual_throughput,
                expected_throughput,
                rel_tol=1e-9,
                abs_tol=1e-12,
        ):
            raise AssertionError(
                "dispatch_throughput_per_second must be derived from "
                "dispatch_eval_count and dispatch latency total."
            )
    if "dispatch_latency_seconds_max" in diagnostics_map:
        max_latency = float(diagnostics_map["dispatch_latency_seconds_max"])
        if max_latency > total_latency and dispatch_eval_count > 0:
            raise AssertionError(
                "dispatch_latency_seconds_max must not exceed total latency."
            )
    if (
            "dispatch_latency_seconds_p95" in diagnostics_map
            and "dispatch_latency_seconds_max" in diagnostics_map
    ):
        p95_latency = float(diagnostics_map["dispatch_latency_seconds_p95"])
        max_latency = float(diagnostics_map["dispatch_latency_seconds_max"])
        if p95_latency > max_latency:
            raise AssertionError(
                "dispatch_latency_seconds_p95 must not exceed max latency."
            )
    if dispatch_eval_count > 0:
        latency_evidence = any(
            float(diagnostics_map.get(name, 0.0)) > 0.0
            for name in (
                "dispatch_latency_seconds_total",
                "dispatch_latency_seconds_mean",
                "dispatch_latency_seconds_max",
                "dispatch_latency_seconds_p95",
            )
        )
        completed_evidence = any(
            int(value) > 0
            for value in _require_mapping(
                diagnostics_map["completed_eval_count_by_worker"],
                "completed_eval_count_by_worker",
            ).values()
        )
        if not latency_evidence and not completed_evidence:
            raise AssertionError(
                "dispatch_eval_count is positive but diagnostics contain no "
                "dispatch latency or completed-eval evidence."
            )

    max_by_worker = _require_mapping(
        diagnostics_map["max_active_evals_per_worker"],
        "max_active_evals_per_worker",
    )
    if not max_by_worker:
        raise AssertionError("max_active_evals_per_worker must be non-empty.")
    for worker_id, active_count in max_by_worker.items():
        if not str(worker_id):
            raise AssertionError(
                "max_active_evals_per_worker keys must be non-empty."
            )
        if (
                not isinstance(active_count, Integral)
                or int(active_count) < 0
                or int(active_count) > 1
        ):
            raise AssertionError(
                "max_active_evals_per_worker values must be active <= 1 "
                "per worker."
            )

    completed_by_worker = _require_mapping(
        diagnostics_map["completed_eval_count_by_worker"],
        "completed_eval_count_by_worker",
    )
    if not completed_by_worker:
        raise AssertionError("completed_eval_count_by_worker must be non-empty.")
    for worker_id, completed_count in completed_by_worker.items():
        if not str(worker_id):
            raise AssertionError(
                "completed_eval_count_by_worker keys must be non-empty."
            )
        if not isinstance(completed_count, Integral):
            raise AssertionError(
                "completed_eval_count_by_worker values must be integers."
            )
        if int(completed_count) < 0:
            raise AssertionError(
                "completed_eval_count_by_worker values must be non-negative."
            )
    completed_count = sum(int(value) for value in completed_by_worker.values())
    failed_count = int(diagnostics_map["failed_eval_count"])
    if completed_count > dispatch_eval_count:
        raise AssertionError(
            "completed_eval_count_by_worker total must not exceed "
            "dispatch_eval_count."
        )
    if completed_count + failed_count > dispatch_eval_count:
        raise AssertionError(
            "completed and failed eval counts must not exceed dispatch_eval_count."
        )
    if completed_count + failed_count != dispatch_eval_count:
        raise AssertionError(
            "dispatch_eval_count must equal completed plus failed eval counts."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that writes JSON records and a Markdown report."""
    parser = argparse.ArgumentParser(
        description="Collect v3 standard-problem speed benchmark records.",
    )
    parser.add_argument(
        "--allocation-target",
        action="append",
        choices=DEFAULT_ALLOCATION_TARGETS,
        help=(
            "Allocation target to run. May be supplied multiple times; "
            "defaults to all targets."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--worker-spec",
        action="append",
        dest="worker_specs",
        help=(
            "Public local-runtime worker spec. May be supplied multiple "
            "times; defaults to cpu:*:2."
        ),
    )
    parser.add_argument("--target-num-live-points", type=int, default=None)
    parser.add_argument("--live-points-per-dimension", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--shell-size", type=int, default=15)
    parser.add_argument("--num-slices", type=int, default=24)
    parser.add_argument("--phantom-burn-in", type=int, default=4)
    parser.add_argument("--direction-kernel", default="ellipsoidal")
    parser.add_argument("--mc-sample-count", type=int, default=1000)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the dated Markdown report and only print JSON records.",
    )
    parser.add_argument(
        "--worker-scaling",
        action="store_true",
        help=(
            "Treat repeated --worker-spec values as a worker-scaling grid. "
            "Each value is benchmarked as a separate one-spec local run."
        ),
    )
    args = parser.parse_args(argv)

    worker_specs = tuple(args.worker_specs or ("cpu:*:2",))
    config = default_standard_problem_speed_config(
        allocation_targets=tuple(
            args.allocation_target or DEFAULT_ALLOCATION_TARGETS
        ),
        seed=args.seed,
        worker_specs=worker_specs,
        target_num_live_points=args.target_num_live_points,
        live_points_per_dimension=args.live_points_per_dimension,
        max_samples=args.max_samples,
        shell_size=args.shell_size,
        num_slices=args.num_slices,
        phantom_burn_in=args.phantom_burn_in,
        direction_kernel=args.direction_kernel,
        mc_sample_count=args.mc_sample_count,
    )

    if args.worker_scaling:
        records = collect_worker_scaling_speed_records(
            config=config,
            worker_specs_grid=tuple((worker_spec,) for worker_spec in worker_specs),
        )
    else:
        records = collect_standard_problem_speed_records(config=config)

    print(json.dumps(records, sort_keys=True))
    if not args.no_report:
        report_dir = (
            args.report_dir
            if args.report_dir is not None
            else Path(__file__).resolve().parent / "reports"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            report_dir
            / f"standard_problem_speed_{date.today().isoformat()}.md"
        )
        lines = [
            "# V3 Standard Problem Speed Report",
            "",
            f"Date: {date.today().isoformat()}",
            "",
            "## Results",
            "",
            "| allocation | workers | total s | run s | MC s | evals | log Z | log Z uncert | max active |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for record in records:
            metadata_map = record["metadata"]
            timings_map = record["timings"]
            diagnostics_map = record["diagnostics"]
            results_map = record["results"]
            lines.append(
                "| "
                f"{metadata_map['allocation_target']} | "
                f"{', '.join(metadata_map['worker_specs'])} | "
                f"{timings_map['total_seconds']:.3f} | "
                f"{timings_map['run_seconds']:.3f} | "
                f"{timings_map['mc_shrinkage_seconds']:.3f} | "
                f"{results_map['likelihood_evaluations']} | "
                f"{results_map['log_Z_mean']:.6g} | "
                f"{results_map['log_Z_uncert']:.6g} | "
                f"{diagnostics_map['max_active_evals_pool']} |"
            )
        lines.extend([
            "",
            "## Notes",
            "",
            "- Timings are wall-clock seconds measured by this script.",
            "- Worker sampler latency is reported as a diagnostic and is not "
            "part of the wall-clock timing fractions.",
            "- Each allocation target owns and tears down its own local load "
            "balancer.",
            "",
        ])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote Markdown report to {report_path}", file=sys.stderr)
    return 0


def build_basic_mvn_problem() -> tuple[Model, float]:
    """Build the full 8D ``basic_mvn`` standard problem and reference logZ."""
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    log_z_ref = _log_normal(data_mu, prior_mu, prior_cov + data_cov)
    return Model(prior_model=_basic_mvn_prior_model), float(log_z_ref)


def _read_diagnostic(
        diagnostics: Any,
        field_name: str,
        *,
        default: Any = _MISSING,
) -> Any:
    """Read one diagnostic field from either a mapping or an object.

    Args:
        diagnostics: Diagnostics mapping or object.
        field_name: Field to read.
        default: Optional default when the field is absent.

    Returns:
        The field value.

    Raises:
        AssertionError: If the field is absent and no default is supplied.
    """
    if isinstance(diagnostics, Mapping) and field_name in diagnostics:
        return diagnostics[field_name]
    if hasattr(diagnostics, field_name):
        return getattr(diagnostics, field_name)
    if default is not _MISSING:
        return default
    raise AssertionError(
        f"likelihood dispatch diagnostics missing {field_name!r}."
    )


def _block_until_ready(value: Any) -> None:
    """Synchronize a JAX/array-like value before stopping a timer.

    Args:
        value: JAX array, NumPy-like object, or plain Python value.
    """
    block_until_ready = getattr(value, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
        return
    if hasattr(value, "tolist"):
        value.tolist()
        return
    try:
        np.asarray(value)
    except (TypeError, ValueError):
        return


def _json_safe_mapping(value: Any, field_name: str) -> dict[str, int]:
    """Return a JSON-safe worker-count mapping.

    Args:
        value: Mapping with arbitrary string-like keys and integer-like values.
        field_name: Name used in validation errors.

    Returns:
        Mapping with string keys and integer values.
    """
    mapping = _require_mapping(value, field_name)
    return {str(key): int(item) for key, item in mapping.items()}


def _coerce_config(
        config: StandardProblemSpeedConfig | Mapping[str, Any],
) -> StandardProblemSpeedConfig:
    """Convert a public config object or mapping into the benchmark dataclass.

    Args:
        config: Existing config dataclass or mapping of config fields.

    Returns:
        Validated benchmark configuration.
    """
    if isinstance(config, StandardProblemSpeedConfig):
        _validate_config(config)
        return config
    if dataclasses.is_dataclass(config):
        data = dataclasses.asdict(config)
    else:
        _require_mapping(config, "config")
        data = dict(config)
    return default_standard_problem_speed_config(
        allocation_targets=data.get("allocation_targets", DEFAULT_ALLOCATION_TARGETS),
        seed=data.get("seed", 0),
        worker_specs=data.get("worker_specs", ("cpu:*:2",)),
        target_num_live_points=data.get("target_num_live_points"),
        live_points_per_dimension=data.get("live_points_per_dimension"),
        max_samples=data.get("max_samples", 1200),
        shell_size=data.get("shell_size", 15),
        num_slices=data.get("num_slices", 24),
        phantom_burn_in=data.get("phantom_burn_in", 4),
        direction_kernel=data.get("direction_kernel", "ellipsoidal"),
        mc_sample_count=data.get("mc_sample_count", 1000),
    )


def _validate_config(config: StandardProblemSpeedConfig) -> None:
    """Validate a benchmark configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If a field cannot produce a supported benchmark run.
    """
    if config.problem != PROBLEM_NAME:
        raise ValueError(f"Only {PROBLEM_NAME!r} is supported.")
    if config.dimension != PROBLEM_DIMENSION:
        raise ValueError(f"Only {PROBLEM_DIMENSION} dimensions are supported.")
    if not config.allocation_targets:
        raise ValueError("At least one allocation target is required.")
    unknown_targets = set(config.allocation_targets) - set(DEFAULT_ALLOCATION_TARGETS)
    if unknown_targets:
        raise ValueError(f"Unknown allocation target(s): {sorted(unknown_targets)}")
    _normalize_worker_specs(config.worker_specs)
    for name in (
            "target_num_live_points",
            "max_samples",
            "shell_size",
            "num_slices",
            "mc_sample_count",
    ):
        value = getattr(config, name)
        if not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(config.seed, Integral) or int(config.seed) < 0:
        raise ValueError("seed must be a non-negative integer.")
    if (
            not isinstance(config.phantom_burn_in, Integral)
            or int(config.phantom_burn_in) < 0
    ):
        raise ValueError("phantom_burn_in must be a non-negative integer.")
    if int(config.phantom_burn_in) > int(config.num_slices) - 1:
        raise ValueError(
            "phantom_burn_in must satisfy 0 <= burn_in <= num_slices - 1."
        )


def _normalize_worker_specs(worker_specs: Sequence[str]) -> tuple[str, ...]:
    """Normalize worker specs to a non-empty tuple of strings.

    Args:
        worker_specs: Worker specification strings.

    Returns:
        Normalized worker spec tuple.

    Raises:
        ValueError: If the input is not a non-string sequence of strings.
    """
    if isinstance(worker_specs, (str, bytes)):
        raise ValueError("worker_specs must be a non-string sequence.")
    try:
        normalized = tuple(worker_specs)
    except TypeError as e:
        raise ValueError("worker_specs must be a sequence.") from e
    if not normalized:
        raise ValueError("worker_specs must be non-empty.")
    for worker_spec in normalized:
        if not isinstance(worker_spec, str) or not worker_spec:
            raise ValueError("worker_specs entries must be non-empty strings.")
    return normalized


def _assert_worker_specs(worker_specs: Any) -> None:
    """Assert that record metadata contains worker spec strings.

    Args:
        worker_specs: Metadata worker spec value.

    Raises:
        AssertionError: If the value is not a non-empty non-string sequence.
    """
    if isinstance(worker_specs, (str, bytes)):
        raise AssertionError(
            "metadata.worker_specs must be a non-string sequence."
        )
    if not isinstance(worker_specs, Sequence):
        raise AssertionError("metadata.worker_specs must be a sequence.")
    if not worker_specs:
        raise AssertionError("metadata.worker_specs must be non-empty.")
    for worker_spec in worker_specs:
        if not isinstance(worker_spec, str) or not worker_spec:
            raise AssertionError(
                "metadata.worker_specs entries must be non-empty strings."
            )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a mapping value.

    Args:
        value: Value to validate.
        name: Name used in validation errors.

    Returns:
        The original value typed as a mapping.

    Raises:
        AssertionError: If value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be a mapping.")
    return value


def _assert_required_keys(
        mapping: Mapping[str, Any],
        required_keys: Sequence[str],
        label: str,
) -> None:
    """Assert that a mapping contains all required schema keys.

    Args:
        mapping: Mapping to validate.
        required_keys: Keys that must be present.
        label: Schema section name used in validation errors.

    Raises:
        AssertionError: If any key is missing.
    """
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise AssertionError(
            f"{label} missing required field(s): {', '.join(missing)}."
        )


def _finite_non_negative(value: Any, name: str) -> float:
    """Return a finite non-negative float from a schema value.

    Args:
        value: Numeric value to validate.
        name: Name used in validation errors.

    Returns:
        Value converted to ``float``.

    Raises:
        AssertionError: If the value is not finite and non-negative.
    """
    if not isinstance(value, Real):
        raise AssertionError(f"{name} must be a real timing value.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AssertionError(f"{name} must be finite.")
    if numeric < 0.0:
        raise AssertionError(f"{name} must be non-negative (>= 0).")
    return numeric


def _software_version() -> str:
    """Return the installed package version, or ``unknown`` from source trees."""
    try:
        return metadata.version("jaxns")
    except metadata.PackageNotFoundError:
        return "unknown"


def _basic_mvn_parameters():
    """Return the fixed 8D Gaussian prior/data parameters.

    Returns:
        Tuple ``(prior_mu, prior_cov, data_mu, data_cov)`` with shapes
        ``[D]``, ``[D, D]``, ``[D]``, and ``[D, D]``.
    """
    # prior_mu, data_mu: [D]
    prior_mu = 6 * jnp.ones(PROBLEM_DIMENSION)
    data_mu = jnp.zeros(PROBLEM_DIMENSION)
    # prior_cov, data_cov: [D, D]
    prior_cov = np.eye(PROBLEM_DIMENSION)
    prior_cov[prior_cov == 0] = 0.99
    data_cov = jnp.eye(PROBLEM_DIMENSION)
    return prior_mu, prior_cov, data_mu, data_cov


def _basic_mvn_prior_model():
    """Evaluate the benchmark prior model likelihood.

    Returns:
        Scalar log likelihood for one sampled point.
    """
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    # x: [D]
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=prior_mu,
            scale_tril=jnp.linalg.cholesky(prior_cov),
        ),
        name="x",
    ).realise()
    return tfpd.MultivariateNormalTriL(
        loc=data_mu,
        scale_tril=jnp.linalg.cholesky(data_cov),
    ).log_prob(x)


def _log_normal(x, mean, cov) -> float:
    """Evaluate a multivariate normal log density.

    Args:
        x: Point with shape ``[D]``.
        mean: Mean vector with shape ``[D]``.
        cov: Covariance matrix with shape ``[D, D]``.

    Returns:
        Scalar log density.
    """
    # l_factor: [D, D]
    l_factor = jnp.linalg.cholesky(cov)
    # dx: [D]
    dx = x - mean
    dx = solve_triangular(l_factor, dx, lower=True)
    normalizer = -0.5 * x.size * jnp.log(2.0 * jnp.pi)
    return float(
        normalizer
        - jnp.sum(jnp.log(jnp.diag(l_factor)))
        - 0.5 * dx @ dx
    )


if __name__ == "__main__":
    raise SystemExit(main())
