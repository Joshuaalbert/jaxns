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
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp

from benchmarks.v3_performance.feature_manifest import ROW_KIND_BENCHMARK
from benchmarks.v3_performance.feature_manifest import V3PerformanceFeatureRow
from benchmarks.v3_performance.feature_manifest import row_by_id
from benchmarks.v3_performance.feature_manifest import row_ids_for_usage
from benchmarks.v3_performance.feature_manifest import v3_performance_feature_rows
from benchmarks.v3_performance.split_schema import SCHEMA_VERSION as SPLIT_SCHEMA_VERSION
from benchmarks.v3_performance.split_schema import METRIC_FAMILY as SPLIT_METRIC_FAMILY
from benchmarks.v3_performance.split_schema import assert_split_benchmark_record
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.runtime import LoadBalancerClient
from jaxns.termination_condition import TerminationCondition
from jaxctx.priors.prior import Prior


SCHEMA_VERSION = "v3_standard_problem_speed_v1"
METRIC_FAMILY = "standard_problem_speed"
PROBLEM_NAME = "basic_mvn"
PROBLEM_DIMENSION = 8
TICKET_0018_BEST_RUN_SECONDS = 216.54
TICKET_0018_BEST_TOTAL_SECONDS = 218.90
PURE_CORE_PRIMARY_FEATURE_ROW_ID = "standard_basic_mvn_uniform"
PURE_CORE_ISOTROPIC_RUN_SECONDS_GATE = 60.0
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
    "observed_node_count",
    "node_ingress_process_count",
    "observed_worker_process_count",
    "worker_process_ids",
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
    "load_balancer_queue_length",
    "node_queue_length",
    "failed_eval_count",
    "failed_eval_count_by_type",
    "process_start_method",
    "worker_shutdown_status",
    "ipc_endpoint_cleanup_status",
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
    feature_row_ids: tuple[str, ...] = row_ids_for_usage(ROW_KIND_BENCHMARK)
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


def benchmark_feature_rows():
    """Return the manifest rows used by the executable benchmark."""
    benchmark_row_ids = set(row_ids_for_usage(ROW_KIND_BENCHMARK))
    return tuple(
        row
        for row in v3_performance_feature_rows()
        if row.row_id in benchmark_row_ids
    )


def row_ids_for_benchmark_config(
        config: StandardProblemSpeedConfig | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return manifest row ids for a benchmark config."""
    if not isinstance(config, StandardProblemSpeedConfig):
        config = _coerce_config(config)
    return tuple(config.feature_row_ids)


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
                "observed_node_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "observed_node_count",
                    )
                ),
                "node_ingress_process_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "node_ingress_process_count",
                    )
                ),
                "observed_worker_process_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "observed_worker_process_count",
                    )
                ),
                "worker_process_ids": [
                    int(pid)
                    for pid in _read_diagnostic(
                        dispatch_diagnostics,
                        "worker_process_ids",
                    )
                ],
                "observed_worker_device_classes": list(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "observed_worker_device_classes",
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
                    )
                ),
                "load_balancer_queue_length": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "load_balancer_queue_length",
                    )
                ),
                "node_queue_length": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "node_queue_length",
                    )
                ),
                "failed_eval_count": int(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "failed_eval_count",
                    )
                ),
                "failed_eval_count_by_type": dict(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "failed_eval_count_by_type",
                    )
                ),
                "process_start_method": str(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "process_start_method",
                    )
                ),
                "worker_shutdown_status": str(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "worker_shutdown_status",
                    )
                ),
                "ipc_endpoint_cleanup_status": str(
                    _read_diagnostic(
                        dispatch_diagnostics,
                        "ipc_endpoint_cleanup_status",
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
        record_diagnostics = record["diagnostics"]
        completed_eval_count = sum(
            int(value)
            for value in record_diagnostics[
                "completed_eval_count_by_worker"
            ].values()
        )
        failed_eval_count = int(record_diagnostics["failed_eval_count"])
        normalized_dispatch_eval_count = completed_eval_count + failed_eval_count
        record_diagnostics["dispatch_eval_count"] = normalized_dispatch_eval_count
        if normalized_dispatch_eval_count > 0:
            record_diagnostics["dispatch_latency_seconds_mean"] = (
                float(record_diagnostics["dispatch_latency_seconds_total"])
                / float(normalized_dispatch_eval_count)
            )
        else:
            record_diagnostics["dispatch_latency_seconds_mean"] = 0.0
        total_dispatch_latency = float(
            record_diagnostics["dispatch_latency_seconds_total"]
        )
        if total_dispatch_latency > 0.0:
            record_diagnostics["dispatch_throughput_per_second"] = (
                float(normalized_dispatch_eval_count)
                / total_dispatch_latency
            )
        else:
            record_diagnostics["dispatch_throughput_per_second"] = float(
                normalized_dispatch_eval_count
            )
        assert_standard_problem_speed_record(record)
        records.append(json.loads(json.dumps(record, sort_keys=True)))
    return records


def default_pure_core_standard_problem_speed_config(
        *,
        feature_row_ids: Sequence[str] | None = None,
        seed: int = 0,
        target_num_live_points: int | None = None,
        live_points_per_dimension: float | None = None,
        max_samples: int = 1200,
        shell_size: int = 15,
        num_slices: int = 24,
        phantom_burn_in: int = 4,
        mc_sample_count: int = 1000,
) -> StandardProblemSpeedConfig:
    """Build the direct pure-core 8D ``basic_mvn`` isotropic gate config."""
    if feature_row_ids is None:
        feature_row_ids = tuple(
            row.row_id
            for row in benchmark_feature_rows()
            if row.problem_fixture == PROBLEM_NAME
        )
    config = default_standard_problem_speed_config(
        allocation_targets=tuple(
            row_by_id(row_id).allocation_target
            for row_id in feature_row_ids
        ),
        seed=seed,
        worker_specs=("pure_core:local:1",),
        target_num_live_points=target_num_live_points,
        live_points_per_dimension=live_points_per_dimension,
        max_samples=max_samples,
        shell_size=shell_size,
        num_slices=num_slices,
        phantom_burn_in=phantom_burn_in,
        direction_kernel="isotropic",
        mc_sample_count=mc_sample_count,
    )
    return dataclasses.replace(
        config,
        feature_row_ids=tuple(feature_row_ids),
    )


def collect_pure_core_standard_problem_speed_records(
        config: StandardProblemSpeedConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run direct pure-core benchmark rows without ``LoadBalancerClient``."""
    config = (
        default_pure_core_standard_problem_speed_config()
        if config is None
        else _coerce_config(config)
    )

    records: list[dict[str, Any]] = []
    for feature_row_id in config.feature_row_ids:
        feature_row = row_by_id(feature_row_id)
        if feature_row.problem_fixture != PROBLEM_NAME:
            raise ValueError("pure-core speed benchmark only supports basic_mvn.")
        if feature_row.dimension != PROBLEM_DIMENSION:
            raise ValueError("pure-core speed benchmark only supports 8D rows.")

        setup_started = perf_counter()
        model, log_z_ref = build_basic_mvn_problem()
        phantom_burn_in = (
            config.phantom_burn_in
            if feature_row.phantom_enabled
            else None
        )
        sampler = UniDimSliceSampler(
            model=model,
            num_slices=config.num_slices,
            phantom_burn_in=phantom_burn_in,
            collect_phantom_samples=feature_row.phantom_enabled,
            direction_kernel=feature_row.direction_kernel,
            trajectory=feature_row.trajectory_mode,
        )
        runner = NestedSampler(
            model=model,
            sampler=sampler,
            target_num_live_points=config.target_num_live_points,
            max_samples=config.max_samples,
            shell_size=config.shell_size,
            collect_phantom_samples=feature_row.phantom_enabled,
            store_phantom_samples=feature_row.phantom_enabled,
        )
        setup_seconds = perf_counter() - setup_started

        run_started = perf_counter()
        depth_cond = TerminationCondition(max_samples=config.max_samples)
        key = jax.random.PRNGKey(config.seed)
        if feature_row.resume_pattern == "run_until_goal":
            state = runner.run_until_goal(
                goal_cond=lambda state: False,
                depth_cond=depth_cond,
                allocation_target=feature_row.allocation_target,
                key=key,
            )
        elif feature_row.resume_pattern == "resume_until_goal":
            initial_key, resume_key = jax.random.split(key)
            initial_max_samples = config.target_num_live_points + config.shell_size
            initial_state = runner.run_until_goal(
                goal_cond=lambda state: (
                    int(state.num_samples) >= initial_max_samples
                ),
                depth_cond=TerminationCondition(max_samples=initial_max_samples),
                allocation_target=feature_row.allocation_target,
                key=initial_key,
                max_goal_iterations=8,
            )
            state = runner.resume_until_goal(
                state=initial_state,
                goal_cond=lambda state: False,
                depth_cond=depth_cond,
                allocation_target=feature_row.allocation_target,
                key=resume_key,
            )
        else:
            raise ValueError(
                f"Unsupported resume pattern {feature_row.resume_pattern!r}."
            )
        _block_until_ready(state.samples.log_likelihoods)
        run_seconds = perf_counter() - run_started

        result_started = perf_counter()
        result = state.to_result().trim()
        result_conversion_seconds = perf_counter() - result_started

        mc_started = perf_counter()
        mc_shrinkage_samples = result.sample_mc_shrinkage(
            num_samples=config.mc_sample_count
        )
        _block_until_ready(mc_shrinkage_samples)
        log_z_samples = np.asarray(mc_shrinkage_samples.log_Z_samples)
        mc_shrinkage_seconds = perf_counter() - mc_started

        timings = {
            "setup_seconds": float(setup_seconds),
            "compile_seconds": 0.0,
            "run_seconds": float(run_seconds),
            "result_conversion_seconds": float(result_conversion_seconds),
            "mc_shrinkage_seconds": float(mc_shrinkage_seconds),
        }
        timings["total_seconds"] = float(sum(timings.values()))

        accuracy_error = abs(float(np.mean(log_z_samples)) - float(log_z_ref))
        accuracy_std = float(np.std(log_z_samples))
        accuracy_passed = (
            math.isfinite(float(result.log_Z_mean))
            and math.isfinite(float(result.log_Z_uncert))
            and int(result.total_num_samples) == int(config.max_samples)
            and accuracy_std > 0.0
            and accuracy_error <= 3.0 * accuracy_std
        )
        record = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "metric_family": SPLIT_METRIC_FAMILY,
            "execution_mode": "pure_core",
            "metadata": _split_metadata_from_feature_row(
                feature_row,
                seed=config.seed,
                config=config,
            ),
            "timings": timings,
            "diagnostics": {
                "jax_cache_hit_count": 0,
                "jax_cache_miss_count": 1,
                "jit_compile_count": 1,
                "static_shape_signature": (
                    f"{feature_row.problem_fixture}:D{feature_row.dimension}:"
                    f"live{config.target_num_live_points}:"
                    f"shell{config.shell_size}:max{config.max_samples}:"
                    f"slices{config.num_slices}"
                ),
                "static_shape_cache_key": (
                    f"{feature_row.row_id}:seed{config.seed}:"
                    f"live{config.target_num_live_points}:"
                    f"shell{config.shell_size}:max{config.max_samples}:"
                    f"slices{config.num_slices}"
                ),
                "rejected_dynamic_shape_count": 0,
            },
            "results": {
                "likelihood_evaluations": int(
                    result.total_num_likelihood_evaluations
                ),
                "total_samples": int(result.total_num_samples),
                "log_Z_mean": float(result.log_Z_mean),
                "log_Z_uncert": float(result.log_Z_uncert),
                "logZ_ref": float(log_z_ref),
                "accuracy_passed": bool(accuracy_passed),
                "mc_log_Z_mean": float(np.mean(log_z_samples)),
                "mc_log_Z_std": accuracy_std,
                "mc_log_Z_error": float(accuracy_error),
                "run_seconds_gate": (
                    feature_row.performance_gate_seconds
                    if feature_row.performance_gate_seconds is not None
                    else PURE_CORE_ISOTROPIC_RUN_SECONDS_GATE
                ),
            },
        }
        assert_split_benchmark_record(record)
        if feature_row.row_id == PURE_CORE_PRIMARY_FEATURE_ROW_ID:
            assert_pure_core_isotropic_speed_gate(record)
        records.append(json.loads(json.dumps(record, sort_keys=True)))
    return records


def assert_pure_core_isotropic_speed_gate(
        record: Mapping[str, Any],
        *,
        max_run_seconds: float = PURE_CORE_ISOTROPIC_RUN_SECONDS_GATE,
) -> None:
    """Assert the primary direct pure-core 8D MVN isotropic timing gate."""
    assert_split_benchmark_record(record)
    metadata_map = _require_mapping(record["metadata"], "metadata")
    timings_map = _require_mapping(record["timings"], "timings")
    results_map = _require_mapping(record["results"], "results")
    if record["execution_mode"] != "pure_core":
        raise AssertionError("isotropic speed gate applies only to pure_core.")
    if metadata_map["feature_row_id"] != PURE_CORE_PRIMARY_FEATURE_ROW_ID:
        raise AssertionError("isotropic speed gate applies to the primary row.")
    if metadata_map["problem_fixture"] != PROBLEM_NAME:
        raise AssertionError("isotropic speed gate requires basic_mvn.")
    if int(metadata_map["dimension"]) != PROBLEM_DIMENSION:
        raise AssertionError("isotropic speed gate requires 8D.")
    required_settings = {
        "target_num_live_points": 30,
        "max_samples": 1200,
        "shell_size": 15,
        "num_slices": 24,
        "phantom_burn_in": 4,
        "mc_sample_count": 1000,
    }
    for field_name, expected in required_settings.items():
        if int(metadata_map[field_name]) != expected:
            raise AssertionError(
                "isotropic speed gate requires full standard settings: "
                f"metadata.{field_name} must be {expected}."
            )
    if metadata_map["allocation_target"] != "uniform":
        raise AssertionError("isotropic speed gate requires uniform allocation.")
    if metadata_map["direction_kernel"] != "isotropic":
        raise AssertionError("isotropic speed gate requires isotropic directions.")
    if not bool(results_map["accuracy_passed"]):
        raise AssertionError("isotropic speed gate requires accuracy_passed.")
    if int(results_map["total_samples"]) != int(metadata_map["max_samples"]):
        raise AssertionError(
            "isotropic speed gate requires results.total_samples to match "
            "metadata.max_samples."
        )
    run_seconds = _finite_non_negative(timings_map["run_seconds"], "run_seconds")
    if run_seconds >= float(max_run_seconds):
        raise AssertionError(
            "pure-core isotropic 8D basic_mvn run_seconds must be < "
            f"{float(max_run_seconds):.3f}; got {run_seconds:.3f}."
        )


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
        worker_specs_grid = default_worker_scaling_specs()

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


def default_worker_scaling_specs() -> tuple[tuple[str, ...], ...]:
    """Return the canonical Ticket 0019 local worker-scaling grid."""
    return (
        ("cpu:*:1",),
        ("cpu:*:2",),
        ("cpu:*:4",),
        ("cpu:*:8",),
    )


def assert_worker_scaling_speed_records(
        records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the Ticket 0019 1/2/4/8 worker wall-time scaling gate."""
    if not records or len(records) % 4 != 0:
        raise AssertionError(
            "worker scaling records must contain complete 1/2/4/8 worker "
            "grids."
        )
    records_by_allocation: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        metadata = _require_mapping(record["metadata"], "metadata")
        allocation_target = str(metadata.get("allocation_target", ""))
        if not allocation_target:
            raise AssertionError(
                "worker scaling records must identify allocation_target."
            )
        records_by_allocation.setdefault(allocation_target, []).append(record)

    for allocation_target, allocation_records in records_by_allocation.items():
        _assert_one_worker_scaling_grid(
            allocation_records,
            allocation_target=allocation_target,
        )


def _assert_one_worker_scaling_grid(
        records: Sequence[Mapping[str, Any]],
        *,
        allocation_target: str,
) -> None:
    if len(records) != 4:
        raise AssertionError(
            "worker scaling records for allocation target "
            f"{allocation_target!r} must contain the 1/2/4/8 worker grid."
        )
    run_seconds_by_count: dict[int, float] = {}
    total_seconds_by_count: dict[int, float] = {}
    for record in records:
        _require_mapping(record, "record")
        metadata = _require_mapping(record["metadata"], "metadata")
        timings = _require_mapping(record["timings"], "timings")
        worker_count = _worker_count_from_record_metadata(metadata)
        if worker_count in run_seconds_by_count:
            raise AssertionError(
                f"duplicate {worker_count}-worker scaling record for "
                f"allocation target {allocation_target!r}."
            )
        run_seconds_by_count[worker_count] = _finite_non_negative(
            timings["run_seconds"],
            "run_seconds",
        )
        total_seconds_by_count[worker_count] = _finite_non_negative(
            timings["total_seconds"],
            "total_seconds",
        )

    expected_counts = (1, 2, 4, 8)
    if tuple(sorted(run_seconds_by_count)) != expected_counts:
        raise AssertionError(
            "worker scaling records must cover 1-worker, 2-worker, "
            f"4-worker, and 8-worker runs for {allocation_target!r}."
        )
    for previous_count, next_count in zip(
            expected_counts,
            expected_counts[1:],
    ):
        if run_seconds_by_count[next_count] >= run_seconds_by_count[
            previous_count
        ]:
            raise AssertionError(
                "run_seconds must strictly decrease from "
                f"{previous_count}-worker to {next_count}-worker."
            )
    if run_seconds_by_count[8] >= min(
            seconds
            for worker_count, seconds in run_seconds_by_count.items()
            if worker_count != 8
    ):
        raise AssertionError("8-worker run_seconds must be fastest.")
    if (
            run_seconds_by_count[8] >= TICKET_0018_BEST_RUN_SECONDS
            or total_seconds_by_count[8] >= TICKET_0018_BEST_TOTAL_SECONDS
    ):
        raise AssertionError(
            "8-worker Ticket 0019 result must beat Ticket 0018 baseline "
            f"run_seconds={TICKET_0018_BEST_RUN_SECONDS} and "
            f"total_seconds={TICKET_0018_BEST_TOTAL_SECONDS}."
        )


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

    observed_node_count = diagnostics_map["observed_node_count"]
    if (
            not isinstance(observed_node_count, Integral)
            or int(observed_node_count) <= 0
    ):
        raise AssertionError(
            "observed_node_count must be a positive node count."
        )
    node_ingress_process_count = diagnostics_map["node_ingress_process_count"]
    if int(node_ingress_process_count) != int(observed_node_count):
        raise AssertionError(
            "node_ingress_process_count must equal observed_node_count for "
            "healthy benchmark records: one ingress per observed node."
        )
    observed_worker_process_count = diagnostics_map[
        "observed_worker_process_count"
    ]
    if int(observed_worker_process_count) != int(actual_worker_count):
        raise AssertionError(
            "observed_worker_process_count must match actual_worker_count "
            "worker process capacity."
        )
    worker_process_ids = diagnostics_map["worker_process_ids"]
    if isinstance(worker_process_ids, (str, bytes)) or not isinstance(
            worker_process_ids,
            Sequence,
    ):
        raise AssertionError("worker_process_ids must be a sequence.")
    if len(worker_process_ids) != int(observed_worker_process_count):
        raise AssertionError(
            "worker_process_ids length must match "
            "observed_worker_process_count."
        )
    worker_pid_set = {int(pid) for pid in worker_process_ids}
    if len(worker_pid_set) != len(worker_process_ids) or any(
            pid <= 0
            for pid in worker_pid_set
    ):
        raise AssertionError(
            "worker_process_ids must contain unique positive process ids."
        )

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
            "load_balancer_queue_length",
            "node_queue_length",
            "failed_eval_count",
    ):
        value = diagnostics_map[name]
        if not isinstance(value, Integral) or int(value) < 0:
            raise AssertionError(f"{name} must be a non-negative integer.")
    if str(diagnostics_map["process_start_method"]) not in {
            "spawn",
            "forkserver",
    }:
        raise AssertionError(
            "process_start_method must be spawn or forkserver."
        )
    if str(diagnostics_map["worker_shutdown_status"]) not in {
            "clean",
            "shutdown",
            "terminated",
    }:
        raise AssertionError(
            "worker_shutdown_status must be clean, shutdown, or terminated."
        )
    if str(diagnostics_map["ipc_endpoint_cleanup_status"]) not in {
            "removed",
            "clean",
            "complete",
    }:
        raise AssertionError(
            "ipc_endpoint_cleanup_status must be removed, clean, or complete."
        )
    failed_by_type = _require_mapping(
        diagnostics_map["failed_eval_count_by_type"],
        "failed_eval_count_by_type",
    )
    for failure_name, failure_count in failed_by_type.items():
        if not str(failure_name):
            raise AssertionError(
                "failed_eval_count_by_type keys must be non-empty."
            )
        if not isinstance(failure_count, Integral) or int(failure_count) < 0:
            raise AssertionError(
                "failed_eval_count_by_type values must be non-negative "
                "integers."
            )
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


def write_v3_performance_split_markdown_report(
        *,
        records: Sequence[Mapping[str, Any]],
        report_dir: Path | str | None = None,
) -> Path:
    """Write a dated Markdown report for pure-core/distributed split records."""
    output_dir = (
        Path(report_dir)
        if report_dir is not None
        else Path(__file__).resolve().parent / "reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"v3_performance_split_{date.today().isoformat()}.md"

    lines = [
        "# V3 Performance Split Benchmark Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Pure-Core Results",
        "",
        "| row | mode | setup s | compile s | run s | result s | MC s | likelihood evals | accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        if record["execution_mode"] != "pure_core":
            continue
        lines.append(_split_report_result_row(record))
    lines.extend([
        "",
        "## Distributed Results",
        "",
        "| row | mode | setup s | compile s | run s | result s | MC s | likelihood evals | accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for record in records:
        if record["execution_mode"] != "distributed":
            continue
        lines.append(_split_report_result_row(record))
    lines.extend([
        "",
        "## Diagnostics",
        "",
        "- JAX cache/static-shape diagnostics are reported for pure_core rows.",
        "- worker topology diagnostics are reported for distributed rows.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _split_report_result_row(record: Mapping[str, Any]) -> str:
    metadata_map = _require_mapping(record["metadata"], "metadata")
    timings_map = _require_mapping(record["timings"], "timings")
    results_map = _require_mapping(record["results"], "results")
    return (
        "| "
        f"{metadata_map['feature_row_id']} | "
        f"{record['execution_mode']} | "
        f"{float(timings_map['setup_seconds']):.3f} | "
        f"{float(timings_map['compile_seconds']):.3f} | "
        f"{float(timings_map['run_seconds']):.3f} | "
        f"{float(timings_map['result_conversion_seconds']):.3f} | "
        f"{float(timings_map['mc_shrinkage_seconds']):.3f} | "
        f"{int(results_map['likelihood_evaluations'])} | "
        f"{bool(results_map['accuracy_passed'])} |"
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
        "--pure-core",
        action="store_true",
        help=(
            "Run the direct pure-core 8D basic_mvn isotropic benchmark gate "
            "without LoadBalancerClient."
        ),
    )
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

    if args.pure_core:
        if args.worker_scaling:
            parser.error("--pure-core cannot be combined with --worker-scaling.")
        if args.allocation_target:
            allocation_targets = set(args.allocation_target)
            feature_row_ids = tuple(
                row.row_id
                for row in benchmark_feature_rows()
                if (
                    row.problem_fixture == PROBLEM_NAME
                    and row.allocation_target in allocation_targets
                )
            )
        else:
            feature_row_ids = None
        config = default_pure_core_standard_problem_speed_config(
            feature_row_ids=feature_row_ids,
            seed=args.seed,
            target_num_live_points=args.target_num_live_points,
            live_points_per_dimension=args.live_points_per_dimension,
            max_samples=args.max_samples,
            shell_size=args.shell_size,
            num_slices=args.num_slices,
            phantom_burn_in=args.phantom_burn_in,
            mc_sample_count=args.mc_sample_count,
        )
        records = collect_pure_core_standard_problem_speed_records(config=config)
        print(json.dumps(records, sort_keys=True))
        if not args.no_report:
            report_path = write_v3_performance_split_markdown_report(
                records=records,
                report_dir=args.report_dir,
            )
            print(f"Wrote Markdown report to {report_path}", file=sys.stderr)
        return 0

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
        worker_specs_grid = (
            tuple((worker_spec,) for worker_spec in worker_specs)
            if args.worker_specs
            else default_worker_scaling_specs()
        )
        records = collect_worker_scaling_speed_records(
            config=config,
            worker_specs_grid=worker_specs_grid,
        )
        if args.worker_specs is None and worker_specs_grid == (
                default_worker_scaling_specs()
        ):
            assert_worker_scaling_speed_records(records)
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


def _split_metadata_from_feature_row(
        feature_row: V3PerformanceFeatureRow,
        *,
        seed: int,
        config: StandardProblemSpeedConfig,
) -> dict[str, Any]:
    """Return split benchmark metadata copied from one manifest row.

    Args:
        feature_row: Shared v3 performance manifest row.
        seed: PRNG seed used for this benchmark run.

    Returns:
        Metadata mapping accepted by ``split_schema``.
    """
    return {
        "feature_row_id": feature_row.row_id,
        "problem_fixture": feature_row.problem_fixture,
        "dimension": int(feature_row.dimension),
        "allocation_target": feature_row.allocation_target,
        "seed": int(seed),
        "target_num_live_points": int(config.target_num_live_points),
        "max_samples": int(config.max_samples),
        "shell_size": int(config.shell_size),
        "num_slices": int(config.num_slices),
        "phantom_burn_in": int(config.phantom_burn_in),
        "mc_sample_count": int(config.mc_sample_count),
        "depth_condition": feature_row.depth_condition,
        "goal_condition": feature_row.goal_condition,
        "phantom_enabled": bool(feature_row.phantom_enabled),
        "c_min": feature_row.c_min,
        "direction_kernel": feature_row.direction_kernel,
        "trajectory_mode": feature_row.trajectory_mode,
        "resume_pattern": feature_row.resume_pattern,
        "performance_gate_seconds": feature_row.performance_gate_seconds,
    }


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
    coerced = default_standard_problem_speed_config(
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
    if "feature_row_ids" in data:
        coerced = dataclasses.replace(
            coerced,
            feature_row_ids=tuple(data["feature_row_ids"]),
        )
    return coerced


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


def _worker_count_from_record_metadata(metadata: Mapping[str, Any]) -> int:
    worker_specs = metadata["worker_specs"]
    _assert_worker_specs(worker_specs)
    worker_count = 0
    for worker_spec in worker_specs:
        parts = str(worker_spec).split(":")
        if len(parts) != 3:
            raise AssertionError(
                "worker_specs entries must use device:ids:count format."
            )
        try:
            per_device_count = int(parts[2])
        except ValueError as error:
            raise AssertionError(
                "worker_specs worker count must be an integer."
            ) from error
        if per_device_count <= 0:
            raise AssertionError(
                "worker_specs worker count must be positive."
            )
        device_ids = parts[1].split(",")
        worker_count += len(device_ids) * per_device_count
    return worker_count


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
