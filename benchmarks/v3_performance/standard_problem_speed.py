"""Speed benchmarks for the v3 8D standard-problem local runtime."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
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
)

tfpd = tfp.distributions
perf_counter = time.perf_counter


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
    """Return an explicit speed-benchmark configuration.

    Args:
        allocation_targets: Allocation targets to benchmark.
        seed: Deterministic run seed.
        worker_specs: Public local-runtime worker specifications.
        target_num_live_points: Absolute target live-point count.
        live_points_per_dimension: Optional live-points-per-dimension policy.
        max_samples: Maximum samples for the benchmark run.
        shell_size: Race-tree shell size.
        num_slices: Slice-sampler slice count.
        phantom_burn_in: Slice-sampler phantom burn-in count.
        direction_kernel: Direction kernel mode passed to the sampler.
        mc_sample_count: Number of MC shrinkage samples to draw.

    Returns:
        A frozen dataclass suitable for collection or JSON metadata.
    """
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

    policy = _live_point_policy(
        target_num_live_points=target_num_live_points,
        live_points_per_dimension=live_points_per_dimension,
    )
    config = StandardProblemSpeedConfig(
        allocation_targets=tuple(allocation_targets),
        seed=int(seed),
        worker_specs=_normalize_worker_specs(worker_specs),
        target_num_live_points=int(target_num_live_points),
        live_points_per_dimension=float(live_points_per_dimension),
        live_point_policy=policy,
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
    if config is None:
        config = default_standard_problem_speed_config()
    else:
        config = _coerce_config(config)

    records = []
    for allocation_target in config.allocation_targets:
        records.append(
            _collect_one_allocation_record(
                config=config,
                allocation_target=allocation_target,
            )
        )
    return records


def collect_worker_scaling_speed_records(
        config: StandardProblemSpeedConfig | Mapping[str, Any] | None = None,
        worker_specs_grid: Sequence[Sequence[str] | str] | None = None,
) -> list[dict[str, Any]]:
    """Run speed records for a grid of local worker specifications.

    Each worker-spec grid point is collected through the same public local-LB
    path as :func:`collect_standard_problem_speed_records`, so every
    allocation-target/scaling run owns and tears down its own local load
    balancer.
    """
    if config is None:
        base_config = default_standard_problem_speed_config()
    else:
        base_config = _coerce_config(config)
    if worker_specs_grid is None:
        worker_specs_grid = (base_config.worker_specs,)

    records: list[dict[str, Any]] = []
    for worker_specs in worker_specs_grid:
        scaling_config = dataclasses.replace(
            base_config,
            worker_specs=_normalize_worker_specs_grid_item(worker_specs),
        )
        _validate_config(scaling_config)
        records.extend(
            collect_standard_problem_speed_records(config=scaling_config)
        )
    return records


def compute_timing_fractions(
        timings: Mapping[str, Any],
) -> dict[str, float]:
    """Return timing fractions derived from measured seconds."""
    measured_names = tuple(
        name for name in REQUIRED_TIMING_FIELDS
        if name != "total_seconds"
    )
    measured_total = sum(_finite_non_negative(timings[name], name)
                         for name in measured_names)
    if measured_total <= 0.0:
        raise ValueError("timing total must be positive.")
    return {
        name.removesuffix("_seconds") + "_fraction": (
            _finite_non_negative(timings[name], name) / measured_total
        )
        for name in measured_names
    }


def assert_standard_problem_speed_record(record: Mapping[str, Any]) -> None:
    """Assert that a speed record matches the public Ticket 0016 schema."""
    _require_mapping(record, "record")
    for key in ("schema_version", "metric_family", "metadata", "timings",
                "timing_fractions", "diagnostics", "results"):
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

    _assert_metadata(metadata_map)
    _assert_diagnostics(diagnostics_map)
    _assert_results(results_map)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that writes JSON speed records to stdout."""
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
        worker_specs_grid = tuple((worker_spec,) for worker_spec in worker_specs)
        records = collect_worker_scaling_speed_records(
            config=config,
            worker_specs_grid=worker_specs_grid,
        )
    else:
        records = collect_standard_problem_speed_records(config=config)
    print(json.dumps(records, sort_keys=True))
    return 0


def build_basic_mvn_problem() -> tuple[Model, float]:
    """Build the full 8D ``basic_mvn`` standard problem and reference logZ."""
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    log_z_ref = _log_normal(data_mu, prior_mu, prior_cov + data_cov)
    return Model(prior_model=_basic_mvn_prior_model), float(log_z_ref)


def _collect_one_allocation_record(
        *,
        config: StandardProblemSpeedConfig,
        allocation_target: str,
) -> dict[str, Any]:
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
        actual_worker_count = _actual_worker_count(lb)
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

        worker_sampler_latency_seconds = _worker_sampler_latency_seconds(result)

        mc_started = perf_counter()
        mc_shrinkage_samples = result.sample_mc_shrinkage(
            num_samples=config.mc_sample_count
        )
        _materialize_mc_shrinkage_samples(mc_shrinkage_samples)
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
        "metadata": _metadata_for_config(
            config=config,
            allocation_target=allocation_target,
        ),
        "timings": timings,
        "timing_fractions": compute_timing_fractions(timings),
        "diagnostics": {
            "actual_worker_count": int(actual_worker_count),
            "worker_sampler_latency_seconds": float(
                worker_sampler_latency_seconds
            ),
        },
        "results": _results_record(
            result=result,
            log_z_ref=log_z_ref,
        ),
    }
    assert_standard_problem_speed_record(record)
    return _json_roundtrip_record(record)


def _metadata_for_config(
        *,
        config: StandardProblemSpeedConfig,
        allocation_target: str,
) -> dict[str, Any]:
    return {
        "problem": config.problem,
        "dimension": int(config.dimension),
        "allocation_target": allocation_target,
        "seed": int(config.seed),
        "worker_specs": list(config.worker_specs),
        "target_num_live_points": int(config.target_num_live_points),
        "live_points_per_dimension": float(config.live_points_per_dimension),
        "live_point_policy": str(config.live_point_policy),
        "max_samples": int(config.max_samples),
        "shell_size": int(config.shell_size),
        "num_slices": int(config.num_slices),
        "phantom_burn_in": int(config.phantom_burn_in),
        "direction_kernel": config.direction_kernel,
        "mc_sample_count": int(config.mc_sample_count),
        "software_version": _software_version(),
    }


def _results_record(
        *,
        result: Any,
        log_z_ref: float,
) -> dict[str, Any]:
    return {
        "likelihood_evaluations": int(
            getattr(result, "total_num_likelihood_evaluations")
        ),
        "total_samples": int(getattr(result, "total_num_samples")),
        "log_Z_mean": float(getattr(result, "log_Z_mean")),
        "log_Z_uncert": float(getattr(result, "log_Z_uncert")),
        "logZ_ref": float(log_z_ref),
    }


def _worker_sampler_latency_seconds(result: Any) -> float:
    diagnostics = _get_result_diagnostics(result)
    worker_runtime = getattr(diagnostics, "worker_runtime", None)
    dispatch_records = getattr(worker_runtime, "dispatch_records", ())
    total = 0.0
    for dispatch_record in dispatch_records:
        if isinstance(dispatch_record, Mapping):
            value = dispatch_record.get("sampler_execution_latency_seconds", 0.0)
        else:
            value = getattr(
                dispatch_record,
                "sampler_execution_latency_seconds",
                0.0,
            )
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and value >= 0.0:
            total += value
    return float(total)


def _actual_worker_count(lb: Any) -> int:
    compute_sectors = getattr(lb, "compute_sectors", ())
    worker_count = sum(
        int(getattr(compute_sector, "num_workers", 0))
        for compute_sector in compute_sectors
    )
    return int(worker_count)


def _materialize_mc_shrinkage_samples(samples: Any) -> None:
    _block_until_ready(samples)
    for attr_name in (
        "log_Z_samples",
        "log_L_blocks",
        "log_L_samples",
        "sample_indices",
    ):
        if hasattr(samples, attr_name):
            _block_until_ready(getattr(samples, attr_name))


def _block_until_ready(value: Any) -> None:
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


def _get_result_diagnostics(result: Any) -> Any:
    if hasattr(result, "get_diagnostics"):
        return result.get_diagnostics()
    return getattr(result, "execution_diagnostics", None)


def _coerce_config(
        config: StandardProblemSpeedConfig | Mapping[str, Any],
) -> StandardProblemSpeedConfig:
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
    if config.problem != PROBLEM_NAME:
        raise ValueError(f"Only {PROBLEM_NAME!r} is supported.")
    if config.dimension != PROBLEM_DIMENSION:
        raise ValueError(f"Only {PROBLEM_DIMENSION} dimensions are supported.")
    if not config.allocation_targets:
        raise ValueError("At least one allocation target is required.")
    unknown_targets = set(config.allocation_targets) - set(DEFAULT_ALLOCATION_TARGETS)
    if unknown_targets:
        raise ValueError(f"Unknown allocation target(s): {sorted(unknown_targets)}")
    if not config.worker_specs:
        raise ValueError("At least one worker spec is required.")
    _validate_worker_specs(config.worker_specs)
    _assert_positive_int(config.seed, "seed", allow_zero=True)
    _assert_positive_int(config.target_num_live_points, "target_num_live_points")
    _assert_positive_int(config.max_samples, "max_samples")
    _assert_positive_int(config.shell_size, "shell_size")
    _assert_positive_int(config.num_slices, "num_slices")
    _assert_positive_int(config.phantom_burn_in, "phantom_burn_in", allow_zero=True)
    if int(config.phantom_burn_in) > int(config.num_slices) - 1:
        raise ValueError(
            "phantom_burn_in must satisfy 0 <= burn_in <= num_slices - 1."
        )
    _assert_positive_int(config.mc_sample_count, "mc_sample_count")


def _live_point_policy(
        *,
        target_num_live_points: int,
        live_points_per_dimension: float,
) -> str:
    if math.isclose(
        float(live_points_per_dimension),
        50.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "50_per_dimension"
    if int(target_num_live_points) == 30:
        return "accepted_standard_gate"
    return "custom"


def _assert_metadata(metadata_map: Mapping[str, Any]) -> None:
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
        allow_zero = name in {"seed", "phantom_burn_in"}
        _assert_positive_int(metadata_map[name], name, allow_zero=allow_zero)
    live_points_per_dimension = float(metadata_map["live_points_per_dimension"])
    if not math.isfinite(live_points_per_dimension):
        raise AssertionError("live_points_per_dimension must be finite.")
    if live_points_per_dimension <= 0.0:
        raise AssertionError("live_points_per_dimension must be positive.")
    expected_lppd = (
        float(metadata_map["target_num_live_points"]) / PROBLEM_DIMENSION
    )
    if not math.isclose(
        live_points_per_dimension,
        expected_lppd,
        rel_tol=1e-9,
        abs_tol=1e-12,
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


def _assert_results(results_map: Mapping[str, Any]) -> None:
    for name in ("log_Z_mean", "log_Z_uncert", "logZ_ref"):
        value = float(results_map[name])
        if not math.isfinite(value):
            raise AssertionError(f"results.{name} must be finite.")
    for name in ("likelihood_evaluations", "total_samples"):
        value = results_map[name]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(f"results.{name} must be positive.")


def _assert_diagnostics(diagnostics_map: Mapping[str, Any]) -> None:
    _finite_non_negative(
        diagnostics_map["worker_sampler_latency_seconds"],
        "worker_sampler_latency_seconds",
    )
    if "actual_worker_count" in diagnostics_map:
        value = diagnostics_map["actual_worker_count"]
        if not isinstance(value, Integral) or int(value) <= 0:
            raise AssertionError(
                "diagnostics.actual_worker_count must be positive."
            )


def _assert_worker_specs(worker_specs: Any) -> None:
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


def _normalize_worker_specs(worker_specs: Sequence[str]) -> tuple[str, ...]:
    if isinstance(worker_specs, (str, bytes)):
        raise ValueError("worker_specs must be a non-string sequence.")
    try:
        normalized = tuple(worker_specs)
    except TypeError as e:
        raise ValueError("worker_specs must be a sequence.") from e
    _validate_worker_specs(normalized)
    return normalized


def _normalize_worker_specs_grid_item(
        worker_specs: Sequence[str] | str,
) -> tuple[str, ...]:
    if isinstance(worker_specs, str):
        return (worker_specs,)
    return _normalize_worker_specs(worker_specs)


def _validate_worker_specs(worker_specs: Any) -> None:
    if isinstance(worker_specs, (str, bytes)):
        raise ValueError("worker_specs must be a non-string sequence.")
    if not isinstance(worker_specs, Sequence):
        raise ValueError("worker_specs must be a sequence.")
    if not worker_specs:
        raise ValueError("worker_specs must be non-empty.")
    for worker_spec in worker_specs:
        if not isinstance(worker_spec, str) or not worker_spec:
            raise ValueError("worker_specs entries must be non-empty strings.")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be a mapping.")
    return value


def _assert_required_keys(
        mapping: Mapping[str, Any],
        required_keys: Sequence[str],
        label: str,
) -> None:
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise AssertionError(
            f"{label} missing required field(s): {', '.join(missing)}."
        )


def _finite_non_negative(value: Any, name: str) -> float:
    if not isinstance(value, Real):
        raise AssertionError(f"{name} must be a real timing value.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AssertionError(f"{name} must be finite.")
    if numeric < 0.0:
        raise AssertionError(f"{name} must be non-negative (>= 0).")
    return numeric


def _assert_positive_int(
        value: Any,
        name: str,
        *,
        allow_zero: bool = False,
) -> None:
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    minimum = 0 if allow_zero else 1
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _software_version() -> str:
    try:
        return metadata.version("jaxns")
    except metadata.PackageNotFoundError:
        return "unknown"


def _json_roundtrip_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record, sort_keys=True))


def _basic_mvn_parameters():
    prior_mu = 6 * jnp.ones(PROBLEM_DIMENSION)
    prior_cov = np.eye(PROBLEM_DIMENSION)
    prior_cov[prior_cov == 0] = 0.99
    data_mu = jnp.zeros(PROBLEM_DIMENSION)
    data_cov = jnp.eye(PROBLEM_DIMENSION)
    return prior_mu, prior_cov, data_mu, data_cov


def _basic_mvn_prior_model():
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
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
    l_factor = jnp.linalg.cholesky(cov)
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
