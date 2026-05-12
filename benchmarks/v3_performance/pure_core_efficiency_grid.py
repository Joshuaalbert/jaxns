"""Pure-core efficiency grid for 8D standard problems.

The benchmark compares settings by likelihood work at fixed evidence
precision. Per-run efficiency uses ``likelihood_evaluations * Var(logZ)``.
Grouped efficiency uses ``mean(likelihood_evaluations) * RMSE(logZ)^2`` across
seeds against the analytic reference log evidence.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REPO_SRC = _REPO_ROOT / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _preconfigure_jax_platform_from_argv(argv: Sequence[str]) -> None:
    """Set the JAX platform environment before importing JAX."""
    platform = "default"
    for index, arg in enumerate(argv):
        if arg == "--jax-platform" and index + 1 < len(argv):
            platform = argv[index + 1]
        elif arg.startswith("--jax-platform="):
            platform = arg.split("=", 1)[1]
    if platform != "default":
        os.environ["JAX_PLATFORM_NAME"] = platform


_preconfigure_jax_platform_from_argv(sys.argv[1:])

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.termination_condition import TerminationCondition
from jaxctx.priors.prior import Prior


tfpd = tfp.distributions
perf_counter = time.perf_counter

SCHEMA_VERSION = "v3_pure_core_efficiency_grid_v1"
DEFAULT_ALLOCATION_TARGETS = (
    "uniform",
    "evidence_improving",
    "posterior_improving",
)
DEFAULT_LOGZ_UNCERT_TARGETS = (0.8, 0.5, 0.35)
DEFAULT_SEEDS = (0, 17, 29)
DEFAULT_PROBLEMS = ("basic_mvn", "spike_slab")


@dataclasses.dataclass(frozen=True, slots=True)
class ProblemSpec:
    """One analytic 8D benchmark problem."""

    name: str
    dimension: int
    build: Callable[[], tuple[Model, float]]


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerSetting:
    """One constrained-sampler setting in the efficiency grid."""

    setting_id: str
    direction_kernel: str
    num_slices: int
    phantom_burn_in: int
    no_step_out: bool = True
    max_shrinkage_steps: int = 32


@dataclasses.dataclass(frozen=True, slots=True)
class EfficiencyGridConfig:
    """Configuration for the pure-core efficiency grid."""

    problems: tuple[str, ...] = DEFAULT_PROBLEMS
    allocation_targets: tuple[str, ...] = DEFAULT_ALLOCATION_TARGETS
    logz_uncert_targets: tuple[float, ...] = DEFAULT_LOGZ_UNCERT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    sampler_settings: tuple[SamplerSetting, ...] = ()
    target_num_live_points: int = 30
    max_samples: int = 2400
    shell_size: int = 15
    min_samples: int = 0
    delta_k: int = 1
    mc_sample_count: int = 512
    max_goal_iterations: int = 256
    report_dir: Path = Path("benchmarks/v3_performance/reports")
    record_json: bool = True
    row_isolation: bool = False


def default_sampler_settings() -> tuple[SamplerSetting, ...]:
    """Return the default grid of readable sampler settings."""
    return (
        SamplerSetting(
            setting_id="isotropic_12",
            direction_kernel="isotropic",
            num_slices=12,
            phantom_burn_in=2,
        ),
        SamplerSetting(
            setting_id="isotropic_24",
            direction_kernel="isotropic",
            num_slices=24,
            phantom_burn_in=4,
        ),
        SamplerSetting(
            setting_id="isotropic_24_stepout",
            direction_kernel="isotropic",
            num_slices=24,
            phantom_burn_in=4,
            no_step_out=False,
        ),
        SamplerSetting(
            setting_id="isotropic_24_ms64",
            direction_kernel="isotropic",
            num_slices=24,
            phantom_burn_in=4,
            max_shrinkage_steps=64,
        ),
        SamplerSetting(
            setting_id="ellipsoidal_12",
            direction_kernel="ellipsoidal",
            num_slices=12,
            phantom_burn_in=2,
        ),
        SamplerSetting(
            setting_id="gmm_12",
            direction_kernel="gmm",
            num_slices=12,
            phantom_burn_in=2,
        ),
        SamplerSetting(
            setting_id="gmm_12_ms64",
            direction_kernel="gmm",
            num_slices=12,
            phantom_burn_in=2,
            max_shrinkage_steps=64,
        ),
    )


def smoke_sampler_settings() -> tuple[SamplerSetting, ...]:
    """Return a minimal setting grid for harness smoke checks."""
    return (
        SamplerSetting(
            setting_id="isotropic_6",
            direction_kernel="isotropic",
            num_slices=6,
            phantom_burn_in=1,
        ),
        SamplerSetting(
            setting_id="gmm_6",
            direction_kernel="gmm",
            num_slices=6,
            phantom_burn_in=1,
        ),
    )


def run_efficiency_grid(config: EfficiencyGridConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all grid rows and return per-run records plus grouped rollups."""
    records: list[dict[str, Any]] = []
    cases = list(iter_grid_cases(config))
    for run_index, case in enumerate(cases, start=1):
        problem, logz_uncert_target, allocation_target, sampler_setting, seed = case
        print(
            (
                f"[{run_index}/{len(cases)}] {problem.name} "
                f"target={logz_uncert_target:g} "
                f"alloc={allocation_target} "
                f"setting={sampler_setting.setting_id} "
                f"seed={seed}"
            ),
            flush=True,
        )
        records.append(
            run_one_case(
                problem=problem,
                logz_uncert_target=float(logz_uncert_target),
                allocation_target=allocation_target,
                sampler_setting=sampler_setting,
                seed=int(seed),
                config=config,
            )
        )
    rollups = rollup_records(records)
    return records, rollups


def run_efficiency_grid_isolated(
        config: EfficiencyGridConfig,
        *,
        jax_platform: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run each benchmark row in a fresh Python process and combine records."""
    records: list[dict[str, Any]] = []
    cases = list(iter_grid_cases(config))
    with tempfile.TemporaryDirectory(prefix="jaxns_efficiency_grid_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for run_index, case in enumerate(cases, start=1):
            problem, logz_uncert_target, allocation_target, sampler_setting, seed = case
            print(
                (
                    f"[{run_index}/{len(cases)}] {problem.name} "
                    f"target={logz_uncert_target:g} "
                    f"alloc={allocation_target} "
                    f"setting={sampler_setting.setting_id} "
                    f"seed={seed}"
                ),
                flush=True,
            )
            record_path = tmp_path / f"record_{run_index:04d}.json"
            command = [
                sys.executable,
                "-m",
                "benchmarks.v3_performance.pure_core_efficiency_grid",
                "--preset",
                "smoke",
                "--problem",
                problem.name,
                "--allocation-target",
                allocation_target,
                "--setting",
                sampler_setting.setting_id,
                "--logz-uncert-target",
                str(float(logz_uncert_target)),
                "--seed",
                str(int(seed)),
                "--target-num-live-points",
                str(int(config.target_num_live_points)),
                "--max-samples",
                str(int(config.max_samples)),
                "--shell-size",
                str(int(config.shell_size)),
                "--min-samples",
                str(int(config.min_samples)),
                "--delta-k",
                str(int(config.delta_k)),
                "--mc-sample-count",
                str(int(config.mc_sample_count)),
                "--max-goal-iterations",
                str(int(config.max_goal_iterations)),
                "--report-dir",
                str(config.report_dir),
                "--single-record-path",
                str(record_path),
            ]
            if jax_platform != "default":
                command.extend(["--jax-platform", jax_platform])
            env = dict(os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                if completed.stdout:
                    print(completed.stdout, flush=True)
                if completed.stderr:
                    print(completed.stderr, flush=True)
                raise RuntimeError(
                    "Isolated benchmark row failed with exit code "
                    f"{completed.returncode}: {' '.join(command)}"
                )
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
    rollups = rollup_records(records)
    return records, rollups


def iter_grid_cases(config: EfficiencyGridConfig):
    """Yield concrete benchmark cases in deterministic report order."""
    problems = problem_specs_by_name()
    for problem_name in config.problems:
        problem = problems[problem_name]
        for logz_uncert_target in config.logz_uncert_targets:
            for allocation_target in config.allocation_targets:
                for sampler_setting in config.sampler_settings:
                    for seed in config.seeds:
                        yield (
                            problem,
                            float(logz_uncert_target),
                            allocation_target,
                            sampler_setting,
                            int(seed),
                        )


def run_one_case(
        *,
        problem: ProblemSpec,
        logz_uncert_target: float,
        allocation_target: str,
        sampler_setting: SamplerSetting,
        seed: int,
        config: EfficiencyGridConfig,
) -> dict[str, Any]:
    """Run one pure-core benchmark case and build its JSON record."""
    setup_started = perf_counter()
    model, logz_ref = problem.build()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=sampler_setting.num_slices,
        no_step_out=sampler_setting.no_step_out,
        max_shrinkage_steps=sampler_setting.max_shrinkage_steps,
        collect_phantom_samples=True,
        phantom_burn_in=sampler_setting.phantom_burn_in,
        direction_kernel=sampler_setting.direction_kernel,
        trajectory="straight_line",
    )
    runner = NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=config.target_num_live_points,
        max_samples=config.max_samples,
        shell_size=config.shell_size,
        collect_phantom_samples=True,
        store_phantom_samples=True,
    )
    setup_seconds = perf_counter() - setup_started

    run_started = perf_counter()

    def goal_cond(state) -> bool:
        result = state.to_result().trim()
        reached_min_samples = int(result.total_num_samples) >= int(
            config.min_samples
        )
        reached_uncert = float(result.log_Z_uncert) <= float(
            logz_uncert_target
        )
        return reached_min_samples and reached_uncert

    state = runner.run_until_goal(
        goal_cond=goal_cond,
        depth_cond=TerminationCondition(max_samples=config.max_samples),
        allocation_target=allocation_target,
        key=jax.random.PRNGKey(seed),
        max_goal_iterations=config.max_goal_iterations,
        delta_K=config.delta_k,
    )
    _block_until_ready(state.samples.log_likelihoods)
    run_seconds = perf_counter() - run_started

    result_started = perf_counter()
    result = state.to_result().trim()
    result_conversion_seconds = perf_counter() - result_started

    mc_started = perf_counter()
    mc_seed = _stable_seed(
        problem.name,
        allocation_target,
        sampler_setting.setting_id,
        logz_uncert_target,
        seed,
    )
    mc_samples = result.sample_mc_shrinkage(
        num_samples=config.mc_sample_count,
        key=jax.random.PRNGKey(mc_seed),
    )
    _block_until_ready(mc_samples.log_Z_samples)
    logz_samples = np.asarray(mc_samples.log_Z_samples, dtype=float)
    mc_seconds = perf_counter() - mc_started

    logz_mean = float(np.mean(logz_samples))
    logz_var = float(np.var(logz_samples))
    logz_std = float(np.sqrt(logz_var))
    error = logz_mean - float(logz_ref)
    squared_error = float(error * error)
    accuracy_within_3_mc_std = bool(abs(error) <= 3.0 * logz_std)
    abs_error_over_mc_std = float(abs(error) / max(logz_std, 1e-12))
    likelihood_evaluations = int(result.total_num_likelihood_evaluations)
    total_seconds = float(
        setup_seconds + run_seconds + result_conversion_seconds + mc_seconds
    )
    reached_uncert_target = float(result.log_Z_uncert) <= float(
        logz_uncert_target
    )
    reached_sample_cap = int(result.total_num_samples) >= int(config.max_samples)

    return {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": "pure_core",
        "metadata": {
            "problem": problem.name,
            "dimension": int(problem.dimension),
            "seed": int(seed),
            "allocation_target": allocation_target,
            "logz_uncert_target": float(logz_uncert_target),
            "sampler_setting": sampler_setting.setting_id,
            "direction_kernel": sampler_setting.direction_kernel,
            "num_slices": int(sampler_setting.num_slices),
            "phantom_burn_in": int(sampler_setting.phantom_burn_in),
            "no_step_out": bool(sampler_setting.no_step_out),
            "max_shrinkage_steps": int(sampler_setting.max_shrinkage_steps),
            "target_num_live_points": int(config.target_num_live_points),
            "max_samples": int(config.max_samples),
            "shell_size": int(config.shell_size),
            "min_samples": int(config.min_samples),
            "delta_k": int(config.delta_k),
            "mc_sample_count": int(config.mc_sample_count),
            "mc_seed": int(mc_seed),
            "jax_platform": str(jax.default_backend()),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "timings": {
            "setup_seconds": float(setup_seconds),
            "run_seconds": float(run_seconds),
            "result_conversion_seconds": float(result_conversion_seconds),
            "mc_shrinkage_seconds": float(mc_seconds),
            "total_seconds": total_seconds,
        },
        "results": {
            "likelihood_evaluations": likelihood_evaluations,
            "total_samples": int(result.total_num_samples),
            "log_Z_mean": float(result.log_Z_mean),
            "log_Z_uncert": float(result.log_Z_uncert),
            "mc_log_Z_mean": logz_mean,
            "mc_log_Z_std": logz_std,
            "mc_log_Z_variance": logz_var,
            "logZ_ref": float(logz_ref),
            "logZ_error": float(error),
            "squared_error": squared_error,
            "accuracy_within_3_mc_std": accuracy_within_3_mc_std,
            "abs_error_over_mc_std": abs_error_over_mc_std,
            "likelihood_evals_times_logZ_variance": float(
                likelihood_evaluations * logz_var
            ),
            "likelihood_evals_times_squared_error": float(
                likelihood_evaluations * squared_error
            ),
            "reached_uncert_target": bool(reached_uncert_target),
            "reached_min_samples": bool(
                int(result.total_num_samples) >= int(config.min_samples)
            ),
            "reached_sample_cap": bool(reached_sample_cap),
        },
    }


def rollup_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed records by problem, target, allocation, and setting."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        key = (
            metadata["problem"],
            metadata["logz_uncert_target"],
            metadata["allocation_target"],
            metadata["sampler_setting"],
        )
        groups[key].append(record)

    rollups: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        problem, target, allocation_target, sampler_setting = key
        evals = np.asarray(
            [item["results"]["likelihood_evaluations"] for item in group],
            dtype=float,
        )
        wall = np.asarray(
            [item["timings"]["total_seconds"] for item in group],
            dtype=float,
        )
        run = np.asarray(
            [item["timings"]["run_seconds"] for item in group],
            dtype=float,
        )
        errors = np.asarray(
            [item["results"]["logZ_error"] for item in group],
            dtype=float,
        )
        variances = np.asarray(
            [item["results"]["mc_log_Z_variance"] for item in group],
            dtype=float,
        )
        variance_efficiencies = np.asarray(
            [
                item["results"]["likelihood_evals_times_logZ_variance"]
                for item in group
            ],
            dtype=float,
        )
        mse = float(np.mean(np.square(errors)))
        rmse = float(np.sqrt(mse))
        mean_evals = float(np.mean(evals))
        rollups.append(
            {
                "problem": problem,
                "logz_uncert_target": float(target),
                "allocation_target": allocation_target,
                "sampler_setting": sampler_setting,
                "num_seeds": int(len(group)),
                "mean_likelihood_evaluations": mean_evals,
                "median_likelihood_evaluations": float(np.median(evals)),
                "mean_wall_seconds": float(np.mean(wall)),
                "median_wall_seconds": float(np.median(wall)),
                "mean_run_seconds": float(np.mean(run)),
                "median_run_seconds": float(np.median(run)),
                "rmse_logZ": rmse,
                "bias_logZ": float(np.mean(errors)),
                "mean_mc_logZ_variance": float(np.mean(variances)),
                "mean_evals_times_variance": float(
                    np.mean(variance_efficiencies)
                ),
                "mean_evals_times_mse": float(mean_evals * mse),
                "target_success_fraction": float(
                    np.mean(
                        [
                            bool(item["results"]["reached_uncert_target"])
                            for item in group
                        ]
                    )
                ),
                "accuracy_success_fraction": float(
                    np.mean(
                        [
                            bool(item["results"]["accuracy_within_3_mc_std"])
                            for item in group
                        ]
                    )
                ),
                "sample_cap_fraction": float(
                    np.mean(
                        [
                            bool(item["results"]["reached_sample_cap"])
                            for item in group
                        ]
                    )
                ),
                "min_sample_fraction": float(
                    np.mean(
                        [
                            bool(item["results"]["reached_min_samples"])
                            for item in group
                        ]
                    )
                ),
            }
        )
    return rollups


def write_report(
        *,
        config: EfficiencyGridConfig,
        records: Sequence[Mapping[str, Any]],
        rollups: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path | None]:
    """Write a dated markdown report and optional JSON records."""
    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    report_path = report_dir / f"pure_core_efficiency_grid_{stamp}.md"
    json_path = (
        report_dir / f"pure_core_efficiency_grid_{stamp}.json"
        if config.record_json
        else None
    )

    best_rows = best_rollups_by_problem_target(rollups)
    usable_best_rows = best_rollups_by_problem_target(
        rollups,
        min_accuracy_fraction=0.5,
    )
    strict_best_rows = best_rollups_by_problem_target(
        rollups,
        min_accuracy_fraction=1.0,
        min_target_fraction=1.0,
    )
    lines = [
        "# Pure-Core Efficiency Grid",
        "",
        f"Generated: `{stamp} UTC`",
        "",
        "## Objective",
        "",
        (
            "Compare pure-JAX nested-sampling settings by likelihood work "
            "needed for fixed log-evidence precision. Lower values are better "
            "for both `mean_evals_times_variance` and `mean_evals_times_mse`."
        ),
        "",
        (
            "The stopping target is `result.log_Z_uncert`; MC shrinkage "
            "variance and analytic-reference RMSE are reported separately. "
            "`mean_evals_times_mse` is `mean(evaluations) * mean(error^2)` "
            "across seeds."
        ),
        "",
        "## Configuration",
        "",
        f"- Problems: `{', '.join(config.problems)}`",
        f"- Allocation targets: `{', '.join(config.allocation_targets)}`",
        f"- LogZ uncertainty targets: `{', '.join(map(str, config.logz_uncert_targets))}`",
        f"- Seeds: `{', '.join(map(str, config.seeds))}`",
        f"- Target live points: `{config.target_num_live_points}`",
        f"- Max samples: `{config.max_samples}`",
        f"- Shell size: `{config.shell_size}`",
        f"- Minimum samples before uncertainty stopping: `{config.min_samples}`",
        f"- delta_K: `{config.delta_k}`",
        f"- MC shrinkage samples: `{config.mc_sample_count}`",
        f"- Row isolation: `{config.row_isolation}`",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- JAX devices: `{', '.join(str(device) for device in jax.devices())}`",
        "",
        "## Best By Problem And Target",
        "",
        _markdown_table(
            best_rows,
            [
                "problem",
                "logz_uncert_target",
                "allocation_target",
                "sampler_setting",
                "mean_likelihood_evaluations",
                "rmse_logZ",
                "mean_evals_times_mse",
                "mean_evals_times_variance",
                "target_success_fraction",
                "accuracy_success_fraction",
            ],
        ),
        "",
        "## Best Usable By Problem And Target",
        "",
        (
            "Rows here require `accuracy_success_fraction >= 0.5`; this "
            "prevents biased early stops from winning solely because the MC "
            "variance estimate is small."
        ),
        "",
        _markdown_table(
            usable_best_rows,
            [
                "problem",
                "logz_uncert_target",
                "allocation_target",
                "sampler_setting",
                "mean_likelihood_evaluations",
                "rmse_logZ",
                "mean_evals_times_mse",
                "mean_evals_times_variance",
                "target_success_fraction",
                "accuracy_success_fraction",
            ],
        ),
        "",
        "## Best Strict By Problem And Target",
        "",
        (
            "Rows here require both `accuracy_success_fraction == 1` and "
            "`target_success_fraction == 1`."
        ),
        "",
        _markdown_table(
            strict_best_rows,
            [
                "problem",
                "logz_uncert_target",
                "allocation_target",
                "sampler_setting",
                "mean_likelihood_evaluations",
                "rmse_logZ",
                "mean_evals_times_mse",
                "mean_evals_times_variance",
                "target_success_fraction",
                "accuracy_success_fraction",
            ],
        ),
        "",
        "## Rollups",
        "",
        _markdown_table(
            rollups,
            [
                "problem",
                "logz_uncert_target",
                "allocation_target",
                "sampler_setting",
                "num_seeds",
                "mean_likelihood_evaluations",
                "mean_run_seconds",
                "mean_wall_seconds",
                "rmse_logZ",
                "bias_logZ",
                "mean_mc_logZ_variance",
                "mean_evals_times_variance",
                "mean_evals_times_mse",
                "target_success_fraction",
                "accuracy_success_fraction",
                "min_sample_fraction",
                "sample_cap_fraction",
            ],
        ),
        "",
        "## Per-Seed Records",
        "",
        _markdown_table(
            per_seed_rows(records),
            [
                "problem",
                "target",
                "allocation",
                "setting",
                "seed",
                "evals",
                "run_s",
                "wall_s",
                "logZ_ref",
                "mc_logZ_mean",
                "logZ_uncert",
                "mc_logZ_std",
                "error",
                "evals_x_var",
                "evals_x_sqerr",
                "err_over_std",
                "accuracy_ok",
                "target_ok",
                "min_samples_ok",
            ],
        ),
        "",
        "## First-Pass Hypotheses",
        "",
    ]
    lines.extend(first_pass_hypotheses(rollups))
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if json_path is not None:
        json_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "config": config_to_json(config),
                    "records": list(records),
                    "rollups": list(rollups),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return report_path, json_path


def best_rollups_by_problem_target(
        rollups: Sequence[Mapping[str, Any]],
        *,
        min_accuracy_fraction: float = 0.0,
        min_target_fraction: float = 0.0,
) -> list[Mapping[str, Any]]:
    """Return the best rollup per problem/target by evals-times-MSE."""
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rollups:
        if float(row["accuracy_success_fraction"]) < float(
            min_accuracy_fraction
        ):
            continue
        if float(row["target_success_fraction"]) < float(min_target_fraction):
            continue
        grouped[(str(row["problem"]), float(row["logz_uncert_target"]))].append(
            row
        )
    best_rows = []
    for _, rows in sorted(grouped.items()):
        best_rows.append(
            min(rows, key=lambda item: float(item["mean_evals_times_mse"]))
        )
    return best_rows


def per_seed_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-run records for compact markdown rendering."""
    rows = []
    for record in records:
        metadata = record["metadata"]
        results = record["results"]
        timings = record["timings"]
        rows.append(
            {
                "problem": metadata["problem"],
                "target": metadata["logz_uncert_target"],
                "allocation": metadata["allocation_target"],
                "setting": metadata["sampler_setting"],
                "seed": metadata["seed"],
                "evals": results["likelihood_evaluations"],
                "run_s": timings["run_seconds"],
                "wall_s": timings["total_seconds"],
                "logZ_ref": results["logZ_ref"],
                "mc_logZ_mean": results["mc_log_Z_mean"],
                "logZ_uncert": results["log_Z_uncert"],
                "mc_logZ_std": results["mc_log_Z_std"],
                "error": results["logZ_error"],
                "evals_x_var": results[
                    "likelihood_evals_times_logZ_variance"
                ],
                "evals_x_sqerr": results[
                    "likelihood_evals_times_squared_error"
                ],
                "err_over_std": results["abs_error_over_mc_std"],
                "accuracy_ok": results["accuracy_within_3_mc_std"],
                "target_ok": results["reached_uncert_target"],
                "min_samples_ok": results["reached_min_samples"],
            }
        )
    return rows


def first_pass_hypotheses(
        rollups: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Generate simple hypotheses from the rollup table."""
    if not rollups:
        return ["- No benchmark rows were produced."]
    best_by_problem = defaultdict(list)
    for row in best_rollups_by_problem_target(rollups):
        best_by_problem[row["problem"]].append(row)
    lines = []
    for problem, rows in sorted(best_by_problem.items()):
        settings = ", ".join(
            f"{row['sampler_setting']}/{row['allocation_target']}"
            for row in rows
        )
        lines.append(
            f"- `{problem}` best rows by target were: {settings}."
        )
    failed_targets = [
        row for row in rollups if float(row["target_success_fraction"]) < 1.0
    ]
    if failed_targets:
        lines.append(
            "- Some rows did not reach the requested uncertainty. Check "
            "`sample_cap_fraction` and `max_goal_iterations` before drawing "
            "strong efficiency conclusions."
        )
    failed_accuracy = [
        row for row in rollups if float(row["accuracy_success_fraction"]) < 1.0
    ]
    if failed_accuracy:
        lines.append(
            "- Some rows failed the analytic logZ accuracy gate; treat those "
            "settings as biased or under-resolved rather than efficient."
        )
    bad_early_targets = [
        row
        for row in rollups
        if (
            float(row["target_success_fraction"]) > 0.0
            and float(row["accuracy_success_fraction"]) == 0.0
        )
    ]
    if bad_early_targets:
        lines.append(
            "- At least one row reached the uncertainty target while missing "
            "the analytic accuracy gate for every seed. Use the minimum-sample "
            "guard or stricter accuracy-gated tables when comparing "
            "`evals * variance`."
        )
    gmm_rows = [
        row for row in rollups if str(row["sampler_setting"]).startswith("gmm")
    ]
    if gmm_rows:
        gmm_best_count = sum(
            1
            for row in best_rollups_by_problem_target(rollups)
            if str(row["sampler_setting"]).startswith("gmm")
        )
        lines.append(
            f"- GMM settings won {gmm_best_count} problem/target groups; "
            "compare against ellipsoidal rows to decide whether GMM fitting "
            "cost is buying lower variance."
        )
    return lines


def problem_specs_by_name() -> dict[str, ProblemSpec]:
    """Return supported 8D analytic problem specs."""
    return {
        "basic_mvn": ProblemSpec(
            name="basic_mvn",
            dimension=8,
            build=build_basic_mvn_problem,
        ),
        "spike_slab": ProblemSpec(
            name="spike_slab",
            dimension=8,
            build=build_spike_slab_problem,
        ),
    }


def build_basic_mvn_problem() -> tuple[Model, float]:
    """Build the 8D Gaussian standard problem and analytic log evidence."""
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    logz_ref = _log_normal(data_mu, prior_mu, prior_cov + data_cov)
    return Model(prior_model=_basic_mvn_prior_model), float(logz_ref)


def build_spike_slab_problem() -> tuple[Model, float]:
    """Build the 8D spike-slab standard problem and analytic log evidence."""
    prior_mean, prior_cov, component_means, component_covs, weights = (
        _spike_slab8_parameters()
    )
    logz_ref = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights,
    )
    return Model(prior_model=_spike_slab8_prior_model), float(logz_ref)


def _basic_mvn_parameters():
    """Return basic MVN parameters with shapes `[D]`, `[D, D]`, `[D]`, `[D, D]`."""
    ndims = 8
    prior_mu = 6 * jnp.ones(ndims)
    prior_cov = np.eye(ndims)
    prior_cov[prior_cov == 0] = 0.99
    data_mu = jnp.zeros(ndims)
    data_cov = jnp.eye(ndims)
    return prior_mu, prior_cov, data_mu, data_cov


def _basic_mvn_prior_model():
    """Return scalar log likelihood for the basic 8D MVN problem."""
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


def _spike_slab8_parameters():
    """Return spike-slab parameters with shapes `[D]`, `[D, D]`, `[K, D]`, `[K, D, D]`, `[K]`."""
    ndims = 8
    prior_mean = jnp.zeros(ndims)
    prior_cov = jnp.diag(4.0 * jnp.ones(ndims))
    component_means = jnp.stack(
        [
            jnp.concatenate([3.5 * jnp.ones(4), jnp.zeros(4)]),
            jnp.concatenate([-3.0 * jnp.ones(4), 1.5 * jnp.ones(4)]),
        ],
        axis=0,
    )
    component_covs = jnp.stack(
        [
            jnp.diag(jnp.concatenate([0.05 * jnp.ones(4), 0.4 * jnp.ones(4)])),
            jnp.diag(jnp.concatenate([0.6 * jnp.ones(4), 0.08 * jnp.ones(4)])),
        ],
        axis=0,
    )
    weights = jnp.asarray([0.25, 0.75])
    return prior_mean, prior_cov, component_means, component_covs, weights


def _spike_slab8_prior_model():
    """Return scalar log likelihood for the spike-slab 8D problem."""
    prior_mean, prior_cov, component_means, component_covs, weights = (
        _spike_slab8_parameters()
    )
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=jnp.linalg.cholesky(prior_cov),
        ),
        name="x",
    ).realise()
    mixture = tfpd.MixtureSameFamily(
        mixture_distribution=tfpd.Categorical(probs=weights),
        components_distribution=tfpd.MultivariateNormalTriL(
            loc=component_means,
            scale_tril=jnp.linalg.cholesky(component_covs),
        ),
    )
    return mixture.log_prob(x)


def _log_normal(x, mean, cov) -> float:
    """Return multivariate normal log density for one point."""
    # x, mean: [D]; cov, scale_tril: [D, D]
    scale_tril = jnp.linalg.cholesky(cov)
    dx = x - mean
    dx = solve_triangular(scale_tril, dx, lower=True)
    log_prob = (
        -0.5 * x.size * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(jnp.diag(scale_tril)))
        - 0.5 * dx @ dx
    )
    return float(log_prob)


def _gaussian_mixture_evidence(
        *,
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
) -> float:
    """Return analytic evidence for a Gaussian prior and Gaussian mixture likelihood."""
    component_logz = jnp.asarray(
        [
            _log_normal(mean, prior_mean, prior_cov + cov)
            for mean, cov in zip(component_means, component_covs)
        ]
    )
    return float(logsumexp(jnp.log(weights) + component_logz))


def _block_until_ready(value: Any) -> None:
    """Synchronize a JAX array or pytree before timing stops."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    leaves = jax.tree.leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _markdown_table(
        rows: Sequence[Mapping[str, Any]],
        columns: Sequence[str],
) -> str:
    """Render a compact markdown table."""
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [_format_markdown_value(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: Any) -> str:
    """Format values for markdown tables."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{float(value):.6g}"
    return str(value)


def _stable_seed(*parts: Any) -> int:
    """Return a deterministic uint32 seed from benchmark row fields."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def config_to_json(config: EfficiencyGridConfig) -> dict[str, Any]:
    """Return a JSON-safe config mapping."""
    return {
        "problems": list(config.problems),
        "allocation_targets": list(config.allocation_targets),
        "logz_uncert_targets": list(config.logz_uncert_targets),
        "seeds": list(config.seeds),
        "sampler_settings": [
            dataclasses.asdict(setting) for setting in config.sampler_settings
        ],
        "target_num_live_points": int(config.target_num_live_points),
        "max_samples": int(config.max_samples),
        "shell_size": int(config.shell_size),
        "min_samples": int(config.min_samples),
        "delta_k": int(config.delta_k),
        "mc_sample_count": int(config.mc_sample_count),
        "max_goal_iterations": int(config.max_goal_iterations),
        "row_isolation": bool(config.row_isolation),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", action="append", choices=sorted(problem_specs_by_name()))
    parser.add_argument("--allocation-target", action="append", choices=DEFAULT_ALLOCATION_TARGETS)
    parser.add_argument("--logz-uncert-target", action="append", type=float)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--setting", action="append")
    parser.add_argument("--target-num-live-points", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=2400)
    parser.add_argument("--shell-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--delta-k", type=int, default=1)
    parser.add_argument("--mc-sample-count", type=int, default=512)
    parser.add_argument("--max-goal-iterations", type=int, default=256)
    parser.add_argument("--report-dir", type=Path, default=Path("benchmarks/v3_performance/reports"))
    parser.add_argument("--jax-platform", choices=("default", "cpu", "gpu"), default="default")
    parser.add_argument(
        "--row-isolation",
        action="store_true",
        help="Run each row in a fresh Python process to release JAX compile memory.",
    )
    parser.add_argument("--single-record-path", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=("smoke", "initial", "full"),
        default="initial",
        help="smoke is tiny, initial is a practical first pass, full expands targets/settings.",
    )
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> EfficiencyGridConfig:
    """Build benchmark config from parsed CLI arguments."""
    settings_by_id = {setting.setting_id: setting for setting in default_sampler_settings()}
    settings_by_id.update(
        {setting.setting_id: setting for setting in smoke_sampler_settings()}
    )
    if args.preset == "smoke":
        default_settings = ("isotropic_6", "gmm_6")
        default_targets = (0.9,)
        default_seeds = (0,)
        default_problems = ("basic_mvn",)
    elif args.preset == "full":
        default_settings = tuple(settings_by_id)
        default_targets = (0.9, 0.7, 0.5, 0.35, 0.25)
        default_seeds = DEFAULT_SEEDS
        default_problems = DEFAULT_PROBLEMS
    else:
        default_settings = ("isotropic_12", "ellipsoidal_12", "gmm_12")
        default_targets = DEFAULT_LOGZ_UNCERT_TARGETS
        default_seeds = DEFAULT_SEEDS
        default_problems = DEFAULT_PROBLEMS

    requested_settings = tuple(args.setting or default_settings)
    unknown_settings = set(requested_settings) - set(settings_by_id)
    if unknown_settings:
        supported = ", ".join(sorted(settings_by_id))
        raise ValueError(
            f"Unknown setting(s) {sorted(unknown_settings)}; supported: {supported}."
        )
    return EfficiencyGridConfig(
        problems=tuple(args.problem or default_problems),
        allocation_targets=tuple(args.allocation_target or DEFAULT_ALLOCATION_TARGETS),
        logz_uncert_targets=tuple(args.logz_uncert_target or default_targets),
        seeds=tuple(args.seed or default_seeds),
        sampler_settings=tuple(settings_by_id[name] for name in requested_settings),
        target_num_live_points=int(args.target_num_live_points),
        max_samples=int(args.max_samples),
        shell_size=int(args.shell_size),
        min_samples=int(args.min_samples),
        delta_k=int(args.delta_k),
        mc_sample_count=int(args.mc_sample_count),
        max_goal_iterations=int(args.max_goal_iterations),
        report_dir=args.report_dir,
        record_json=not bool(args.no_json),
        row_isolation=bool(args.row_isolation),
    )


def configure_jax_platform(platform: str) -> None:
    """Set the requested JAX platform before the first backend use."""
    if platform == "default":
        return
    os.environ["JAX_PLATFORM_NAME"] = platform
    jax.config.update("jax_platform_name", platform)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    configure_jax_platform(args.jax_platform)
    config = config_from_args(args)
    if args.single_record_path is not None:
        cases = list(iter_grid_cases(config))
        if len(cases) != 1:
            raise ValueError("--single-record-path requires exactly one grid row.")
        problem, logz_uncert_target, allocation_target, sampler_setting, seed = (
            cases[0]
        )
        record = run_one_case(
            problem=problem,
            logz_uncert_target=logz_uncert_target,
            allocation_target=allocation_target,
            sampler_setting=sampler_setting,
            seed=seed,
            config=config,
        )
        args.single_record_path.parent.mkdir(parents=True, exist_ok=True)
        args.single_record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0
    if args.row_isolation:
        records, rollups = run_efficiency_grid_isolated(
            config,
            jax_platform=args.jax_platform,
        )
    else:
        records, rollups = run_efficiency_grid(config)
    report_path, json_path = write_report(
        config=config,
        records=records,
        rollups=rollups,
    )
    print(f"Wrote markdown report: {report_path}")
    if json_path is not None:
        print(f"Wrote JSON records: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
