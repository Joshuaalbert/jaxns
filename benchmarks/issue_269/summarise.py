"""Summarise the paired allocation-cadence screen for issue 269."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _median_iqr(values: np.ndarray) -> str:
    low, median, high = np.percentile(values, [25, 50, 75])
    return f"{median:.3g} [{low:.3g}, {high:.3g}]"


def _row_summary(records: list[dict[str, object]]) -> dict[str, object]:
    evidence_error = np.asarray([
        record["log_Z_error"] for record in records
    ], dtype=float)
    uncertainty = np.asarray([
        record["log_Z_uncert"] for record in records
    ], dtype=float)
    z_score = evidence_error / uncertainty
    mode_error = np.asarray([
        record["mode_mass_error"] for record in records
    ], dtype=float)
    evaluations = np.asarray([
        record["likelihood_evaluations"] for record in records
    ], dtype=float)
    ess = np.asarray([record["ess"] for record in records], dtype=float)
    return {
        "seeds": len(records),
        "evidence_bias": float(np.mean(evidence_error)),
        "evidence_rms": _rms(evidence_error),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "z_mean": float(np.mean(z_score)),
        "z_sd": (
            float(np.std(z_score, ddof=1))
            if len(z_score) > 1
            else 0.0
        ),
        "coverage_1sigma": float(np.mean(np.abs(z_score) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z_score) <= 2.0)),
        "mode_bias": float(np.mean(mode_error)),
        "mode_rms": _rms(mode_error),
        # A mass below 1e-3 is less than 1% of the analytic 0.1163 weak-mode
        # mass and coincides with runs whose best weak-mode likelihood remains
        # tens of log units below its peak.
        "mode_loss_rate": float(np.mean(np.asarray([
            record["mode_mass"] < 1e-3 for record in records
        ]))),
        "samples": _median_iqr(np.asarray([
            record["samples"] for record in records
        ], dtype=float)),
        "evaluations": _median_iqr(evaluations),
        "ess": _median_iqr(ess),
        "evaluations_per_ess": _median_iqr(evaluations / ess),
        "run_s": _median_iqr(np.asarray([
            record["run_s"] for record in records
        ], dtype=float)),
        "goal_iterations": _median_iqr(np.asarray([
            record["goal_loop_iterations"] for record in records
        ], dtype=float)),
        "goal_success_rate": float(np.mean(np.asarray([
            record["goal_reached"] for record in records
        ], dtype=float))),
        "goal_register_uncertainty": _median_iqr(np.asarray([
            record["goal_register_log_Z_uncert"] for record in records
        ], dtype=float)),
        "final_root_degree": _median_iqr(np.asarray([
            record["final_root_out_degree"] for record in records
        ], dtype=float)),
    }


def _paired_bootstrap(
        baseline: list[dict[str, object]],
        candidate: list[dict[str, object]],
        draws: int = 20_000,
) -> dict[str, object]:
    """Estimate paired relative changes without normal-error assumptions."""
    baseline_by_seed = {int(record["seed"]): record for record in baseline}
    candidate_by_seed = {int(record["seed"]): record for record in candidate}
    seeds = sorted(baseline_by_seed.keys() & candidate_by_seed.keys())
    if len(seeds) < 2:
        raise ValueError("Paired bootstrap needs at least two shared seeds.")

    def values(records_by_seed, field: str) -> np.ndarray:
        return np.asarray([
            records_by_seed[seed][field] for seed in seeds
        ], dtype=float)

    baseline_mode = values(baseline_by_seed, "mode_mass_error")
    candidate_mode = values(candidate_by_seed, "mode_mass_error")
    baseline_loss = values(baseline_by_seed, "mode_mass") < 1e-3
    candidate_loss = values(candidate_by_seed, "mode_mass") < 1e-3
    baseline_evidence = values(baseline_by_seed, "log_Z_error")
    candidate_evidence = values(candidate_by_seed, "log_Z_error")
    baseline_efficiency = values(
        baseline_by_seed,
        "likelihood_evaluations_per_ess",
    )
    candidate_efficiency = values(
        candidate_by_seed,
        "likelihood_evaluations_per_ess",
    )
    rng = np.random.default_rng(269)
    indices = rng.integers(0, len(seeds), size=(draws, len(seeds)))

    def rms_rows(values_: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(values_[indices]), axis=1))

    mode_improvement = (
        1.0 - rms_rows(candidate_mode) / rms_rows(baseline_mode)
    )
    mode_loss_improvement = (
        np.mean(baseline_loss[indices], axis=1)
        - np.mean(candidate_loss[indices], axis=1)
    )
    evidence_regression = (
        rms_rows(candidate_evidence) / rms_rows(baseline_evidence) - 1.0
    )
    efficiency_regression = (
        np.median(candidate_efficiency[indices], axis=1)
        / np.median(baseline_efficiency[indices], axis=1)
        - 1.0
    )

    def interval(samples: np.ndarray) -> dict[str, float]:
        low, median, high = np.percentile(samples, [2.5, 50, 97.5])
        return {
            "median": float(median),
            "low": float(low),
            "high": float(high),
        }

    return {
        "paired_seeds": len(seeds),
        "mode_loss_improvement": interval(mode_loss_improvement),
        "mode_rms_improvement": interval(mode_improvement),
        "evidence_rms_regression": interval(evidence_regression),
        "evaluations_per_ess_regression": interval(efficiency_regression),
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records_by_schedule: dict[str, list[dict[str, object]]] = {}
    for record in payload["records"]:
        records_by_schedule.setdefault(record["schedule"], []).append(record)
    if "baseline" not in records_by_schedule:
        raise ValueError("A baseline row is required for paired comparisons.")

    configured_schedules = [
        schedule["name"]
        for schedule in payload["scientific_config"]["schedules"]
    ]
    summary = {
        schedule: _row_summary(records_by_schedule[schedule])
        for schedule in configured_schedules
        if schedule in records_by_schedule
    }
    comparisons = {
        schedule: _paired_bootstrap(
            records_by_schedule["baseline"],
            records_by_schedule[schedule],
        )
        for schedule in configured_schedules
        if schedule != "baseline" and schedule in records_by_schedule
    }

    lines = [
        "# Issue 269 allocation-cadence screen",
        "",
        (
            "| Schedule | N | Goal hit | Mode loss | Mode RMS | "
            "Evidence RMS | z SD | Evaluations / ESS | Goal iterations | "
            "Final roots | Run seconds |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]
    for schedule in configured_schedules:
        if schedule not in summary:
            continue
        row = summary[schedule]
        lines.append(
            f"| `{schedule}` | {row['seeds']} | "
            f"{_percent(row['goal_success_rate'])} | "
            f"{_percent(row['mode_loss_rate'])} | "
            f"{row['mode_rms']:.4f} | "
            f"{row['evidence_rms']:.4f} | {row['z_sd']:.2f} | "
            f"{row['evaluations_per_ess']} | {row['goal_iterations']} | "
            f"{row['final_root_degree']} | "
            f"{row['run_s']} |"
        )

    lines.extend([
        "",
        "## Paired bootstrap versus baseline",
        "",
        (
            "Intervals are paired-seed 95% percentile intervals. Positive "
            "mode values are improvements; positive evidence and efficiency "
            "values are regressions."
        ),
        "",
        "Mode-loss improvement is an absolute percentage-point reduction.",
        "",
        (
            "| Schedule | Mode-loss improvement | Mode RMS improvement | "
            "Evidence RMS regression | Evaluations / ESS regression |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for schedule in configured_schedules:
        if schedule not in comparisons:
            continue
        comparison = comparisons[schedule]

        def render(interval: dict[str, float]) -> str:
            return (
                f"{_percent(interval['median'])} "
                f"[{_percent(interval['low'])}, "
                f"{_percent(interval['high'])}]"
            )

        lines.append(
            f"| `{schedule}` | "
            f"{render(comparison['mode_loss_improvement'])} | "
            f"{render(comparison['mode_rms_improvement'])} | "
            f"{render(comparison['evidence_rms_regression'])} | "
            f"{render(comparison['evaluations_per_ess_regression'])} |"
        )

    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output is not None:
        json_output = args.output.with_suffix(".summary.json")
        raw_input = args.input.resolve()
        if args.output.resolve() == raw_input:
            raise ValueError("Markdown output would overwrite the raw input.")
        if json_output.resolve() == raw_input:
            raise ValueError(
                "Summary JSON output would overwrite the raw input."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        json_output.write_text(
            json.dumps(
                {"summary": summary, "comparisons": comparisons},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
