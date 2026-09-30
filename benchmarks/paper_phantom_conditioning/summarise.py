"""Summarise the paired phantom-prefix sweep and emit LaTeX rows."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

PROBLEM_LABELS = {
    "basic_mvn": "G8",
    "weak_curved_mvn8": "CG8",
    "spike_slab": "SS8",
    "correlated_spike_slab8": "CSS8",
}
PROTOCOL = "phantom_prefix_sweep_10d_case_mc2048"
PHANTOM_MULTIPLIERS = tuple(range(10))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _rms(values: list[float]) -> float:
    return math.sqrt(_mean([value * value for value in values]))


def _bootstrap_rms(
        errors: list[float],
        indices: np.ndarray,
) -> np.ndarray:
    """Calculate RMS for each joint resample of the completed trees."""
    values = np.asarray(errors, dtype=float)
    return np.sqrt(np.mean(np.square(values[indices]), axis=1))


def _load(paths: list[Path]) -> dict[str, list[dict]]:
    records = {}
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["protocol"] != PROTOCOL:
                raise ValueError(f"{path} contains another protocol.")
            records.setdefault(record["case"], []).append(record)
    for case_records in records.values():
        case_records.sort(key=lambda record: record["seed"])
    return records


def _summarise_case(records: list[dict]) -> dict:
    """Calculate calibration metrics for each paired phantom prefix."""
    seeds = [record["seed"] for record in records]
    if seeds != list(range(30)):
        raise ValueError(
            f"{records[0]['case']} must contain seeds 0--29 exactly once; "
            f"got {seeds}."
        )
    settings = (
        "protocol",
        "truth_log_Z",
        "dimension",
        "root_degree",
        "replacement_width",
        "num_slices",
        "maximum_retained_phantoms_per_chain",
        "phantom_multipliers",
        "goal_log_Z_uncert",
        "depth_dlog_Z",
        "mc_draws",
        "mc_batch_size",
    )
    reference = records[0]
    if any(
        record[field] != reference[field]
        for record in records
        for field in settings
    ):
        raise ValueError(
            f"{reference['case']} contains unmatched scientific settings."
        )
    if any(
        not record["achieved_goal_log_Z_uncert"]
        < record["goal_log_Z_uncert"]
        for record in records
    ):
        raise ValueError(f"{reference['case']} contains an early-stopped run.")
    if any(record["mc_draws"] != reference["mc_draws"] for record in records):
        raise ValueError(
            f"{reference['case']} does not use one shared per-state "
            "MC draw count."
        )

    metrics = {}
    truth = reference["truth_log_Z"]
    classic_errors = [
        record["evidence"]["0"]["log_Z_mean"] - truth
        for record in records
    ]
    # Resample the tree index, not individual MC evidence draws. All prefixes
    # use these same indices so uncertainty in their RMS difference preserves
    # the paired experimental design.
    generator = np.random.default_rng(
        10_000 * list(PROBLEM_LABELS).index(reference["case"])
    )
    bootstrap_indices = generator.integers(
        0,
        len(records),
        size=(10_000, len(records)),
    )
    classic_bootstrap_rms = _bootstrap_rms(
        classic_errors,
        bootstrap_indices,
    )
    for multiplier in PHANTOM_MULTIPLIERS:
        evidence = [record["evidence"][str(multiplier)] for record in records]
        errors = [item["log_Z_mean"] - truth for item in evidence]
        # Pool within-tree variances. The stratified draw counts sum to the
        # requested case-level MC budget, while every tree informs calibration.
        uncertainty = math.sqrt(_mean([
            item["log_Z_uncert"] ** 2 for item in evidence
        ]))
        rms = _rms(errors)
        bootstrap_rms = _bootstrap_rms(errors, bootstrap_indices)
        metrics[str(multiplier)] = {
            "bias": _mean(errors),
            "rms": rms,
            # This is the sampling standard error of the 30-tree RMS
            # estimator, not the within-tree nested-sampling uncertainty.
            "rms_standard_error": float(np.std(bootstrap_rms, ddof=1)),
            # A calibrated MC shrinkage model has empirical error on the same
            # scale as its reported uncertainty, and hence Z_bias near one.
            "Z_bias": rms / uncertainty,
            "mean_mc_log_Z_uncert": uncertainty,
            "mean_gate_active_fraction": _mean([
                item["gate_active_fraction"] for item in evidence
            ]),
        }
        if multiplier:
            bootstrap_difference = bootstrap_rms - classic_bootstrap_rms
            lower, upper = np.quantile(
                bootstrap_difference,
                [0.025, 0.975],
            )
            metrics[str(multiplier)]["rms_difference_from_classic"] = (
                rms - _rms(classic_errors)
            )
            metrics[str(multiplier)][
                "rms_difference_standard_error"
            ] = float(np.std(bootstrap_difference, ddof=1))
            metrics[str(multiplier)]["rms_difference_interval_95"] = (
                float(lower),
                float(upper),
            )

    best_multiplier = min(
        range(1, 10),
        key=lambda multiplier: metrics[str(multiplier)]["rms"],
    )
    best_overall_multiplier = min(
        PHANTOM_MULTIPLIERS,
        key=lambda multiplier: metrics[str(multiplier)]["rms"],
    )
    best_bias_multiplier = min(
        PHANTOM_MULTIPLIERS,
        key=lambda multiplier: abs(metrics[str(multiplier)]["bias"]),
    )
    mean_evaluations = _mean([
        record["likelihood_evaluations"] for record in records
    ])
    mean_classic_samples = _mean([
        record["classic_samples"] for record in records
    ])
    mean_valid_clusters = _mean([
        record["valid_phantom_clusters"] for record in records
    ])
    mean_retained_phantom_samples = _mean([
        record["maximum_phantom_samples"] for record in records
    ])

    output = {
        "case": reference["case"],
        "problem_label": PROBLEM_LABELS[reference["case"]],
        "seeds": len(records),
        "dimension": reference["dimension"],
        "num_slices": reference["num_slices"],
        "mc_draws": reference["mc_draws"],
        "best_phantom_multiplier": best_multiplier,
        "best_overall_multiplier": best_overall_multiplier,
        "best_absolute_bias_multiplier": best_bias_multiplier,
        "mean_likelihood_evaluations": mean_evaluations,
        "mean_classic_samples": mean_classic_samples,
        "mean_valid_phantom_clusters": mean_valid_clusters,
        "mean_retained_phantom_samples": mean_retained_phantom_samples,
        "metrics": metrics,
    }

    mode_mass = [
        record["mode_mass"]
        for record in records
        if record["mode_mass"] is not None
    ]
    if mode_mass:
        mode_truth = reference["mode_mass_truth"]
        mode_errors = [value - mode_truth for value in mode_mass]
        classic_errors = [
            record["evidence"]["0"]["log_Z_mean"] - truth
            for record in records
        ]
        best_errors = [
            record["evidence"][str(best_multiplier)]["log_Z_mean"] - truth
            for record in records
        ]
        output["mode_diagnostic"] = {
            "truth": mode_truth,
            "mean": _mean(mode_mass),
            "rms_error": _rms(mode_errors),
            "classic_evidence_correlation": statistics.correlation(
                classic_errors,
                mode_errors,
            ),
            "best_phantom_evidence_correlation": statistics.correlation(
                best_errors,
                mode_errors,
            ),
        }
    return output


def _format(value: float, *, signed: bool, bold: bool) -> str:
    formatted = f"{value:+.4f}" if signed else f"{value:.4f}"
    return f"\\textbf{{{formatted}}}" if bold else formatted


def _print_latex(summary: dict) -> None:
    label = summary["problem_label"]
    best = summary["best_phantom_multiplier"]
    best_bias = summary["best_absolute_bias_multiplier"]
    best_rms = summary["best_overall_multiplier"]
    print(
        f"{summary['case']}: best retained prefix {best}D; "
        f"mean likelihood evaluations "
        f"{summary['mean_likelihood_evaluations']:,.0f}"
    )
    selected = summary["metrics"][str(best)]
    interval = selected["rms_difference_interval_95"]
    print(
        f"{summary['case']}: selected-prefix RMS difference "
        f"{selected['rms_difference_from_classic']:+.4f} +/- "
        f"{selected['rms_difference_standard_error']:.4f}, paired 95% "
        f"interval [{interval[0]:+.4f}, {interval[1]:+.4f}]"
    )
    for multiplier in PHANTOM_MULTIPLIERS:
        metric = summary["metrics"][str(multiplier)]
        conditioning = "Classic" if multiplier == 0 else f"{multiplier}D"
        if multiplier == best:
            # Mark the exploratory phantom-only selection separately from
            # bold metric winners, which may still belong to classic NS.
            conditioning = f"{conditioning}$^{{\\dagger}}$"
        print(
            f"{label} & {conditioning} & "
            f"{_format(metric['bias'], signed=True, bold=multiplier == best_bias)} & "
            f"{_format(metric['rms'], signed=False, bold=multiplier == best_rms)} "
            f"$\\pm$ {_format(metric['rms_standard_error'], signed=False, bold=False)} & "
            f"{_format(metric['Z_bias'], signed=False, bold=False)} & "
            f"{summary['mean_likelihood_evaluations']:,.0f} \\\\"
        )
    mode = summary.get("mode_diagnostic")
    if mode is not None:
        print(
            f"{summary['case']}: mode mass truth {mode['truth']:.4f}, "
            f"mean {mode['mean']:.4f}, RMS error "
            f"{mode['rms_error']:.4f}, classic correlation "
            f"{mode['classic_evidence_correlation']:+.3f}, selected-prefix "
            f"correlation {mode['best_phantom_evidence_correlation']:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    grouped = _load(args.inputs)
    if set(grouped) != set(PROBLEM_LABELS):
        raise ValueError(
            f"Expected paper problems {tuple(PROBLEM_LABELS)}, got "
            f"{tuple(grouped)}."
        )

    summaries = []
    for case in PROBLEM_LABELS:
        records = grouped[case]
        seeds = [record["seed"] for record in records]
        if len(records) != 30 or seeds != list(range(30)):
            raise ValueError(f"{case} does not contain paired seeds 0--29.")
        summary = _summarise_case(records)
        summaries.append(summary)
        _print_latex(summary)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
