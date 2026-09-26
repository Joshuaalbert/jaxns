"""Summarise the state-backed isotropic phantom-prefix experiments."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = "phantom_prefix_sweep_10d_state_mc2048_staged_gmm_shift3"
CASE_LABELS = {
    "basic_mvn": "G8",
    "weak_curved_mvn8": "CG8",
    "spike_slab": "SS8",
    "correlated_spike_slab8": "CSS8",
}
# The paper isolates phantom conditioning by holding the direction law fixed.
# GMM directions remain an implementation feature, not an experimental arm.
DIRECTIONS = ("isotropic",)
PREFIXES = tuple(range(10))
BOOTSTRAP_DRAWS = 10_000


def _validate_posterior_schema(records: list[dict]) -> None:
    """Reject prefix-local posterior diagnostics from historical outputs."""
    if any(
        "mode_mass_mean" in prefix
        for record in records
        for prefix in record["evidence"].values()
    ):
        raise ValueError(
            "Mode mass must be stored once per completed classic tree, not "
            "inside phantom-prefix evidence records."
        )


def _load(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """Load unique complete records grouped by problem and direction."""
    grouped = {}
    for path in paths:
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Incomplete JSON in {path} at line {line_number}."
                ) from error
            if record["protocol"] != PROTOCOL:
                raise ValueError(f"{path} contains another protocol.")
            key = (record["case"], record["direction"])
            grouped.setdefault(key, []).append(record)

    for (case, direction), records in grouped.items():
        records.sort(key=lambda record: record["seed"])
        seeds = [record["seed"] for record in records]
        if seeds != list(range(30)):
            raise ValueError(
                f"{case}/{direction} must contain seeds 0--29 exactly once; "
                f"got {seeds}."
            )
    return grouped


def _bootstrap_indices(case: str, num_records: int) -> np.ndarray:
    """Use one seed-index resampling field for every phantom prefix."""
    generator = np.random.default_rng(
        BOOTSTRAP_DRAWS * list(CASE_LABELS).index(case)
    )
    return generator.integers(
        0,
        num_records,
        size=(BOOTSTRAP_DRAWS, num_records),
    )


def _summarise_arm(
        records: list[dict],
        indices: np.ndarray,
) -> dict:
    """Compute accuracy, calibration, and work dispersion for one arm."""
    reference = records[0]
    settings = (
        "protocol",
        "case",
        "direction",
        "truth_log_Z",
        "dimension",
        "root_degree",
        "replacement_width",
        "num_slices",
        "maximum_retained_phantoms_per_chain",
        "goal_log_Z_uncert",
        "depth_dlog_Z",
        "mc_draws",
        "mc_batch_size",
    )
    if any(
        record[field] != reference[field]
        for record in records
        for field in settings
    ):
        raise ValueError(
            f"{reference['case']}/{reference['direction']} has unmatched "
            "scientific settings."
        )
    if any(
        record["achieved_goal_log_Z_uncert"]
        >= record["goal_log_Z_uncert"]
        for record in records
    ):
        raise ValueError(
            f"{reference['case']}/{reference['direction']} stopped early."
        )
    if any(record["mc_draws"] != reference["mc_draws"] for record in records):
        raise ValueError(
            f"{reference['case']}/{reference['direction']} has the wrong "
            "per-state MC draw count."
        )
    _validate_posterior_schema(records)

    evaluations = np.asarray([
        record["likelihood_evaluations"] for record in records
    ], dtype=np.float64)
    run_seconds = np.asarray([
        record["run_seconds"] for record in records
    ], dtype=np.float64)
    metrics = {}
    errors_by_prefix = {}
    rms_bootstrap_by_prefix = {}
    truth = float(reference["truth_log_Z"])
    for prefix in PREFIXES:
        evidence = [record["evidence"][str(prefix)] for record in records]
        errors = np.asarray([
            item["log_Z_mean"] - truth for item in evidence
        ])
        uncertainties = np.asarray([
            item["log_Z_uncert"] for item in evidence
        ])
        if np.any(uncertainties <= 0.0):
            raise ValueError("Every MC log-evidence uncertainty must be positive.")
        bias = float(np.mean(errors))
        error_sd = float(np.std(errors, ddof=1))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        uncertainty = float(np.sqrt(np.mean(np.square(uncertainties))))
        standardized_errors = errors / uncertainties
        rms_bootstrap = np.sqrt(np.mean(
            np.square(errors[indices]),
            axis=1,
        ))
        uncertainty_bootstrap = np.sqrt(np.mean(
            np.square(uncertainties[indices]),
            axis=1,
        ))
        ratio_bootstrap = uncertainty_bootstrap / rms_bootstrap
        calibration_variance_bootstrap = np.var(
            standardized_errors[indices],
            axis=1,
            ddof=1,
        )
        errors_by_prefix[prefix] = errors
        rms_bootstrap_by_prefix[prefix] = rms_bootstrap
        metrics[str(prefix)] = {
            "bias": bias,
            "error_standard_deviation": error_sd,
            "rms": rms,
            "rms_bootstrap_standard_deviation": float(
                np.std(rms_bootstrap, ddof=1)
            ),
            "mc_log_Z_uncert": uncertainty,
            "uncertainty_over_rms": uncertainty / rms,
            "uncertainty_over_rms_bootstrap_standard_deviation": float(
                np.std(ratio_bootstrap, ddof=1)
            ),
            "standardized_error_mean": float(np.mean(standardized_errors)),
            "standardized_error_variance": float(np.var(
                standardized_errors,
                ddof=1,
            )),
            "standardized_error_variance_bootstrap_standard_deviation": float(
                np.std(calibration_variance_bootstrap, ddof=1)
            ),
            "standardized_error_variance_bootstrap_interval_95": [
                float(np.quantile(calibration_variance_bootstrap, 0.025)),
                float(np.quantile(calibration_variance_bootstrap, 0.975)),
            ],
            "mean_gate_active_fraction": float(np.mean([
                item["gate_active_fraction"] for item in evidence
            ])),
        }
        if reference["mode_mass_truth"] is not None:
            mode_mass = np.asarray([
                record["mode_mass"] for record in records
            ], dtype=np.float64)
            mode_errors = mode_mass - float(reference["mode_mass_truth"])
            mode_rms_bootstrap = np.sqrt(np.mean(
                np.square(mode_errors[indices]),
                axis=1,
            ))
            metrics[str(prefix)].update({
                "mode_mass_mean": float(np.mean(mode_mass)),
                "mode_mass_rms": float(np.sqrt(np.mean(
                    np.square(mode_errors)
                ))),
                "mode_mass_rms_bootstrap_standard_deviation": float(
                    np.std(mode_rms_bootstrap, ddof=1)
                ),
            })

    # Selection is descriptive: all nine prefixes were inspected on the same
    # trees. Keeping the paired bootstrap difference makes that post-selection
    # comparison explicit and reproducible without implying a held-out test.
    selected = min(
        PREFIXES[1:],
        key=lambda prefix: metrics[str(prefix)]["rms"],
    )
    rms_difference_bootstrap = (
        rms_bootstrap_by_prefix[selected] - rms_bootstrap_by_prefix[0]
    )
    selected_vs_classic = {
        "selected_prefix": selected,
        "rms_difference": (
            metrics[str(selected)]["rms"] - metrics["0"]["rms"]
        ),
        "relative_rms_change": (
            metrics[str(selected)]["rms"] / metrics["0"]["rms"] - 1.0
        ),
        "rms_difference_bootstrap_standard_deviation": float(
            np.std(rms_difference_bootstrap, ddof=1)
        ),
        "rms_difference_bootstrap_interval_95": [
            float(np.quantile(rms_difference_bootstrap, 0.025)),
            float(np.quantile(rms_difference_bootstrap, 0.975)),
        ],
    }

    mode_diagnostic = None
    if reference["mode_mass_truth"] is not None:
        mode_truth = float(reference["mode_mass_truth"])
        classic_mode_mass = np.asarray([
            record["mode_mass"]
            for record in records
        ], dtype=np.float64)
        classic_mode_errors = classic_mode_mass - mode_truth
        mode_diagnostic = {
            "truth": mode_truth,
            "classic_mean": float(np.mean(classic_mode_mass)),
            "classic_rms_error": float(np.sqrt(np.mean(
                np.square(classic_mode_errors)
            ))),
            "classic_log_Z_error_correlation": float(np.corrcoef(
                classic_mode_errors,
                errors_by_prefix[0],
            )[0, 1]),
        }

    return {
        "case": reference["case"],
        "direction": reference["direction"],
        "seeds": len(records),
        "mean_likelihood_evaluations": float(np.mean(evaluations)),
        "likelihood_evaluations_standard_deviation": float(
            np.std(evaluations, ddof=1)
        ),
        "mean_run_seconds": float(np.mean(run_seconds)),
        "run_seconds_standard_deviation": float(
            np.std(run_seconds, ddof=1)
        ),
        "total_state_bytes": sum(
            record["state_checkpoint"]["state_bytes"]
            for record in records
        ),
        "mean_retained_phantom_samples": float(np.mean([
            record["maximum_phantom_samples"] for record in records
        ])),
        "mean_phantom_clusters": float(np.mean([
            record["valid_phantom_clusters"] for record in records
        ])),
        "selected_vs_classic": selected_vs_classic,
        "mode_diagnostic": mode_diagnostic,
        "metrics": metrics,
    }


def _format_measurement(
        measurement: float,
        uncertainty: float,
        signed: bool = False,
) -> str:
    """Round an uncertainty first and align its measurement's precision."""
    if not math.isfinite(uncertainty) or uncertainty <= 0.0:
        raise ValueError("A reported uncertainty must be positive and finite.")

    exponent = math.floor(math.log10(uncertainty))
    first_digit = int(uncertainty / 10.0 ** exponent)
    significant_figures = 2 if first_digit in (1, 2) else 1

    # Round before formatting because carrying can change the exponent and
    # therefore the decimal place shared by the measurement and uncertainty.
    initial_places = significant_figures - 1 - exponent
    rounded_uncertainty = round(uncertainty, initial_places)
    rounded_exponent = math.floor(math.log10(rounded_uncertainty))
    decimal_places = max(
        0,
        significant_figures - 1 - rounded_exponent,
    )
    rounded_measurement = round(measurement, decimal_places)

    # Do not let a small negative estimate render as ``-0.00`` after rounding.
    if rounded_measurement == 0.0:
        measurement_text = f"{0.0:.{decimal_places}f}"
    elif signed:
        measurement_text = f"{rounded_measurement:+.{decimal_places}f}"
    else:
        measurement_text = f"{rounded_measurement:.{decimal_places}f}"
    uncertainty_text = f"{rounded_uncertainty:.{decimal_places}f}"
    return f"{measurement_text} $\\pm$ {uncertainty_text}"


def _print_latex(summary: dict[str, dict]) -> None:
    """Emit each problem table with its minimum-RMSE prefix bolded."""
    for case, case_label in CASE_LABELS.items():
        arm = summary[f"{case}/isotropic"]
        # Select using the displayed RMSE precision. If rows are visually
        # tied, retain the first prefix so classic is not displaced by an
        # empirically negligible difference hidden by the reported precision.
        displayed_rms = {
            prefix: float(_format_measurement(
                arm["metrics"][str(prefix)]["rms"],
                arm["metrics"][str(prefix)][
                    "rms_bootstrap_standard_deviation"
                ],
            ).split(" ", maxsplit=1)[0])
            for prefix in PREFIXES
        }
        best_prefix = min(PREFIXES, key=displayed_rms.get)
        print(f"% {case_label}")
        for prefix in PREFIXES:
            if prefix == 0:
                prefix_label = "$0D$ (classic)"
            else:
                prefix_label = f"${prefix}D$"
            metric = arm["metrics"][str(prefix)]
            bias = _format_measurement(
                metric["bias"],
                metric["error_standard_deviation"],
                signed=True,
            )
            rms = _format_measurement(
                metric["rms"],
                metric["rms_bootstrap_standard_deviation"],
            )
            fields = [prefix_label, bias, rms]
            if case in ("spike_slab", "correlated_spike_slab8"):
                fields.append(_format_measurement(
                    metric["mode_mass_rms"],
                    metric[
                        "mode_mass_rms_bootstrap_standard_deviation"
                    ],
                ))
            if prefix == best_prefix:
                fields = [f"\\textbf{{{field}}}" for field in fields]
                # The prefix is entirely mathematical, so text bolding does
                # not change its glyphs. Bold it explicitly in math mode.
                if prefix == 0:
                    fields[0] = (
                        "$\\boldsymbol{0D}$ \\textbf{(classic)}"
                    )
                else:
                    fields[0] = f"$\\boldsymbol{{{prefix}D}}$"
            print(" & ".join(fields) + " \\\\")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    records = _load(args.paths)
    expected = {
        (case, direction)
        for case in CASE_LABELS
        for direction in DIRECTIONS
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(f"Direction suite mismatch; missing={missing}, extra={extra}.")

    summary = {}
    for case in CASE_LABELS:
        indices = _bootstrap_indices(case, 30)
        for direction in DIRECTIONS:
            key = f"{case}/{direction}"
            summary[key] = _summarise_arm(
                records[(case, direction)],
                indices,
            )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_latex(summary)


if __name__ == "__main__":
    main()
