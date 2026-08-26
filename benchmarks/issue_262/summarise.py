"""Aggregate matched streaming-direction evidence by standard problem."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _rmse(values: list[float]) -> float:
    return math.sqrt(_mean([value * value for value in values]))


def _optional(
        records: list[dict],
        name: str,
        reducer,
) -> float | None:
    values = [record[name] for record in records if record[name] is not None]
    return None if not values else reducer(values)


def _summarise(records: list[dict]) -> dict:
    errors = [record["mc_log_Z_error"] for record in records]
    classic_errors = [record["log_Z_error"] for record in records]
    z_scores = [record["mc_z_score"] for record in records]
    # The first seed in every fresh cell pays compilation. Later seeds reuse
    # the same executable and define full-run steady timing.
    steady = records[1:] if len(records) > 1 else records
    return {
        "problem": records[0]["case"],
        "direction": records[0]["direction"],
        "phantoms": records[0]["phantoms"],
        "n": len(records),
        "bias_log_Z": _mean(errors),
        "rmse_log_Z": _rmse(errors),
        "classic_bias_log_Z": _mean(classic_errors),
        "classic_rmse_log_Z": _rmse(classic_errors),
        "mean_reported_sigma": _mean([
            record["mc_log_Z_std"] for record in records
        ]),
        "mean_z": _mean(z_scores),
        "sd_z": statistics.stdev(z_scores),
        "coverage_1sigma": _mean([abs(value) <= 1.0 for value in z_scores]),
        "mean_likelihood_evaluations": _mean([
            record["likelihood_evaluations"] for record in records
        ]),
        "median_steady_run_s": statistics.median([
            record["run_s"] for record in steady
        ]),
        "mean_ess": _mean([record["ess"] for record in records]),
        "posterior_mean_rmse": _optional(
            records,
            "posterior_mean_error",
            _rmse,
        ),
        "mode_mass_bias": _optional(records, "mode_mass_error", _mean),
        "mode_mass_rmse": _optional(records, "mode_mass_error", _rmse),
        "mean_updates": _mean([record["fit_updates"] for record in records]),
        "mean_isotropic_fraction": _optional(
            records,
            "isotropic_fraction",
            _mean,
        ),
        "peak_rss_mib": max(
            record["process_peak_rss_kib"] for record in records
        ) / 1024.0,
    }


def _interval(
        values: list[float],
        alpha: float = 0.05,
) -> tuple[float, float]:
    ordered = sorted(values)
    lower = max(0, math.floor(0.5 * alpha * len(ordered)))
    upper = min(
        len(ordered) - 1,
        math.ceil((1.0 - 0.5 * alpha) * len(ordered)) - 1,
    )
    return ordered[lower], ordered[upper]


def _comparison(
        candidate: dict,
        reference: dict,
        candidate_records: list[dict],
        reference_records: list[dict],
) -> dict:
    candidate_by_seed = {
        record["seed"]: record for record in candidate_records
    }
    reference_by_seed = {
        record["seed"]: record for record in reference_records
    }
    seeds = sorted(candidate_by_seed)
    rng = random.Random(
        f"{candidate['problem']}:{candidate['phantoms']}:"
        f"{reference['direction']}:262"
    )
    bias_differences = []
    rmse_differences = []
    classic_bias_differences = []
    classic_rmse_differences = []
    evaluation_changes = []
    posterior_differences = []
    mode_differences = []
    for _ in range(5000):
        selected = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        candidate_errors = [
            candidate_by_seed[seed]["mc_log_Z_error"] for seed in selected
        ]
        reference_errors = [
            reference_by_seed[seed]["mc_log_Z_error"] for seed in selected
        ]
        candidate_classic_errors = [
            candidate_by_seed[seed]["log_Z_error"] for seed in selected
        ]
        reference_classic_errors = [
            reference_by_seed[seed]["log_Z_error"] for seed in selected
        ]
        bias_differences.append(
            _mean(candidate_errors) - _mean(reference_errors)
        )
        rmse_differences.append(
            _rmse(candidate_errors) - _rmse(reference_errors)
        )
        classic_bias_differences.append(
            _mean(candidate_classic_errors) - _mean(reference_classic_errors)
        )
        classic_rmse_differences.append(
            _rmse(candidate_classic_errors) - _rmse(reference_classic_errors)
        )
        evaluation_changes.append(100.0 * (
            _mean([
                candidate_by_seed[seed]["likelihood_evaluations"]
                for seed in selected
            ])
            / _mean([
                reference_by_seed[seed]["likelihood_evaluations"]
                for seed in selected
            ])
            - 1.0
        ))
        if candidate["posterior_mean_rmse"] is not None:
            posterior_differences.append(
                _rmse([
                    candidate_by_seed[seed]["posterior_mean_error"]
                    for seed in selected
                ])
                - _rmse([
                    reference_by_seed[seed]["posterior_mean_error"]
                    for seed in selected
                ])
            )
        if candidate["mode_mass_rmse"] is not None:
            mode_differences.append(
                _rmse([
                    candidate_by_seed[seed]["mode_mass_error"]
                    for seed in selected
                ])
                - _rmse([
                    reference_by_seed[seed]["mode_mass_error"]
                    for seed in selected
                ])
            )

    bias_low, bias_high = _interval(bias_differences)
    rmse_low, rmse_high = _interval(rmse_differences)
    classic_bias_low, classic_bias_high = _interval(
        classic_bias_differences
    )
    classic_rmse_low, classic_rmse_high = _interval(
        classic_rmse_differences
    )
    # There are twenty streaming-versus-reference rows per reference family.
    # Bonferroni intervals keep one isolated nominal 95% miss from being
    # presented as a family-level bias or RMSE discovery.
    family_alpha = 0.05 / 20.0
    classic_bias_family_low, classic_bias_family_high = _interval(
        classic_bias_differences,
        family_alpha,
    )
    classic_rmse_family_low, classic_rmse_family_high = _interval(
        classic_rmse_differences,
        family_alpha,
    )
    evaluation_low, evaluation_high = _interval(evaluation_changes)
    posterior_low = posterior_high = None
    if posterior_differences:
        posterior_low, posterior_high = _interval(posterior_differences)
    mode_low = mode_high = None
    if mode_differences:
        mode_low, mode_high = _interval(mode_differences)
    return {
        "problem": candidate["problem"],
        "phantoms": candidate["phantoms"],
        "reference": reference["direction"],
        "n": candidate["n"],
        "bias_difference": (
            candidate["bias_log_Z"] - reference["bias_log_Z"]
        ),
        "bias_difference_ci_low": bias_low,
        "bias_difference_ci_high": bias_high,
        "rmse_ratio": candidate["rmse_log_Z"] / reference["rmse_log_Z"],
        "rmse_difference_ci_low": rmse_low,
        "rmse_difference_ci_high": rmse_high,
        "classic_bias_difference": (
            candidate["classic_bias_log_Z"]
            - reference["classic_bias_log_Z"]
        ),
        "classic_bias_difference_ci_low": classic_bias_low,
        "classic_bias_difference_ci_high": classic_bias_high,
        "classic_bias_difference_family_ci_low": classic_bias_family_low,
        "classic_bias_difference_family_ci_high": classic_bias_family_high,
        "classic_rmse_ratio": (
            candidate["classic_rmse_log_Z"]
            / reference["classic_rmse_log_Z"]
        ),
        "classic_rmse_difference_ci_low": classic_rmse_low,
        "classic_rmse_difference_ci_high": classic_rmse_high,
        "classic_rmse_difference_family_ci_low": classic_rmse_family_low,
        "classic_rmse_difference_family_ci_high": classic_rmse_family_high,
        "likelihood_evaluation_change_pct": 100.0 * (
            candidate["mean_likelihood_evaluations"]
            / reference["mean_likelihood_evaluations"]
            - 1.0
        ),
        "likelihood_evaluation_change_ci_low": evaluation_low,
        "likelihood_evaluation_change_ci_high": evaluation_high,
        "steady_time_change_pct": 100.0 * (
            candidate["median_steady_run_s"]
            / reference["median_steady_run_s"]
            - 1.0
        ),
        "evidence_cost_ratio": (
            candidate["rmse_log_Z"] ** 2
            * candidate["mean_likelihood_evaluations"]
            / (
                reference["rmse_log_Z"] ** 2
                * reference["mean_likelihood_evaluations"]
            )
        ),
        "mean_z_difference": candidate["mean_z"] - reference["mean_z"],
        "sd_z_difference": candidate["sd_z"] - reference["sd_z"],
        "ess_change_pct": 100.0 * (
            candidate["mean_ess"] / reference["mean_ess"] - 1.0
        ),
        "posterior_mean_rmse_ratio": (
            None
            if candidate["posterior_mean_rmse"] is None
            else candidate["posterior_mean_rmse"]
            / reference["posterior_mean_rmse"]
        ),
        "posterior_mean_rmse_difference_ci_low": posterior_low,
        "posterior_mean_rmse_difference_ci_high": posterior_high,
        "mode_mass_rmse_ratio": (
            None
            if candidate["mode_mass_rmse"] is None
            else candidate["mode_mass_rmse"]
            / reference["mode_mass_rmse"]
        ),
        "mode_mass_rmse_difference_ci_low": mode_low,
        "mode_mass_rmse_difference_ci_high": mode_high,
        "peak_rss_change_mib": (
            candidate["peak_rss_mib"] - reference["peak_rss_mib"]
        ),
    }


def _validate(grouped: dict[tuple, list[dict]]) -> None:
    """Reject incomplete, unpaired, or scientifically unmatched cells."""
    expected_directions = {"isotropic", "ellipsoidal", "streaming"}
    by_problem = defaultdict(dict)
    for key, records in grouped.items():
        seeds = [record["seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in {key}.")
        if len(records) < 30:
            raise ValueError(f"Need at least 30 seeds in {key}.")
        for record in records:
            for field in (
                "run_s",
                "log_Z_error",
                "mc_log_Z_error",
                "mc_log_Z_std",
                "mc_z_score",
                "likelihood_evaluations",
                "ess",
            ):
                if not math.isfinite(record[field]):
                    raise ValueError(f"Non-finite {field} in {key}.")
        by_problem[key[0]][(key[1], key[2])] = records

    settings = (
        "truth_log_Z",
        "dimension",
        "root_degree",
        "replacement_width",
        "num_slices",
        "dlogZ",
    )
    for problem, cells in by_problem.items():
        expected = {
            (direction, phantoms)
            for direction in expected_directions
            for phantoms in (False, True)
        }
        if set(cells) != expected:
            raise ValueError(f"Incomplete policy matrix for {problem}.")
        seed_sets = [
            {record["seed"] for record in records}
            for records in cells.values()
        ]
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            raise ValueError(f"Unpaired seeds for {problem}.")
        reference = next(iter(cells.values()))[0]
        for records in cells.values():
            for record in records:
                if any(record[field] != reference[field] for field in settings):
                    raise ValueError(f"Unmatched settings for {problem}.")

        # Retaining chain prefixes must not alter classic generation for a
        # fixed direction policy. This also guards benchmark bookkeeping.
        for direction in expected_directions:
            off = {record["seed"]: record for record in cells[(direction, False)]}
            on = {record["seed"]: record for record in cells[(direction, True)]}
            for seed in off:
                for field in (
                    "classic_samples",
                    "likelihood_evaluations",
                    "log_Z_mean",
                    "log_Z_uncert",
                    "ess",
                    "mode_mass",
                    "posterior_mean_error",
                    "fit_updates",
                ):
                    if off[seed][field] != on[seed][field]:
                        raise ValueError(
                            f"Phantoms changed {field} for {problem}, "
                            f"{direction}, seed {seed}."
                        )


def _format(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}"


def _write_table(path: Path, records: list[dict], columns: list[str]) -> None:
    with path.open("w") as stream:
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for record in records:
            stream.write(
                "| "
                + " | ".join(_format(record[column]) for column in columns)
                + " |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path, required=True)
    parser.add_argument("--comparisons-markdown", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in sorted(args.input_dir.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                grouped[(
                    record["case"],
                    record["direction"],
                    record["phantoms"],
                )].append(record)
    _validate(grouped)
    summaries = [
        _summarise(sorted(records, key=lambda record: record["seed"]))
        for _, records in sorted(grouped.items())
    ]
    indexed = {
        (record["problem"], record["direction"], record["phantoms"]): record
        for record in summaries
    }
    comparisons = [
        _comparison(
            indexed[(problem, "streaming", phantoms)],
            indexed[(problem, reference, phantoms)],
            grouped[(problem, "streaming", phantoms)],
            grouped[(problem, reference, phantoms)],
        )
        for problem in sorted({record["problem"] for record in summaries})
        for phantoms in (False, True)
        for reference in ("ellipsoidal", "isotropic")
    ]

    for path, records in (
        (args.csv, summaries),
        (args.comparisons_csv, comparisons),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    _write_table(
        args.markdown,
        summaries,
        [
            "problem",
            "direction",
            "phantoms",
            "n",
            "bias_log_Z",
            "rmse_log_Z",
            "classic_bias_log_Z",
            "classic_rmse_log_Z",
            "mean_z",
            "sd_z",
            "mean_likelihood_evaluations",
            "median_steady_run_s",
            "posterior_mean_rmse",
            "mode_mass_rmse",
            "mean_updates",
            "peak_rss_mib",
        ],
    )
    _write_table(
        args.comparisons_markdown,
        comparisons,
        [
            "problem",
            "phantoms",
            "reference",
            "n",
            "bias_difference",
            "bias_difference_ci_low",
            "bias_difference_ci_high",
            "rmse_ratio",
            "classic_bias_difference",
            "classic_bias_difference_family_ci_low",
            "classic_bias_difference_family_ci_high",
            "classic_rmse_ratio",
            "likelihood_evaluation_change_pct",
            "steady_time_change_pct",
            "evidence_cost_ratio",
            "posterior_mean_rmse_ratio",
            "mode_mass_rmse_ratio",
        ],
    )


if __name__ == "__main__":
    main()
