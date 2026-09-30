"""Aggregate issue-246 JSONL evidence by standard problem and method."""

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


def _optional_metric(
        records: list[dict],
        name: str,
        reducer,
) -> float | None:
    values = [record[name] for record in records if record[name] is not None]
    return None if not values else reducer(values)


def _summarise(records: list[dict]) -> dict:
    errors = [record["mc_log_Z_error"] for record in records]
    z_scores = [record["mc_z_score"] for record in records]
    # Seed zero pays compilation for each fresh matrix-cell process. Later
    # seeds reuse the same executable and define steady execution timing.
    steady_records = [
        record for record in records if record["seed"] != records[0]["seed"]
    ]
    if not steady_records:
        steady_records = records
    return {
        "problem": records[0]["case"],
        "direction": records[0]["direction"],
        "phantoms": records[0]["phantoms"],
        "n": len(records),
        "bias_log_Z": _mean(errors),
        "rmse_log_Z": _rmse(errors),
        "mean_reported_sigma": _mean([
            record["mc_log_Z_std"] for record in records
        ]),
        "mean_z": _mean(z_scores),
        "sd_z": (
            statistics.stdev(z_scores) if len(z_scores) > 1 else math.nan
        ),
        "coverage_1sigma": _mean([
            abs(value) <= 1.0 for value in z_scores
        ]),
        "mean_likelihood_evaluations": _mean([
            record["likelihood_evaluations"] for record in records
        ]),
        "median_steady_run_s": statistics.median([
            record["run_s"] for record in steady_records
        ]),
        "median_steady_result_s": statistics.median([
            record["result_s"] for record in steady_records
        ]),
        "median_steady_mc_s": statistics.median([
            record["mc_s"] for record in steady_records
        ]),
        "first_run_s": records[0]["run_s"],
        "mean_classic_samples": _mean([
            record["classic_samples"] for record in records
        ]),
        "mean_phantom_samples": _mean([
            record["phantom_samples"] for record in records
        ]),
        "mean_ess": _mean([record["ess"] for record in records]),
        "mode_mass_bias": _optional_metric(
            records,
            "mode_mass_error",
            _mean,
        ),
        "mode_mass_rmse": _optional_metric(
            records,
            "mode_mass_error",
            _rmse,
        ),
        "mean_fit_updates": _mean([
            record["fit_updates"] for record in records
        ]),
        "mean_isotropic_fraction": _optional_metric(
            records,
            "isotropic_fraction",
            _mean,
        ),
        "peak_rss_mib": max(
            record["process_peak_rss_kib"] for record in records
        ) / 1024.0,
    }


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


def _validate(grouped: dict[tuple, list[dict]]) -> None:
    """Reject incomplete or unmatched evidence before aggregation."""
    by_problem = defaultdict(dict)
    finite_fields = (
        "run_s",
        "result_s",
        "mc_s",
        "log_Z_mean",
        "log_Z_uncert",
        "mc_log_Z_mean",
        "mc_log_Z_std",
        "mc_z_score",
        "likelihood_evaluations",
        "ess",
    )
    for key, records in grouped.items():
        seeds = [record["seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in benchmark cell {key}.")
        for record in records:
            for field in finite_fields:
                if not math.isfinite(record[field]):
                    raise ValueError(f"Non-finite {field} in {key}, seed {record['seed']}.")
        by_problem[key[0]][(key[1], key[2])] = records

    settings = (
        "truth_log_Z",
        "dimension",
        "root_degree",
        "replacement_width",
        "num_slices",
        "dlogZ",
    )
    scientific_outputs = (
        "classic_samples",
        "likelihood_evaluations",
        "log_Z_mean",
        "log_Z_uncert",
        "ess",
        "mode_mass",
        "fit_updates",
        "directions",
        "isotropic_directions",
    )
    for problem, cells in by_problem.items():
        if len(cells) != 4:
            raise ValueError(f"Expected four matrix cells for {problem}.")
        seed_sets = [
            {record["seed"] for record in records}
            for records in cells.values()
        ]
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            raise ValueError(f"Unpaired seeds for {problem}.")
        reference = next(iter(cells.values()))[0]
        for records in cells.values():
            if any(
                record[field] != reference[field]
                for record in records
                for field in settings
            ):
                raise ValueError(f"Unmatched scientific settings for {problem}.")

        # Phantom retention changes evidence conditioning and result storage,
        # but must not alter the classic Markov chain for a fixed direction.
        for direction in ("isotropic", "ellipsoidal"):
            off = {
                record["seed"]: record
                for record in cells[(direction, False)]
            }
            on = {
                record["seed"]: record
                for record in cells[(direction, True)]
            }
            for seed in off:
                for field in scientific_outputs:
                    if off[seed][field] != on[seed][field]:
                        raise ValueError(
                            f"Phantom collection changed {field} for "
                            f"{problem}, {direction}, seed {seed}."
                        )


def _comparison(
        isotropic: dict,
        ellipsoidal: dict,
        isotropic_records: list[dict],
        ellipsoidal_records: list[dict],
) -> dict:
    isotropic_by_seed = {
        record["seed"]: record for record in isotropic_records
    }
    ellipsoidal_by_seed = {
        record["seed"]: record for record in ellipsoidal_records
    }
    seeds = sorted(isotropic_by_seed)
    rng = random.Random(
        f"{isotropic['problem']}:{isotropic['phantoms']}:246"
    )
    rmse_differences = []
    mode_differences = []
    evaluation_changes = []
    for _ in range(5000):
        selected = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        isotropic_errors = [
            isotropic_by_seed[seed]["mc_log_Z_error"] for seed in selected
        ]
        ellipsoidal_errors = [
            ellipsoidal_by_seed[seed]["mc_log_Z_error"] for seed in selected
        ]
        rmse_differences.append(
            _rmse(ellipsoidal_errors) - _rmse(isotropic_errors)
        )
        evaluation_changes.append(100.0 * (
            _mean([
                ellipsoidal_by_seed[seed]["likelihood_evaluations"]
                for seed in selected
            ])
            / _mean([
                isotropic_by_seed[seed]["likelihood_evaluations"]
                for seed in selected
            ])
            - 1.0
        ))
        if isotropic["mode_mass_rmse"] is not None:
            isotropic_mode_errors = [
                isotropic_by_seed[seed]["mode_mass_error"]
                for seed in selected
            ]
            ellipsoidal_mode_errors = [
                ellipsoidal_by_seed[seed]["mode_mass_error"]
                for seed in selected
            ]
            mode_differences.append(
                _rmse(ellipsoidal_mode_errors)
                - _rmse(isotropic_mode_errors)
            )

    def interval(values: list[float]) -> tuple[float, float]:
        values = sorted(values)
        return values[124], values[4874]

    rmse_low, rmse_high = interval(rmse_differences)
    evaluation_low, evaluation_high = interval(evaluation_changes)
    mode_low = None
    mode_high = None
    if mode_differences:
        mode_low, mode_high = interval(mode_differences)
    mode_ratio = None
    if isotropic["mode_mass_rmse"] is not None:
        mode_ratio = (
            ellipsoidal["mode_mass_rmse"]
            / isotropic["mode_mass_rmse"]
        )
    return {
        "problem": isotropic["problem"],
        "phantoms": isotropic["phantoms"],
        "n": isotropic["n"],
        "isotropic_rmse_log_Z": isotropic["rmse_log_Z"],
        "ellipsoidal_rmse_log_Z": ellipsoidal["rmse_log_Z"],
        "rmse_ratio": (
            ellipsoidal["rmse_log_Z"] / isotropic["rmse_log_Z"]
        ),
        "rmse_difference_ci_low": rmse_low,
        "rmse_difference_ci_high": rmse_high,
        "likelihood_evaluation_change_pct": 100.0 * (
            ellipsoidal["mean_likelihood_evaluations"]
            / isotropic["mean_likelihood_evaluations"]
            - 1.0
        ),
        "likelihood_evaluation_change_ci_low": evaluation_low,
        "likelihood_evaluation_change_ci_high": evaluation_high,
        "steady_time_change_pct": 100.0 * (
            ellipsoidal["median_steady_run_s"]
            / isotropic["median_steady_run_s"]
            - 1.0
        ),
        "isotropic_mode_mass_rmse": isotropic["mode_mass_rmse"],
        "ellipsoidal_mode_mass_rmse": ellipsoidal["mode_mass_rmse"],
        "mode_mass_rmse_ratio": mode_ratio,
        "mode_mass_rmse_difference_ci_low": mode_low,
        "mode_mass_rmse_difference_ci_high": mode_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--comparisons-csv", type=Path)
    parser.add_argument("--comparisons-markdown", type=Path)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for input_dir in args.input_dirs:
        for path in sorted(input_dir.glob("*.jsonl")):
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    key = (
                        record["case"],
                        record["direction"],
                        record["phantoms"],
                    )
                    grouped[key].append(record)
    _validate(grouped)
    summaries = [
        _summarise(sorted(records, key=lambda record: record["seed"]))
        for _, records in sorted(grouped.items())
    ]
    if not summaries:
        raise ValueError(
            f"No JSONL benchmark records found in {args.input_dirs}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    columns = [
        "problem",
        "direction",
        "phantoms",
        "n",
        "bias_log_Z",
        "rmse_log_Z",
        "mean_z",
        "sd_z",
        "mean_likelihood_evaluations",
        "median_steady_run_s",
        "median_steady_result_s",
        "median_steady_mc_s",
        "mean_phantom_samples",
        "mode_mass_rmse",
        "mean_fit_updates",
        "mean_isotropic_fraction",
    ]
    with args.markdown.open("w") as stream:
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for summary in summaries:
            stream.write(
                "| "
                + " | ".join(_format(summary[column]) for column in columns)
                + " |\n"
            )

    indexed = {
        (summary["problem"], summary["direction"], summary["phantoms"]): summary
        for summary in summaries
    }
    comparisons = [
        _comparison(
            indexed[(problem, "isotropic", phantoms)],
            indexed[(problem, "ellipsoidal", phantoms)],
            grouped[(problem, "isotropic", phantoms)],
            grouped[(problem, "ellipsoidal", phantoms)],
        )
        for problem in sorted({summary["problem"] for summary in summaries})
        for phantoms in (False, True)
    ]
    if args.comparisons_csv is not None:
        args.comparisons_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.comparisons_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(comparisons[0]),
            )
            writer.writeheader()
            writer.writerows(comparisons)
    if args.comparisons_markdown is not None:
        comparison_columns = [
            "problem",
            "phantoms",
            "n",
            "isotropic_rmse_log_Z",
            "ellipsoidal_rmse_log_Z",
            "likelihood_evaluation_change_pct",
            "steady_time_change_pct",
            "isotropic_mode_mass_rmse",
            "ellipsoidal_mode_mass_rmse",
        ]
        with args.comparisons_markdown.open("w") as stream:
            stream.write("| " + " | ".join(comparison_columns) + " |\n")
            stream.write(
                "| "
                + " | ".join("---" for _ in comparison_columns)
                + " |\n"
            )
            for comparison in comparisons:
                stream.write(
                    "| "
                    + " | ".join(
                        _format(comparison[column])
                        for column in comparison_columns
                    )
                    + " |\n"
                )


if __name__ == "__main__":
    main()
