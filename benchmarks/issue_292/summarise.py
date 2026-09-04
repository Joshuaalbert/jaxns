"""Validate and summarise paired issue-292 JSONL evidence."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean(records: list[dict], field: str) -> float:
    return statistics.fmean(record[field] for record in records)


def _rmse(records: list[dict], field: str) -> float:
    return math.sqrt(statistics.fmean(
        record[field] ** 2 for record in records
    ))


def _optional(records: list[dict], field: str, reducer):
    values = [record[field] for record in records if record[field] is not None]
    return None if not values else reducer(values)


def _summary(records: list[dict]) -> dict:
    """Aggregate one feature arm without hiding its sample count."""
    return {
        "case": records[0]["case"],
        "phantom_seeding": records[0]["phantom_seeding"],
        "n": len(records),
        "log_Z_bias": _mean(records, "log_Z_error"),
        "log_Z_rmse": _rmse(records, "log_Z_error"),
        "mean_log_Z_uncert": _mean(records, "log_Z_uncert"),
        "mean_log_Z_z": _mean(records, "log_Z_z"),
        "variance_log_Z_z": (
            None
            if len(records) < 2
            else statistics.variance(
                record["log_Z_z"] for record in records
            )
        ),
        "mode_mass_bias": _optional(
            records,
            "mode_mass_error",
            statistics.fmean,
        ),
        "mode_mass_rmse": _optional(
            records,
            "mode_mass_error",
            lambda values: math.sqrt(statistics.fmean(
                value * value for value in values
            )),
        ),
        "mode_loss_rate": _optional(
            records,
            "mode_lost",
            statistics.fmean,
        ),
        "mean_likelihood_evaluations": _mean(
            records,
            "likelihood_evaluations",
        ),
        "mean_classic_samples": _mean(records, "classic_samples"),
        "mean_goal_loops": _mean(records, "goal_loops"),
        "median_warm_wall_s": statistics.median(
            record["warm_wall_s"] for record in records
        ),
        "mean_state_mib": _mean(records, "state_bytes") / 2 ** 20,
        "retained_phantom_capacity": records[0][
            "retained_phantom_capacity"
        ],
        "mean_retained_phantom_samples": _mean(
            records,
            "retained_phantom_samples",
        ),
        "pool_capacity": _mean(records, "pool_capacity"),
        "mean_pool_active": _mean(records, "pool_active"),
        "mean_pool_staging": _mean(records, "pool_staging"),
        "mean_pool_active_fraction": _mean(
            records,
            "pool_active_fraction",
        ),
    }


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _paired_comparison(
        case: str,
        off_records: list[dict],
        on_records: list[dict],
        bootstrap_samples: int,
) -> dict:
    """Return on-minus-off effects with paired seed-bootstrap intervals."""
    rng = np.random.default_rng(292)
    count = len(off_records)
    indices = rng.integers(
        0,
        count,
        size=(bootstrap_samples, count),
    )  # [B, N]

    def values(records: list[dict], field: str) -> np.ndarray:
        return np.asarray([record[field] for record in records], dtype=float)

    off_error = values(off_records, "log_Z_error")
    on_error = values(on_records, "log_Z_error")
    off_rmse = np.sqrt(np.mean(np.square(off_error[indices]), axis=1))
    on_rmse = np.sqrt(np.mean(np.square(on_error[indices]), axis=1))
    bias_difference = np.mean(
        on_error[indices] - off_error[indices],
        axis=1,
    )

    def mean_percent_change(field: str) -> np.ndarray:
        off = values(off_records, field)[indices]
        on = values(on_records, field)[indices]
        return 100.0 * (np.mean(on, axis=1) / np.mean(off, axis=1) - 1.0)

    off_wall = values(off_records, "warm_wall_s")[indices]
    on_wall = values(on_records, "warm_wall_s")[indices]
    wall_change = 100.0 * (
        np.median(on_wall, axis=1) / np.median(off_wall, axis=1) - 1.0
    )
    comparison = {
        "case": case,
        "n": count,
        "log_Z_bias_difference": float(np.mean(on_error - off_error)),
        "log_Z_bias_difference_ci": _interval(bias_difference),
        "log_Z_rmse_difference": float(
            np.sqrt(np.mean(np.square(on_error)))
            - np.sqrt(np.mean(np.square(off_error)))
        ),
        "log_Z_rmse_difference_ci": _interval(on_rmse - off_rmse),
        "likelihood_evaluation_change_pct": float(
            100.0 * (
                np.mean(values(on_records, "likelihood_evaluations"))
                / np.mean(values(off_records, "likelihood_evaluations"))
                - 1.0
            )
        ),
        "likelihood_evaluation_change_pct_ci": _interval(
            mean_percent_change("likelihood_evaluations")
        ),
        "classic_sample_change_pct": float(
            100.0 * (
                np.mean(values(on_records, "classic_samples"))
                / np.mean(values(off_records, "classic_samples"))
                - 1.0
            )
        ),
        "classic_sample_change_pct_ci": _interval(
            mean_percent_change("classic_samples")
        ),
        "goal_loop_difference": float(np.mean(
            values(on_records, "goal_loops")
            - values(off_records, "goal_loops")
        )),
        "goal_loop_difference_ci": _interval(np.mean(
            values(on_records, "goal_loops")[indices]
            - values(off_records, "goal_loops")[indices],
            axis=1,
        )),
        "median_warm_wall_change_pct": float(
            100.0 * (
                np.median(values(on_records, "warm_wall_s"))
                / np.median(values(off_records, "warm_wall_s"))
                - 1.0
            )
        ),
        "median_warm_wall_change_pct_ci": _interval(wall_change),
        "state_size_change_pct": float(
            100.0 * (
                np.mean(values(on_records, "state_bytes"))
                / np.mean(values(off_records, "state_bytes"))
                - 1.0
            )
        ),
        "state_size_change_pct_ci": _interval(
            mean_percent_change("state_bytes")
        ),
    }
    if off_records[0]["mode_mass_error"] is not None:
        off_mode = values(off_records, "mode_mass_error")
        on_mode = values(on_records, "mode_mass_error")
        off_mode_rmse = np.sqrt(np.mean(np.square(off_mode[indices]), axis=1))
        on_mode_rmse = np.sqrt(np.mean(np.square(on_mode[indices]), axis=1))
        off_lost = values(off_records, "mode_lost")
        on_lost = values(on_records, "mode_lost")
        comparison.update({
            "mode_mass_rmse_difference": float(
                np.sqrt(np.mean(np.square(on_mode)))
                - np.sqrt(np.mean(np.square(off_mode)))
            ),
            "mode_mass_rmse_difference_ci": _interval(
                on_mode_rmse - off_mode_rmse
            ),
            "mode_loss_rate_difference": float(np.mean(on_lost - off_lost)),
            "mode_loss_rate_difference_ci": _interval(np.mean(
                on_lost[indices] - off_lost[indices],
                axis=1,
            )),
        })
    return comparison


def _validate(grouped: dict[tuple[str, bool], list[dict]]) -> None:
    """Reject incomplete, unpaired, or scientifically unmatched inputs."""
    finite_fields = (
        "log_Z_mean",
        "log_Z_uncert",
        "log_Z_error",
        "log_Z_z",
        "likelihood_evaluations",
        "classic_samples",
        "goal_loops",
        "warm_wall_s",
        "state_bytes",
    )
    cases = {key[0] for key in grouped}
    settings = (
        "truth_log_Z",
        "dimension",
        "root_degree",
        "replacement_width",
        "num_slices",
        "configured_phantom_capacity",
        "retained_phantom_capacity",
        "dlogZ",
        "mc_draws",
        "source_id",
    )
    for case in cases:
        if (case, False) not in grouped or (case, True) not in grouped:
            raise ValueError(f"Expected both phantom-seeding arms for {case}.")
        off = sorted(grouped[(case, False)], key=lambda record: record["seed"])
        on = sorted(grouped[(case, True)], key=lambda record: record["seed"])
        off_seeds = [record["seed"] for record in off]
        on_seeds = [record["seed"] for record in on]
        if len(off_seeds) != len(set(off_seeds)):
            raise ValueError(f"Duplicate disabled-arm seeds for {case}.")
        if off_seeds != on_seeds:
            raise ValueError(f"Unpaired seed sets for {case}.")
        reference = off[0]
        for record in off + on:
            if any(record[field] != reference[field] for field in settings):
                raise ValueError(f"Unmatched scientific settings for {case}.")
            for field in finite_fields:
                if not math.isfinite(record[field]):
                    raise ValueError(
                        f"Non-finite {field} for {case}, seed "
                        f"{record['seed']}."
                    )
        if any(record["pool_capacity"] != 0 for record in off):
            raise ValueError(f"Disabled arm unexpectedly owns a pool for {case}.")
        if any(record["pool_capacity"] <= 0 for record in on):
            raise ValueError(f"Enabled arm has no seed pool for {case}.")


def _format(value) -> str:
    if value is None:
        return "--"
    if type(value) is bool:
        return "on" if value else "off"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        return value
    return f"{value:.5g}"


def _format_effect(comparison: dict, field: str) -> str:
    """Format one estimate with its paired 95% bootstrap interval."""
    if field not in comparison:
        return "--"
    interval = comparison[f"{field}_ci"]
    return (
        f"{_format(comparison[field])} "
        f"[{_format(interval[0])}, {_format(interval[1])}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")

    grouped = defaultdict(list)
    for input_dir in args.input_dirs:
        for path in sorted(input_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    grouped[(record["case"], record["phantom_seeding"])].append(
                        record
                    )
    if not grouped:
        raise ValueError(f"No JSONL records found in {args.input_dirs}.")
    _validate(grouped)

    summaries = []
    comparisons = []
    for case in sorted({key[0] for key in grouped}):
        off = sorted(grouped[(case, False)], key=lambda record: record["seed"])
        on = sorted(grouped[(case, True)], key=lambda record: record["seed"])
        summaries.extend([_summary(off), _summary(on)])
        comparisons.append(_paired_comparison(
            case,
            off,
            on,
            args.bootstrap_samples,
        ))

    report = {
        "bootstrap_samples": args.bootstrap_samples,
        "summaries": summaries,
        "paired_comparisons": comparisons,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        "case",
        "phantom_seeding",
        "n",
        "log_Z_bias",
        "log_Z_rmse",
        "variance_log_Z_z",
        "mode_mass_rmse",
        "mode_loss_rate",
        "mean_likelihood_evaluations",
        "mean_classic_samples",
        "mean_goal_loops",
        "median_warm_wall_s",
        "mean_state_mib",
        "retained_phantom_capacity",
        "mean_retained_phantom_samples",
        "pool_capacity",
        "mean_pool_active",
        "mean_pool_staging",
    )
    with args.markdown.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for summary in summaries:
            stream.write(
                "| "
                + " | ".join(_format(summary[column]) for column in columns)
                + " |\n"
            )
        stream.write("\nPaired effects are `on - off`; brackets give paired "
                     "seed-bootstrap 95% intervals.\n\n")
        comparison_columns = (
            "case",
            "log_Z_rmse_difference",
            "likelihood_evaluation_change_pct",
            "mode_mass_rmse_difference",
            "mode_loss_rate_difference",
            "median_warm_wall_change_pct",
            "state_size_change_pct",
        )
        stream.write("| " + " | ".join(comparison_columns) + " |\n")
        stream.write(
            "| " + " | ".join("---" for _ in comparison_columns) + " |\n"
        )
        for comparison in comparisons:
            cells = [comparison["case"]]
            cells.extend(
                _format_effect(comparison, field)
                for field in comparison_columns[1:]
            )
            stream.write("| " + " | ".join(cells) + " |\n")


if __name__ == "__main__":
    main()
