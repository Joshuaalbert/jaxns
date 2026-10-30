"""Reduce complete, independently seeded direction-law comparisons."""

import csv
import json
from pathlib import Path

import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    groups = (
        ("Curved / isotropic", "curved-isotropic-*.jsonl", 30),
        ("Curved / covariance", "curved-covariance-*.jsonl", 30),
        ("Straight / isotropic", "straight-isotropic.jsonl", 10),
        ("Straight / covariance", "straight-covariance.jsonl", 10),
        ("Curved / core 1e-6", "core-1e-6.jsonl", 3),
        ("Curved / core 1e-10", "core-1e-10.jsonl", 3),
    )
    summaries = []
    records_by_group = {}
    for label, pattern, expected_count in groups:
        records = [
            json.loads(line)
            for path in sorted((directory / "raw").glob(pattern))
            for line in path.read_text().splitlines()
        ]
        records.sort(key=lambda record: record["seed"])
        seeds = [record["seed"] for record in records]
        if seeds != list(range(expected_count)):
            raise ValueError(f"Incomplete or duplicate seeds for {label}: {seeds}")
        if not all(
            record["termination_reason"] == 0 and record["depth_reached"]
            for record in records
        ):
            raise ValueError(f"Hard-stopped run in {label}.")
        records_by_group[label] = records
        values = {
            key: np.asarray([record[key] for record in records])
            for key in (
                "log_Z", "log_Z_uncert", "evaluations", "peak_gap",
                "std_x1", "std_x16", "std_x17", "std_x5",
                "posterior_mean_log_X_error", "geometry_updates",
            )
        }
        error = values["log_Z"]
        summaries.append({
            "group": label,
            "seeds": len(records),
            "mean_log_Z": float(error.mean()),
            "sd_log_Z": float(error.std(ddof=1)),
            "rmse_log_Z": float(np.sqrt(np.mean(error ** 2))),
            "mean_quoted_uncertainty": float(values["log_Z_uncert"].mean()),
            "within_two_quoted_sd": int(np.sum(
                np.abs(error) <= 2 * values["log_Z_uncert"],
            )),
            "mean_evaluations": float(values["evaluations"].mean()),
            "median_peak_gap": float(np.median(values["peak_gap"])),
            **{
                f"median_{key}": float(np.median(values[key]))
                for key in ("std_x1", "std_x16", "std_x17", "std_x5")
            },
            "mean_log_X_error": float(values["posterior_mean_log_X_error"].mean()),
            "mean_geometry_updates": float(values["geometry_updates"].mean()),
        })
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    baseline = records_by_group["Curved / isotropic"]
    covariance = records_by_group["Curved / covariance"]
    first = np.asarray([row["log_Z"] for row in baseline])
    second = np.asarray([row["log_Z"] for row in covariance])
    bootstrap_indices = np.random.default_rng(312).integers(0, 30, (10000, 30))
    rmse_change = (
        np.sqrt(np.mean(second[bootstrap_indices] ** 2, axis=1))
        - np.sqrt(np.mean(first[bootstrap_indices] ** 2, axis=1))
    )
    output = {
        "groups": summaries,
        "paired_bootstrap_95pct_interval_rmse_change": (
            np.quantile(rmse_change, [0.025, 0.975]).tolist()
        ),
        "timing_note": (
            "Scientific cohorts run concurrently. The initial curved cohort "
            "pinned the main thread after backend import; its timings are not "
            "used as single-core performance evidence."
        ),
    }
    (directory / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
