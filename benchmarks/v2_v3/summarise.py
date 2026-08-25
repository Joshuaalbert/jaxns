"""Summarise and optionally validate maintained release-benchmark records."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.v2_v3.schema import load_records, validate_release_matrix


def _median_iqr(values: list[float], digits: int = 3) -> str:
    data = np.asarray(values, dtype=float)
    median, lower, upper = np.percentile(data, [50, 25, 75])
    return (
        f"{median:.{digits}f} "
        f"[{lower:.{digits}f}, {upper:.{digits}f}]"
    )


def _optional_median(rows: list[dict], field: str, digits: int = 3) -> str:
    values = [row[field] for row in rows if field in row]
    if not values:
        return "—"
    return _median_iqr(values, digits)


def summarise(records: list[dict]) -> str:
    """Return one reviewable Markdown row per implementation/problem/mode."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        conditioning = record.get(
            "conditioning",
            "phantom" if record["phantoms"] else "classic",
        )
        implementation = {
            "current": "v3",
            "v2-pypi": "v2",
            "main": "v2",
        }.get(record["implementation"], record["implementation"])
        groups[(implementation, record["case"], conditioning)].append(record)

    lines = [
        (
            "| implementation | case | conditioning | n | MC bias | MC RMSE | "
            "reported σ mean | mean z | SD z | 2σ coverage | MC−expectation | "
            "core s median [IQR] | total s median [IQR] | evals median [IQR] | "
            "ESS/eval | posterior mean RMSE [IQR] | mode max error [IQR] | "
            "missed/incorrect modes |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for key in sorted(groups):
        implementation, case, conditioning = key
        rows = groups[key]
        mc_error = np.asarray([row["mc_log_Z_error"] for row in rows])
        z_scores = np.asarray([row["mc_z_score"] for row in rows])
        coverage = np.asarray([
            abs(row["mc_log_Z_error"]) <= 2.0 * row["mc_log_Z_std"]
            for row in rows
        ])
        total_s = [
            row["run_s"] + row["result_s"] + row["mc_s"]
            for row in rows
        ]
        ess_per_eval = [
            row["ess"] / max(row["likelihood_evaluations"], 1)
            for row in rows
        ]
        missed = sum(row.get("posterior_missed_mode_count", 0) for row in rows)
        incorrect = sum(
            row.get("posterior_incorrect_mode_weight_count", 0)
            for row in rows
        )
        lines.append(
            f"| {implementation} | {case} | {conditioning} | {len(rows)} | "
            f"{np.mean(mc_error):.4f} | "
            f"{np.sqrt(np.mean(np.square(mc_error))):.4f} | "
            f"{np.mean([row['mc_log_Z_std'] for row in rows]):.4f} | "
            f"{np.mean(z_scores):.2f} | {np.std(z_scores):.2f} | "
            f"{np.mean(coverage):.1%} | "
            f"{np.mean([row['mc_log_Z_mean'] - row['log_Z_mean'] for row in rows]):.4f} | "
            f"{_median_iqr([row['run_s'] for row in rows])} | "
            f"{_median_iqr(total_s)} | "
            f"{_median_iqr([row['likelihood_evaluations'] for row in rows], 0)} | "
            f"{np.median(ess_per_eval):.3e} | "
            f"{_optional_median(rows, 'posterior_mean_rmse')} | "
            f"{_optional_median(rows, 'posterior_mode_weight_max_abs_error')} | "
            f"{missed}/{incorrect} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release-gate", action="store_true")
    args = parser.parse_args()
    records = load_records(args.inputs)
    if args.release_gate:
        validate_release_matrix(records)
    report = summarise(records)
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
