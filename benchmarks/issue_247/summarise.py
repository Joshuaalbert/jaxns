"""Aggregate issue 247 JSONL measurements into a reviewable Markdown table."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(paths):
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    return records


def _median_iqr(values):
    values = np.asarray(values, dtype=float)
    return (
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def _format_median_iqr(values, digits=3):
    median, lower, upper = _median_iqr(values)
    return f"{median:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def summarise(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["implementation"], record["case"], record["phantoms"])].append(record)

    lines = [
        "| implementation | case | phantoms | n | P/chain | phantom samples median [IQR] | expectation bias | expectation RMSE | expectation failures | MC bias | MC RMSE | mean z | SD z | 2σ coverage | MC failures | core s median [IQR] | result s median [IQR] | MC s median [IQR] | total s median [IQR] | evals median [IQR] | peak MiB | ESS/eval | gate active | Kish active median [IQR] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(groups):
        implementation, case, phantoms = key
        rows = groups[key]
        expectation_error = np.asarray([
            row["log_Z_error"] for row in rows
        ])
        mc_error = np.asarray([row["mc_log_Z_error"] for row in rows])
        z_scores = np.asarray([
            row.get("mc_z_score", np.nan) for row in rows
        ])
        coverage = np.asarray([
            abs(row["mc_log_Z_error"]) <= 2.0 * row["mc_log_Z_std"]
            for row in rows
        ])
        finite = np.asarray([
            np.isfinite(row["log_Z_mean"])
            and np.isfinite(row["mc_log_Z_mean"])
            and np.isfinite(row["mc_log_Z_std"])
            for row in rows
        ])
        expectation_coverage = np.asarray([
            abs(row["log_Z_error"]) <= 3.0 * row["log_Z_uncert"]
            for row in rows
        ])
        expectation_failures = np.sum(~finite | ~expectation_coverage)
        mc_failures = np.sum(~finite | ~coverage)
        ess_per_eval = [
            row["ess"] / max(row["likelihood_evaluations"], 1)
            for row in rows
        ]
        total_time = [
            row["run_s"] + row["result_s"] + row["mc_s"]
            for row in rows
        ]
        lines.append(
            f"| {implementation} | {case} | {phantoms} | {len(rows)} | "
            f"{int(np.median([row['num_retained_phantoms'] for row in rows]))} | "
            f"{_format_median_iqr([row['phantom_samples'] for row in rows], 0)} | "
            f"{np.mean(expectation_error):.4f} | "
            f"{np.sqrt(np.mean(np.square(expectation_error))):.4f} | "
            f"{expectation_failures} | "
            f"{np.mean(mc_error):.4f} | "
            f"{np.sqrt(np.mean(np.square(mc_error))):.4f} | "
            f"{np.mean(z_scores):.2f} | "
            f"{np.std(z_scores):.2f} | "
            f"{np.mean(coverage):.1%} | "
            f"{mc_failures} | "
            f"{_format_median_iqr([row['run_s'] for row in rows])} | "
            f"{_format_median_iqr([row['result_s'] for row in rows])} | "
            f"{_format_median_iqr([row['mc_s'] for row in rows])} | "
            f"{_format_median_iqr(total_time)} | "
            f"{_format_median_iqr([row['likelihood_evaluations'] for row in rows], 0)} | "
            f"{max(row['process_peak_rss_kib'] for row in rows) / 1024.0:.1f} | "
            f"{np.median(ess_per_eval):.3e} | "
            f"{np.median([row.get('phantom_gate_active_fraction', 0.0) for row in rows]):.1%} | "
            f"{_format_median_iqr([row.get('phantom_kish_median_active', 0.0) for row in rows], 1)} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarise(_load(args.inputs))
    if args.output is None:
        print(report, end="")
    else:
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
