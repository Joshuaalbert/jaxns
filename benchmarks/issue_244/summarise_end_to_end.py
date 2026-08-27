"""Summarise issue-244 end-to-end records against the develop baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_by_key = {}
    for path in args.candidate:
        records = json.loads(path.read_text(encoding="utf-8"))["records"]
        for record in records:
            key = (
                record["case"],
                record["phantoms"],
                record["seed"],
            )
            candidate_by_key[key] = record
    candidate = list(candidate_by_key.values())
    baseline_records = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )["records"]
    paired = {}
    if args.paired is not None:
        paired_payload = json.loads(
            args.paired.read_text(encoding="utf-8")
        )
        paired_case = paired_payload["case"]
        for phantoms in (False, True):
            paired[paired_case, phantoms] = [
                record
                for record in paired_payload["records"]
                if record["phantoms"] == phantoms
                and record["seed"] != 0
            ]
    groups = sorted({
        (record["case"], record["phantoms"])
        for record in candidate
    })
    lines = [
        "# Issue 244 end-to-end standard problems",
        "",
        (
            "Each row contains 30 independent seeds. Timings exclude seed "
            "zero's first compilation. The z-score is `(estimate - truth) / "
            "reported uncertainty`; calibrated repetitions should have mean "
            "near zero and standard deviation near one. When a paired timing "
            "file is supplied, its alternating same-process measurements "
            "replace the corresponding cross-process timing row."
        ),
        "",
        "## Accuracy and calibration",
        "",
        (
            "| problem | phantoms | n | develop bias | candidate bias | "
            "develop RMSE | candidate RMSE | develop mean z | candidate "
            "mean z | develop SD z | candidate SD z |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    exact_fields = (
        "log_Z_mean",
        "log_Z_uncert",
        "ess",
        "classic_samples",
        "phantom_samples",
        "depth_loop_iterations",
    )
    performance_rows = []
    for case, phantoms in groups:
        selected = [
            record
            for record in candidate
            if record["case"] == case
            and record["phantoms"] == phantoms
        ]
        baseline = [
            record
            for record in baseline_records
            if record["case"] == case
            and record["phantoms"] == phantoms
        ]
        baseline_by_seed = {
            record["seed"]: record for record in baseline
        }
        exact = sum(
            all(
                record[field] == baseline_by_seed[record["seed"]][field]
                for field in exact_fields
            )
            for record in selected
        )
        baseline_errors = np.asarray([
            record["log_Z_error"] for record in baseline
        ])
        candidate_errors = np.asarray([
            record["log_Z_error"] for record in selected
        ])
        baseline_z = np.asarray([
            record["log_Z_error"] / record["log_Z_uncert"]
            for record in baseline
        ])
        candidate_z = np.asarray([
            record["log_Z_error"] / record["log_Z_uncert"]
            for record in selected
        ])
        baseline_times = [
            float(record["core_s"])
            for record in baseline
            if record["seed"] != 0
        ]
        candidate_times = [
            float(record["core_s"])
            for record in selected
            if record["seed"] != 0
        ]
        baseline_time = statistics.median(baseline_times)
        candidate_time = statistics.median(candidate_times)
        paired_records = paired.get((case, phantoms), [])
        if paired_records:
            baseline_time = statistics.median(
                record["current_s"] for record in paired_records
            )
            candidate_time = statistics.median(
                record["candidate_s"] for record in paired_records
            )
        lines.append(
            f"| {case} | {phantoms} | {len(selected)} | "
            f"{np.mean(baseline_errors):+.5f} | "
            f"{np.mean(candidate_errors):+.5f} | "
            f"{np.sqrt(np.mean(np.square(baseline_errors))):.5f} | "
            f"{np.sqrt(np.mean(np.square(candidate_errors))):.5f} | "
            f"{np.mean(baseline_z):+.3f} | {np.mean(candidate_z):+.3f} | "
            f"{np.std(baseline_z, ddof=1):.3f} | "
            f"{np.std(candidate_z, ddof=1):.3f} |"
        )
        baseline_ess = statistics.median(
            record["ess"] for record in baseline
        )
        candidate_ess = statistics.median(
            record["ess"] for record in selected
        )
        performance_rows.append(
            f"| {case} | {phantoms} | {len(selected)} | "
            f"{exact}/{len(selected)} | {baseline_ess:.1f} | "
            f"{candidate_ess:.1f} | {baseline_time:.4f} | "
            f"{candidate_time:.4f} | "
            f"{candidate_time / baseline_time:.3f} |"
        )
    lines.extend([
        "",
        "## Performance and deterministic accounting",
        "",
        (
            "`Exact` counts seeds whose evidence estimate, uncertainty, ESS, "
            "classic/phantom sample counts, and depth-loop count match "
            "`develop` bitwise. The separate mechanism benchmark measures "
            "physical likelihood work exactly."
        ),
        "",
        (
            "| problem | phantoms | n | exact | develop ESS | candidate ESS "
            "| develop core s | candidate core s | wall ratio |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *performance_rows,
    ])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
