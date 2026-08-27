"""Summarise matched standard-problem distributed benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def median_iqr(values, digits: int = 3) -> str:
    """Format a median and interquartile interval."""
    values = np.asarray(values, dtype=float)
    lower, median, upper = np.percentile(values, [25, 50, 75])
    if digits == 0:
        return f"{median:,.0f} [{lower:,.0f}, {upper:,.0f}]"
    return (
        f"{median:.{digits}f} "
        f"[{lower:.{digits}f}, {upper:.{digits}f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    records = document["records"]
    lines = [
        (
            "| Problem | Runner | n | Expectation bias | SE bias | "
            "Expectation RMS | MC bias | MC RMS | Mean MC z | SD MC z | "
            "MC coverage | Mode-mass bias | Mode-mass RMS | "
            "Core run s median [IQR] | End-to-end s median [IQR] | "
            "Likelihood evals median [IQR] | ESS median [IQR] | "
            "Likelihood evals / ESS median [IQR] | GMM updates median [IQR] | "
            "Isotropic directions |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for runner in ("local", "distributed"):
        matching = [record for record in records if record["runner"] == runner]
        if not matching:
            continue
        evidence_errors = np.asarray([
            record["log_Z_error"] for record in matching
        ])
        mode_errors = np.asarray([
            record["mode_mass_error"] for record in matching
            if record["mode_mass_error"] is not None
        ])
        mc_errors = np.asarray([
            record.get("mc_log_Z_error", record["log_Z_error"])
            for record in matching
        ])
        z_scores = np.asarray([
            record.get("mc_log_Z_error", record["log_Z_error"])
            / record.get("mc_log_Z_uncert", record["log_Z_uncert"])
            for record in matching
        ])
        directions = np.sum([
            record["directions"] for record in matching
        ])
        isotropic = np.sum([
            record["isotropic_directions"] for record in matching
        ])
        if mode_errors.size:
            mode_bias = f"{np.mean(mode_errors):+.5f}"
            mode_rms = f"{np.sqrt(np.mean(np.square(mode_errors))):.5f}"
        else:
            mode_bias = "n/a"
            mode_rms = "n/a"
        phantoms = "on" if document["collect_phantoms"] else "off"
        wall = median_iqr([record["elapsed_s"] for record in matching])
        end_to_end = median_iqr([
            record["elapsed_s"] + record["result_s"] + record["mc_s"]
            for record in matching
        ])
        evaluations = median_iqr(
            [record["likelihood_evaluations"] for record in matching],
            0,
        )
        ess = median_iqr([record["ess"] for record in matching], 1)
        evaluations_per_ess = median_iqr([
            record["likelihood_evaluations"] / record["ess"]
            for record in matching
        ], 1)
        updates = median_iqr(
            [record["gmm_updates"] for record in matching],
            1,
        )
        lines.append(
            f"| {document['case']} (phantoms {phantoms}) | {runner} | "
            f"{len(matching)} | "
            f"{np.mean(evidence_errors):+.5f} | "
            f"{np.std(evidence_errors, ddof=1) / np.sqrt(len(matching)):.5f} | "
            f"{np.sqrt(np.mean(np.square(evidence_errors))):.5f} | "
            f"{np.mean(mc_errors):+.5f} | "
            f"{np.sqrt(np.mean(np.square(mc_errors))):.5f} | "
            f"{np.mean(z_scores):+.2f} | {np.std(z_scores, ddof=1):.2f} | "
            f"{np.mean(np.abs(z_scores) <= 2.0):.1%} | "
            f"{mode_bias} | {mode_rms} | "
            f"{wall} | {end_to_end} | {evaluations} | {ess} | "
            f"{evaluations_per_ess} | "
            f"{updates} | "
            f"{isotropic / max(directions, 1):.3f} |"
        )
    text = "\n".join(lines) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
