"""Measure paired-seed accuracy on the unchanged standard basic problem."""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Benchmark helpers live under cicd rather than the installed package. Append
# only the repository root so an explicit PYTHONPATH still selects the JAXNS
# implementation being measured.
sys.path.append(str(REPO_ROOT))

from cicd.tests.test_ns_standard_problems import _basic_model_case
from jaxns.core import NestedSampler


def main() -> None:
    """Print multi-seed classic evidence accuracy and work as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of canonical integer seeds starting from zero.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-seed records after computing the same summary.",
    )
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("At least two seeds are required for calibration.")

    model, truth = _basic_model_case()
    runner = NestedSampler(model)
    records = []
    for seed in range(args.seeds):
        started = time.perf_counter()
        state = runner.run(jax.random.PRNGKey(seed))
        jax.block_until_ready(state)
        run_seconds = time.perf_counter() - started

        started = time.perf_counter()
        result = state.to_result().trim()
        ensemble = result.sample_evidence_mc(
            num_samples=1000,
            conditioning="classic",
            key=jax.random.PRNGKey(290 + seed),
        )
        samples = np.asarray(ensemble.log_Z_samples)  # [M]
        analysis_seconds = time.perf_counter() - started
        records.append({
            "seed": seed,
            "estimate": float(np.mean(samples)),
            "uncertainty": float(np.std(samples)),
            "num_samples": int(state.num_samples),
            "evaluations": int(result.total_num_likelihood_evaluations),
            "run_seconds": run_seconds,
            "analysis_seconds": analysis_seconds,
        })

    errors = np.asarray(
        [row["estimate"] - float(truth) for row in records]
    )  # [R]
    standardised = np.asarray(
        [
            error / row["uncertainty"]
            for error, row in zip(errors, records, strict=True)
        ]
    )  # [R]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    bias_standard_error = float(
        np.std(errors, ddof=1) / np.sqrt(errors.shape[0])
    )
    # Delta-method uncertainty for sqrt(mean(error**2)). This quantifies the
    # finite 30-seed comparison without pretending the RMS itself is exact.
    rmse_standard_error = float(
        np.std(np.square(errors), ddof=1)
        / (2.0 * rmse * np.sqrt(errors.shape[0]))
    )
    summary = {
        "truth": float(truth),
        "bias": float(np.mean(errors)),
        "bias_standard_error": bias_standard_error,
        "rmse": rmse,
        "rmse_standard_error": rmse_standard_error,
        "standardised_mean": float(np.mean(standardised)),
        "standardised_sd": float(np.std(standardised, ddof=1)),
        "samples_mean": float(np.mean([
            row["num_samples"] for row in records
        ])),
        "evaluations_mean": float(np.mean([
            row["evaluations"] for row in records
        ])),
        # Seed zero includes lowering and compilation. Remaining seeds reuse
        # the exact specialization and therefore measure steady execution.
        "cold_run_seconds": records[0]["run_seconds"],
        "warm_run_median_seconds": float(np.median([
            row["run_seconds"] for row in records[1:]
        ])),
        "warm_run_range_seconds": [
            float(np.min([row["run_seconds"] for row in records[1:]])),
            float(np.max([row["run_seconds"] for row in records[1:]])),
        ],
        "analysis_median_seconds": float(np.median([
            row["analysis_seconds"] for row in records
        ])),
    }
    if not args.summary_only:
        summary["records"] = records
    print(json.dumps(summary))


main()
