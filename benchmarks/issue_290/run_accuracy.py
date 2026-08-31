"""Measure paired-seed accuracy on the unchanged standard basic problem."""

import json

import jax
import numpy as np

from cicd.tests.test_ns_standard_problems import _basic_model_case
from jaxns.core import NestedSampler


def main() -> None:
    """Print ten-seed classic evidence accuracy and work as JSON."""
    model, truth = _basic_model_case()
    runner = NestedSampler(model)
    records = []
    for seed in range(10):
        state = runner.run(jax.random.PRNGKey(seed))
        result = state.to_result().trim()
        ensemble = result.sample_evidence_mc(
            num_samples=1000,
            conditioning="classic",
            key=jax.random.PRNGKey(290 + seed),
        )
        samples = np.asarray(ensemble.log_Z_samples)  # [M]
        records.append({
            "seed": seed,
            "estimate": float(np.mean(samples)),
            "uncertainty": float(np.std(samples)),
            "num_samples": int(state.num_samples),
            "evaluations": int(result.total_num_likelihood_evaluations),
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
    print(json.dumps({
        "truth": float(truth),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "standardised_mean": float(np.mean(standardised)),
        "standardised_sd": float(np.std(standardised, ddof=1)),
        "samples_mean": float(np.mean([
            row["num_samples"] for row in records
        ])),
        "evaluations_mean": float(np.mean([
            row["evaluations"] for row in records
        ])),
        "records": records,
    }))


main()
