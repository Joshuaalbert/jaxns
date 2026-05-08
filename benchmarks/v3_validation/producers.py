"""Small producer helpers for v3 validation benchmark records."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from benchmarks.v3_validation.schema_checks import (
    REQUIRED_PERFORMANCE_GUARDRAIL_NAMES,
    compute_calibration_summary,
)


def produce_multi_seed_evidence_calibration(
        runs: Sequence[Mapping[str, Any]],
        *,
        logZ_ref: float,
) -> list[dict[str, Any]]:
    records = []
    for run in runs:
        samples = np.asarray(run["log_Z_samples"], dtype=float)
        summary = compute_calibration_summary(samples, logZ_ref=logZ_ref)
        calibration = {
            **summary,
            "rmse_logZ": float(np.sqrt(np.mean(np.square(samples - logZ_ref)))),
            "reported_uncertainty_logZ": float(
                run["reported_uncertainty_logZ"]
            ),
            "empirical_uncertainty_logZ": float(
                run["empirical_uncertainty_logZ"]
            ),
            "expectation_logZ": float(run["expectation_logZ"]),
            "mc_shrinkage_logZ": float(run["mc_shrinkage_logZ"]),
        }
        for key in ("rho_g", "rho_fit"):
            if key in run:
                calibration[key] = [
                    float(value)
                    for value in run[key]
                ]
        records.append(
            {
                "metric_family": "evidence_calibration",
                "metadata": copy.deepcopy(run["metadata"]),
                "evidence_calibration": calibration,
            }
        )
    return records


def produce_multi_seed_calibration_rollups(
        calibration_table: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any], dict[str, Any]] = {}
    order: list[tuple[str, Any]] = []
    for record in calibration_table:
        metadata = record["metadata"]
        method_setting = metadata["method_setting"]
        key = (
            str(metadata["problem"]),
            _method_setting_key(method_setting),
        )
        if key not in groups:
            groups[key] = {
                "problem": str(metadata["problem"]),
                "method_setting": copy.deepcopy(method_setting),
                "z_logZ": [],
            }
            order.append(key)
        groups[key]["z_logZ"].append(
            float(record["evidence_calibration"]["z_logZ"])
        )

    rollups = []
    for key in order:
        group = groups[key]
        z_scores = np.asarray(group["z_logZ"], dtype=float)
        method_setting = group["method_setting"]
        rollups.append(
            {
                "problem": group["problem"],
                "method": str(method_setting["method"]),
                "method_setting": method_setting,
                "num_seeds": int(z_scores.size),
                "mean_z_logZ": float(np.mean(z_scores)),
                "sd_z_logZ": float(np.std(z_scores, ddof=1)),
            }
        )
    return rollups


def produce_rmse_vs_likelihood_pareto(
        calibration_records: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_records_match_metadata(
        calibration_records,
        metadata,
        context="rmse_vs_likelihood",
    )
    return _produce_rmse_vs_likelihood_pareto_for_group(
        calibration_records,
        metadata=metadata,
    )


def produce_grouped_rmse_vs_likelihood_pareto(
        calibration_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _produce_rmse_vs_likelihood_pareto_for_group(
            rows,
            metadata=_aggregate_rmse_group_metadata(rows),
        )
        for _, rows in _group_records_by_problem_method(
            calibration_records,
            context="rmse_vs_likelihood",
        )
    ]


def _produce_rmse_vs_likelihood_pareto_for_group(
        calibration_records: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
) -> dict[str, Any]:
    grouped_rows = _group_records_by_likelihood_evaluations(calibration_records)
    likelihood_evaluations = []
    rmse_logZ = []
    num_seeds = []
    for evaluations, rows in grouped_rows:
        seed_count = _assert_independent_seed_records(
            rows,
            context=f"likelihood_evaluations={evaluations}",
        )
        biases = np.asarray([
            _calibration_bias(row["evidence_calibration"])
            for row in rows
        ], dtype=float)
        likelihood_evaluations.append(int(evaluations))
        rmse_logZ.append(float(np.sqrt(np.mean(np.square(biases)))))
        num_seeds.append(seed_count)
    mse_cost = [
        rmse ** 2 * evaluations
        for rmse, evaluations in zip(rmse_logZ, likelihood_evaluations, strict=True)
    ]

    best_rmse_so_far = np.inf
    pareto_efficient = []
    for rmse in rmse_logZ:
        is_efficient = rmse < best_rmse_so_far
        pareto_efficient.append(bool(is_efficient))
        best_rmse_so_far = min(best_rmse_so_far, rmse)

    best_index = int(np.argmin(np.asarray(mse_cost, dtype=float)))
    dominance_summary = {
        "best_index": best_index,
        "best_likelihood_evaluations": likelihood_evaluations[best_index],
        "best_mse_times_likelihood_evaluations": mse_cost[best_index],
        "dominated_indices": [
            idx
            for idx, value in enumerate(mse_cost)
            if idx != best_index and value > mse_cost[best_index]
        ],
    }
    return {
        "metric_family": "rmse_vs_likelihood",
        "metadata": copy.deepcopy(metadata),
        "rmse_vs_likelihood": {
            "likelihood_evaluations": likelihood_evaluations,
            "num_seeds": num_seeds,
            "rmse_logZ": rmse_logZ,
            "mse_times_likelihood_evaluations": mse_cost,
            "pareto_efficient": pareto_efficient,
            "dominance_summary": dominance_summary,
        },
    }


def produce_posterior_quality(
        runs: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_records_match_metadata(
        runs,
        metadata,
        context="posterior_quality",
    )
    seed_count = _assert_independent_seed_records(runs, context="posterior_quality")
    aggregate_metadata = _aggregate_metadata(metadata, runs)
    effective_sample_size = float(sum(
        float(run["effective_sample_size"])
        for run in runs
    ))
    return {
        "metric_family": "posterior_quality",
        "metadata": aggregate_metadata,
        "posterior_quality": {
            "num_seeds": seed_count,
            "wasserstein_mc": float(np.mean([
                float(run["wasserstein_mc"])
                for run in runs
            ])),
            "reference_sample_count": int(sum(
                int(run["reference_sample_count"])
                for run in runs
            )),
            "posterior_sample_count": int(sum(
                int(run["posterior_sample_count"])
                for run in runs
            )),
            "effective_sample_size": effective_sample_size,
            "likelihood_evaluations_per_effective_sample": (
                _likelihood_evaluations_per_ess(
                    aggregate_metadata,
                    effective_sample_size,
                )
            ),
        },
    }


def produce_grouped_posterior_quality(
        runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        produce_posterior_quality(rows, metadata=rows[0]["metadata"])
        for _, rows in _group_records_by_problem_method(
            runs,
            context="posterior_quality",
        )
    ]


def produce_posterior_wasserstein(
        runs: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_records_match_metadata(
        runs,
        metadata,
        context="posterior_wasserstein",
    )
    seed_count = _assert_independent_seed_records(
        runs,
        context="posterior_wasserstein",
    )
    aggregate_metadata = _aggregate_metadata(metadata, runs)
    wasserstein_by_seed = []
    reference_sample_count = 0
    posterior_sample_count = 0
    effective_sample_size = 0.0
    dimension = None
    for run in runs:
        reference = _as_sample_matrix(
            run["reference_samples"],
            "reference_samples",
        )
        posterior = _as_sample_matrix(
            run["posterior_samples"],
            "posterior_samples",
        )
        wasserstein_by_seed.append(_wasserstein_distance(reference, posterior))
        reference_sample_count += int(reference.shape[0])
        posterior_sample_count += int(posterior.shape[0])
        effective_sample_size += float(run["effective_sample_size"])
        if dimension is None:
            dimension = int(reference.shape[1])
        elif dimension != int(reference.shape[1]):
            raise ValueError("All posterior Wasserstein runs must share dimension.")

    return {
        "metric_family": "posterior_wasserstein",
        "metadata": aggregate_metadata,
        "posterior_wasserstein": {
            "num_seeds": seed_count,
            "wasserstein": float(np.mean(wasserstein_by_seed)),
            "reference_sample_count": reference_sample_count,
            "posterior_sample_count": posterior_sample_count,
            "effective_sample_size": effective_sample_size,
            "likelihood_evaluations_per_effective_sample": (
                _likelihood_evaluations_per_ess(
                    aggregate_metadata,
                    effective_sample_size,
                )
            ),
            "dimension": int(dimension),
        },
    }


def produce_grouped_posterior_wasserstein(
        runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        produce_posterior_wasserstein(rows, metadata=rows[0]["metadata"])
        for _, rows in _group_records_by_problem_method(
            runs,
            context="posterior_wasserstein",
        )
    ]


def produce_performance_guardrail(
        *,
        metadata: Mapping[str, Any],
        name: str,
        observed_seconds: float,
        threshold_seconds: float,
        rationale: str,
        comparison: str = "<=",
) -> dict[str, Any]:
    if comparison == "<=":
        passed = observed_seconds <= threshold_seconds
    elif comparison == "<":
        passed = observed_seconds < threshold_seconds
    else:
        raise ValueError("comparison must be '<=' or '<'.")
    return {
        "metric_family": "performance_guardrail",
        "metadata": copy.deepcopy(metadata),
        "performance_guardrail": {
            "name": str(name),
            "observed_seconds": float(observed_seconds),
            "threshold_seconds": float(threshold_seconds),
            "comparison": comparison,
            "passed": bool(passed),
            "rationale": str(rationale),
        },
    }


def produce_performance_guardrail_suite(
        *,
        metadata: Mapping[str, Any],
        observed_seconds_by_name: Mapping[str, float],
        threshold_seconds_by_name: Mapping[str, float],
        rationales_by_name: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    rationales = {} if rationales_by_name is None else rationales_by_name
    records = []
    for name, observed_seconds in observed_seconds_by_name.items():
        if name not in REQUIRED_PERFORMANCE_GUARDRAIL_NAMES:
            raise ValueError(f"Unknown performance guardrail: {name!r}.")
        if name not in threshold_seconds_by_name:
            raise ValueError(
                f"Missing threshold_seconds for performance guardrail {name!r}."
            )
        records.append(
            produce_performance_guardrail(
                metadata=metadata,
                name=name,
                observed_seconds=observed_seconds,
                threshold_seconds=threshold_seconds_by_name[name],
                rationale=rationales.get(
                    name,
                    f"{name} hot-path guardrail for release timing history.",
                ),
            )
        )
    missing = REQUIRED_PERFORMANCE_GUARDRAIL_NAMES.difference(
        observed_seconds_by_name
    )
    if missing:
        raise ValueError(
            f"Missing performance guardrail observations: {sorted(missing)}."
        )
    return records


def append_timing_history(
        previous_history: Sequence[Mapping[str, Any]],
        new_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    next_history = copy.deepcopy(list(previous_history))
    next_history.append(copy.deepcopy(new_row))
    return next_history


def _group_records_by_likelihood_evaluations(
        records: Sequence[Mapping[str, Any]],
) -> list[tuple[int, list[Mapping[str, Any]]]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        evaluations = int(record["metadata"]["likelihood_evaluations"])
        groups.setdefault(evaluations, []).append(record)
    if not groups:
        raise ValueError("At least one calibration record is required.")
    return [
        (evaluations, groups[evaluations])
        for evaluations in sorted(groups)
    ]


def _group_records_by_problem_method(
        records: Sequence[Mapping[str, Any]],
        *,
        context: str,
) -> list[tuple[tuple[str, Any], list[Mapping[str, Any]]]]:
    groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
    order: list[tuple[str, Any]] = []
    for record in records:
        key = _problem_method_key(record["metadata"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    if not groups:
        raise ValueError(f"{context} requires at least one record.")
    return [
        (key, groups[key])
        for key in order
    ]


def _assert_records_match_metadata(
        records: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        *,
        context: str,
) -> None:
    expected_key = _problem_method_key(metadata)
    for record in records:
        actual_key = _problem_method_key(record["metadata"])
        if actual_key != expected_key:
            raise ValueError(
                f"{context} records must share problem and full "
                "method_setting; use the grouped producer for mixed inputs."
            )


def _assert_independent_seed_records(
        records: Sequence[Mapping[str, Any]],
        *,
        context: str,
) -> int:
    rows = list(records)
    if len(rows) < 2:
        raise ValueError(f"{context} requires at least two independent seeds.")
    seeds = [
        int(row["metadata"]["seed"])
        for row in rows
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{context} contains duplicate seed records.")
    return len(seeds)


def _calibration_bias(calibration: Mapping[str, Any]) -> float:
    if "bias_logZ" in calibration:
        return float(calibration["bias_logZ"])
    return float(calibration["hat_logZ"]) - float(calibration["logZ_ref"])


def _aggregate_rmse_group_metadata(
        records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("rmse_vs_likelihood requires at least one record.")
    aggregate = copy.deepcopy(rows[0]["metadata"])
    aggregate["seed"] = int(min(
        int(row["metadata"]["seed"])
        for row in rows
    ))
    aggregate["likelihood_evaluations"] = int(max(
        int(row["metadata"]["likelihood_evaluations"])
        for row in rows
    ))
    wall_clocks = [
        row["metadata"]["wall_clock_seconds"]
        for row in rows
    ]
    if any(value is None for value in wall_clocks):
        aggregate["wall_clock_seconds"] = None
    else:
        aggregate["wall_clock_seconds"] = float(sum(
            float(value)
            for value in wall_clocks
        ))
    aggregate["worker_count"] = int(max(
        int(row["metadata"]["worker_count"])
        for row in rows
    ))
    return aggregate


def _aggregate_metadata(
        metadata: Mapping[str, Any],
        runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = copy.deepcopy(metadata)
    aggregate["likelihood_evaluations"] = int(sum(
        int(run["metadata"]["likelihood_evaluations"])
        for run in runs
    ))
    aggregate["wall_clock_seconds"] = float(sum(
        float(run["metadata"]["wall_clock_seconds"])
        for run in runs
        if run["metadata"]["wall_clock_seconds"] is not None
    ))
    return aggregate


def _likelihood_evaluations_per_ess(
        metadata: Mapping[str, Any],
        effective_sample_size: float,
) -> float:
    if effective_sample_size <= 0.0:
        raise ValueError("effective_sample_size must be positive.")
    return float(metadata["likelihood_evaluations"]) / float(effective_sample_size)


def _as_sample_matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    samples = np.asarray(value, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if samples.shape[0] < 1:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain finite values.")
    return samples


def _wasserstein_distance(reference: np.ndarray, posterior: np.ndarray) -> float:
    if reference.shape[1] != posterior.shape[1]:
        raise ValueError(
            "reference_samples and posterior_samples must have the same "
            "dimension."
        )
    paired_count = min(reference.shape[0], posterior.shape[0])
    if paired_count < 1:
        raise ValueError("At least one reference and posterior sample is required.")
    reference_sorted = _sort_samples(reference)
    posterior_sorted = _sort_samples(posterior)
    ref_idx = np.linspace(
        0,
        reference.shape[0] - 1,
        paired_count,
    ).round().astype(int)
    post_idx = np.linspace(
        0,
        posterior.shape[0] - 1,
        paired_count,
    ).round().astype(int)
    return float(np.mean(
        np.linalg.norm(
            reference_sorted[ref_idx] - posterior_sorted[post_idx],
            axis=1,
        )
    ))


def _sort_samples(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] == 1:
        return samples[np.argsort(samples[:, 0])]
    keys = tuple(samples[:, dim] for dim in reversed(range(samples.shape[1])))
    return samples[np.lexsort(keys)]


def _method_setting_key(method_setting: Mapping[str, Any]) -> Any:
    return _freeze_key(method_setting)


def _problem_method_key(metadata: Mapping[str, Any]) -> tuple[str, Any]:
    return (
        str(metadata["problem"]),
        _method_setting_key(metadata["method_setting"]),
    )


def _freeze_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_key(child))
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze_key(child) for child in value)
    if isinstance(value, np.generic):
        return value.item()
    return value
