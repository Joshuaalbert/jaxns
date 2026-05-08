import importlib

import numpy as np
import pytest

from benchmarks.v3_validation.schema_checks import (
    assert_calibration_table,
    assert_calibration_rollup_table,
    assert_performance_guardrail_record,
    assert_performance_guardrail_suite,
    assert_posterior_quality_record,
    assert_posterior_wasserstein_record,
    assert_rmse_vs_likelihood_record,
    assert_timing_history_append_only,
)


def _metadata(
        seed: int = 0,
        evaluations: int = 128,
        method_setting: dict | None = None,
        problem: str = "three_atom_step",
) -> dict:
    if method_setting is None:
        method_setting = {
            "method": "phantom-conditioned",
            "allocation": "uniform",
        }
    return {
        "method_setting": method_setting,
        "seed": seed,
        "problem": problem,
        "likelihood_evaluations": evaluations,
        "wall_clock_seconds": 0.01,
        "worker_count": 1,
        "commit": "test-producer",
    }


def _dynamic_target_setting(target: str) -> dict:
    return {
        "method": "phantom-conditioned",
        "allocation": {
            "mode": "dynamic",
            "target": target,
        },
    }


def _require_producers():
    try:
        return importlib.import_module("benchmarks.v3_validation.producers")
    except ModuleNotFoundError as error:
        raise AssertionError(
            "Ticket 0010 requires benchmarks.v3_validation.producers with "
            "multi-seed calibration, RMSE/Pareto, posterior Wasserstein, "
            "likelihood-evaluations-per-ESS, MSE-cost dominance, full "
            "method_setting rollup, performance guardrail, aggregate z_logZ "
            "rollup, and append-only timing-history producers."
        ) from error


def _calibration_record(
        *,
        seed: int,
        evaluations: int,
        bias_logZ: float,
        problem: str = "three_atom_step",
        method_setting: dict | None = None,
) -> dict:
    return {
        "metadata": _metadata(
            seed=seed,
            evaluations=evaluations,
            problem=problem,
            method_setting=method_setting,
        ),
        "evidence_calibration": {
            "bias_logZ": bias_logZ,
        },
    }


def _posterior_quality_run(
        *,
        seed: int,
        evaluations: int,
        wasserstein_mc: float,
        problem: str = "three_atom_step",
        method_setting: dict | None = None,
) -> dict:
    return {
        "metadata": _metadata(
            seed=seed,
            evaluations=evaluations,
            problem=problem,
            method_setting=method_setting,
        ),
        "wasserstein_mc": wasserstein_mc,
        "reference_sample_count": 10_000,
        "posterior_sample_count": 1_000,
        "effective_sample_size": 25.0,
    }


def _posterior_wasserstein_run(
        *,
        seed: int,
        evaluations: int,
        offset: float,
        problem: str = "three_atom_step",
        method_setting: dict | None = None,
) -> dict:
    reference = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    return {
        "metadata": _metadata(
            seed=seed,
            evaluations=evaluations,
            problem=problem,
            method_setting=method_setting,
        ),
        "reference_samples": reference,
        "posterior_samples": reference + offset,
        "effective_sample_size": 25.0,
    }


def test_multi_seed_calibration_producer_emits_schema_checked_table():
    producers = _require_producers()
    runs = [
        {
            "metadata": _metadata(seed=0, evaluations=64),
            "log_Z_samples": [1.00, 1.05, 1.10],
            "reported_uncertainty_logZ": 0.05,
            "empirical_uncertainty_logZ": 0.05,
            "expectation_logZ": 1.04,
            "mc_shrinkage_logZ": 1.05,
            "rho_g": [1.0, 0.9],
            "rho_fit": [0.98, 0.91],
        },
        {
            "metadata": _metadata(seed=1, evaluations=64),
            "log_Z_samples": [0.99, 1.03, 1.08],
            "reported_uncertainty_logZ": 0.05,
            "empirical_uncertainty_logZ": 0.05,
            "expectation_logZ": 1.03,
            "mc_shrinkage_logZ": 1.04,
            "rho_g": [1.0, 0.88],
            "rho_fit": [0.97, 0.90],
        },
        {
            "metadata": _metadata(
                seed=0,
                evaluations=64,
                method_setting={
                    "method": "phantom-conditioned",
                    "allocation": "dynamic",
                },
            ),
            "log_Z_samples": [1.02, 1.04, 1.09],
            "reported_uncertainty_logZ": 0.05,
            "empirical_uncertainty_logZ": 0.05,
            "expectation_logZ": 1.04,
            "mc_shrinkage_logZ": 1.05,
            "rho_g": [1.0, 0.86],
            "rho_fit": [0.96, 0.89],
        },
        {
            "metadata": _metadata(
                seed=1,
                evaluations=64,
                method_setting={
                    "method": "phantom-conditioned",
                    "allocation": "dynamic",
                },
            ),
            "log_Z_samples": [1.01, 1.06, 1.07],
            "reported_uncertainty_logZ": 0.05,
            "empirical_uncertainty_logZ": 0.05,
            "expectation_logZ": 1.04,
            "mc_shrinkage_logZ": 1.05,
            "rho_g": [1.0, 0.85],
            "rho_fit": [0.95, 0.88],
        },
    ]

    table = producers.produce_multi_seed_evidence_calibration(
        runs,
        logZ_ref=1.02,
    )

    assert_calibration_table(table)
    assert [row["metadata"]["seed"] for row in table] == [0, 1, 0, 1]

    rollups = producers.produce_multi_seed_calibration_rollups(table)
    assert_calibration_rollup_table(table, rollups)
    uniform_rows = [
        row
        for row in table
        if row["metadata"]["method_setting"]["allocation"] == "uniform"
    ]
    uniform_z_scores = [
        row["evidence_calibration"]["z_logZ"]
        for row in uniform_rows
    ]
    assert len(rollups) == 2
    assert rollups[0]["problem"] == "three_atom_step"
    assert rollups[0]["method"] == "phantom-conditioned"
    assert rollups[0]["method_setting"] == {
        "method": "phantom-conditioned",
        "allocation": "uniform",
    }
    assert rollups[0]["num_seeds"] == 2
    np.testing.assert_allclose(
        rollups[0]["mean_z_logZ"],
        np.mean(uniform_z_scores),
    )
    np.testing.assert_allclose(
        rollups[0]["sd_z_logZ"],
        np.std(uniform_z_scores, ddof=1),
    )
    assert rollups[1]["method"] == "phantom-conditioned"
    assert rollups[1]["method_setting"]["allocation"] == "dynamic"


def test_rmse_vs_likelihood_pareto_producer_aggregates_calibration_records():
    producers = _require_producers()
    calibration_records = [
        {
            "metadata": _metadata(seed=0, evaluations=64),
            "evidence_calibration": {
                "bias_logZ": 0.20,
            },
        },
        {
            "metadata": _metadata(seed=1, evaluations=64),
            "evidence_calibration": {
                "bias_logZ": -0.10,
            },
        },
        {
            "metadata": _metadata(seed=0, evaluations=128),
            "evidence_calibration": {
                "bias_logZ": 0.10,
            },
        },
        {
            "metadata": _metadata(seed=1, evaluations=128),
            "evidence_calibration": {
                "bias_logZ": -0.05,
            },
        },
        {
            "metadata": _metadata(seed=0, evaluations=256),
            "evidence_calibration": {
                "bias_logZ": 0.13,
            },
        },
        {
            "metadata": _metadata(seed=1, evaluations=256),
            "evidence_calibration": {
                "bias_logZ": -0.15,
            },
        },
    ]

    record = producers.produce_rmse_vs_likelihood_pareto(
        calibration_records,
        metadata=_metadata(seed=0, evaluations=256),
    )

    assert_rmse_vs_likelihood_record(record)
    assert record["rmse_vs_likelihood"]["num_seeds"] == [2, 2, 2]
    assert record["rmse_vs_likelihood"]["pareto_efficient"] == [
        True,
        True,
        False,
    ]
    np.testing.assert_allclose(
        record["rmse_vs_likelihood"]["rmse_logZ"],
        [
            np.sqrt((0.20 ** 2 + (-0.10) ** 2) / 2.0),
            np.sqrt((0.10 ** 2 + (-0.05) ** 2) / 2.0),
            np.sqrt((0.13 ** 2 + (-0.15) ** 2) / 2.0),
        ],
    )
    np.testing.assert_allclose(
        record["rmse_vs_likelihood"]["mse_times_likelihood_evaluations"],
        [1.6, 0.8, 5.0432],
    )
    assert record["rmse_vs_likelihood"]["dominance_summary"][
        "dominated_indices"
    ] == [0, 2]

    with pytest.raises(ValueError, match="at least two"):
        producers.produce_rmse_vs_likelihood_pareto(
            calibration_records[:1],
            metadata=_metadata(seed=0, evaluations=64),
        )


def test_grouped_rmse_vs_likelihood_scopes_problem_and_full_method_setting():
    producers = _require_producers()
    dynamic_evidence_setting = _dynamic_target_setting("evidence")
    dynamic_posterior_setting = _dynamic_target_setting("posterior")
    calibration_records = [
        _calibration_record(
            seed=0,
            evaluations=64,
            bias_logZ=0.10,
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=64,
            bias_logZ=-0.10,
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=0,
            evaluations=128,
            bias_logZ=0.05,
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=128,
            bias_logZ=-0.05,
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=0,
            evaluations=64,
            bias_logZ=0.30,
            method_setting=dynamic_posterior_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=64,
            bias_logZ=0.30,
            method_setting=dynamic_posterior_setting,
        ),
        _calibration_record(
            seed=0,
            evaluations=128,
            bias_logZ=0.20,
            method_setting=dynamic_posterior_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=128,
            bias_logZ=0.20,
            method_setting=dynamic_posterior_setting,
        ),
        _calibration_record(
            seed=0,
            evaluations=64,
            bias_logZ=-0.40,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=64,
            bias_logZ=0.40,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=0,
            evaluations=128,
            bias_logZ=-0.10,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
        _calibration_record(
            seed=1,
            evaluations=128,
            bias_logZ=0.10,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
    ]

    records = producers.produce_grouped_rmse_vs_likelihood_pareto(
        calibration_records,
    )

    assert len(records) == 3
    for record in records:
        assert_rmse_vs_likelihood_record(record)
    keys = [
        (
            record["metadata"]["problem"],
            record["metadata"]["method_setting"],
        )
        for record in records
    ]
    assert keys == [
        ("three_atom_step", dynamic_evidence_setting),
        ("three_atom_step", dynamic_posterior_setting),
        ("gaussian_shell", dynamic_evidence_setting),
    ]
    assert {
        record["metadata"]["method_setting"]["allocation"]["mode"]
        for record in records[:2]
    } == {"dynamic"}
    assert [
        record["metadata"]["method_setting"]["allocation"]["target"]
        for record in records[:2]
    ] == ["evidence", "posterior"]
    np.testing.assert_allclose(
        records[0]["rmse_vs_likelihood"]["rmse_logZ"],
        [0.10, 0.05],
    )
    np.testing.assert_allclose(
        records[1]["rmse_vs_likelihood"]["rmse_logZ"],
        [0.30, 0.20],
    )
    np.testing.assert_allclose(
        records[2]["rmse_vs_likelihood"]["rmse_logZ"],
        [0.40, 0.10],
    )

    with pytest.raises(ValueError, match="problem and full method_setting"):
        producers.produce_rmse_vs_likelihood_pareto(
            calibration_records,
            metadata=_metadata(method_setting=dynamic_evidence_setting),
        )


def test_posterior_quality_producer_reports_likelihood_evaluations_per_ess():
    producers = _require_producers()
    runs = [
        {
            "metadata": _metadata(seed=6, evaluations=256),
            "wasserstein_mc": 0.100,
            "reference_sample_count": 10_000,
            "posterior_sample_count": 1_000,
            "effective_sample_size": 32.0,
        },
        {
            "metadata": _metadata(seed=7, evaluations=384),
            "wasserstein_mc": 0.150,
            "reference_sample_count": 10_000,
            "posterior_sample_count": 1_000,
            "effective_sample_size": 48.0,
        },
    ]

    record = producers.produce_posterior_quality(
        runs,
        metadata=_metadata(seed=6, evaluations=1),
    )

    assert_posterior_quality_record(record)
    assert record["posterior_quality"]["num_seeds"] == 2
    assert record["posterior_quality"]["wasserstein_mc"] == 0.125
    assert record["posterior_quality"]["effective_sample_size"] == 80.0
    assert record["posterior_quality"][
        "likelihood_evaluations_per_effective_sample"
    ] == 8.0

    with pytest.raises(ValueError, match="at least two"):
        producers.produce_posterior_quality(
            runs[:1],
            metadata=_metadata(seed=6, evaluations=1),
        )


def test_grouped_posterior_quality_scopes_problem_and_full_method_setting():
    producers = _require_producers()
    dynamic_evidence_setting = _dynamic_target_setting("evidence")
    dynamic_posterior_setting = _dynamic_target_setting("posterior")
    runs = [
        _posterior_quality_run(
            seed=0,
            evaluations=100,
            wasserstein_mc=0.10,
            method_setting=dynamic_evidence_setting,
        ),
        _posterior_quality_run(
            seed=1,
            evaluations=100,
            wasserstein_mc=0.20,
            method_setting=dynamic_evidence_setting,
        ),
        _posterior_quality_run(
            seed=0,
            evaluations=100,
            wasserstein_mc=0.40,
            method_setting=dynamic_posterior_setting,
        ),
        _posterior_quality_run(
            seed=1,
            evaluations=100,
            wasserstein_mc=0.50,
            method_setting=dynamic_posterior_setting,
        ),
        _posterior_quality_run(
            seed=0,
            evaluations=100,
            wasserstein_mc=0.70,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
        _posterior_quality_run(
            seed=1,
            evaluations=100,
            wasserstein_mc=0.80,
            problem="gaussian_shell",
            method_setting=dynamic_evidence_setting,
        ),
    ]

    records = producers.produce_grouped_posterior_quality(runs)

    assert len(records) == 3
    for record in records:
        assert_posterior_quality_record(record)
    keys = [
        (
            record["metadata"]["problem"],
            record["metadata"]["method_setting"],
        )
        for record in records
    ]
    assert keys == [
        ("three_atom_step", dynamic_evidence_setting),
        ("three_atom_step", dynamic_posterior_setting),
        ("gaussian_shell", dynamic_evidence_setting),
    ]
    assert {
        record["metadata"]["method_setting"]["allocation"]["mode"]
        for record in records[:2]
    } == {"dynamic"}
    assert [
        record["metadata"]["method_setting"]["allocation"]["target"]
        for record in records[:2]
    ] == ["evidence", "posterior"]
    np.testing.assert_allclose(
        [
            record["posterior_quality"]["wasserstein_mc"]
            for record in records
        ],
        [0.15, 0.45, 0.75],
    )
    assert [
        record["posterior_quality"][
            "likelihood_evaluations_per_effective_sample"
        ]
        for record in records
    ] == [4.0, 4.0, 4.0]

    wasserstein_records = producers.produce_grouped_posterior_wasserstein(
        [
            _posterior_wasserstein_run(
                seed=0,
                evaluations=100,
                offset=0.10,
                method_setting=dynamic_evidence_setting,
            ),
            _posterior_wasserstein_run(
                seed=1,
                evaluations=100,
                offset=0.20,
                method_setting=dynamic_evidence_setting,
            ),
            _posterior_wasserstein_run(
                seed=0,
                evaluations=100,
                offset=0.40,
                method_setting=dynamic_posterior_setting,
            ),
            _posterior_wasserstein_run(
                seed=1,
                evaluations=100,
                offset=0.50,
                method_setting=dynamic_posterior_setting,
            ),
            _posterior_wasserstein_run(
                seed=0,
                evaluations=100,
                offset=0.70,
                problem="gaussian_shell",
                method_setting=dynamic_evidence_setting,
            ),
            _posterior_wasserstein_run(
                seed=1,
                evaluations=100,
                offset=0.80,
                problem="gaussian_shell",
                method_setting=dynamic_evidence_setting,
            ),
        ]
    )
    assert len(wasserstein_records) == 3
    for record in wasserstein_records:
        assert_posterior_wasserstein_record(record)
    wasserstein_keys = [
        (
            record["metadata"]["problem"],
            record["metadata"]["method_setting"],
        )
        for record in wasserstein_records
    ]
    assert wasserstein_keys == [
        ("three_atom_step", dynamic_evidence_setting),
        ("three_atom_step", dynamic_posterior_setting),
        ("gaussian_shell", dynamic_evidence_setting),
    ]
    assert {
        record["metadata"]["method_setting"]["allocation"]["mode"]
        for record in wasserstein_records[:2]
    } == {"dynamic"}
    assert [
        record["metadata"]["method_setting"]["allocation"]["target"]
        for record in wasserstein_records[:2]
    ] == ["evidence", "posterior"]
    np.testing.assert_allclose(
        [
            record["posterior_wasserstein"]["wasserstein"]
            for record in wasserstein_records
        ],
        [0.15, 0.45, 0.75],
    )

    with pytest.raises(ValueError, match="problem and full method_setting"):
        producers.produce_posterior_quality(
            runs,
            metadata=_metadata(method_setting=dynamic_evidence_setting),
        )


def test_posterior_wasserstein_producer_emits_posterior_metric_only():
    producers = _require_producers()
    reference = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    posterior_a = np.asarray([[0.25], [1.25], [2.25], [3.25]], dtype=float)
    posterior_b = np.asarray([[0.75], [1.75], [2.75], [3.75]], dtype=float)
    runs = [
        {
            "metadata": _metadata(seed=2, evaluations=128),
            "reference_samples": reference,
            "posterior_samples": posterior_a,
            "effective_sample_size": 32.0,
        },
        {
            "metadata": _metadata(seed=3, evaluations=128),
            "reference_samples": reference,
            "posterior_samples": posterior_b,
            "effective_sample_size": 32.0,
        },
    ]

    record = producers.produce_posterior_wasserstein(
        runs,
        metadata=_metadata(seed=2, evaluations=1),
    )

    assert_posterior_wasserstein_record(record)
    assert record["posterior_wasserstein"]["num_seeds"] == 2
    assert record["posterior_wasserstein"]["wasserstein"] == 0.5
    assert record["posterior_wasserstein"][
        "likelihood_evaluations_per_effective_sample"
    ] == 4.0

    with pytest.raises(ValueError, match="at least two"):
        producers.produce_posterior_wasserstein(
            runs[:1],
            metadata=_metadata(seed=2, evaluations=1),
        )


def test_performance_guardrail_producer_records_threshold_rationale():
    producers = _require_producers()

    record = producers.produce_performance_guardrail(
        metadata=_metadata(seed=3),
        name="block_construction",
        observed_seconds=0.21,
        threshold_seconds=0.50,
        rationale="Python baseline guardrail for trend regressions.",
    )

    assert_performance_guardrail_record(record)
    assert record["performance_guardrail"]["passed"] is True


def test_performance_guardrail_suite_producer_covers_ticket_hot_paths():
    producers = _require_producers()
    observed_seconds = {
        "block_construction": 0.01,
        "shrinkage_sampling": 0.02,
        "phantom_counting": 0.03,
        "rho_bootstrap": 0.04,
        "trajectories": 0.05,
        "serialization": 0.01,
        "worker_task_latency": 0.06,
    }
    threshold_seconds = {
        name: seconds * 10.0
        for name, seconds in observed_seconds.items()
    }

    records = producers.produce_performance_guardrail_suite(
        metadata=_metadata(seed=7),
        observed_seconds_by_name=observed_seconds,
        threshold_seconds_by_name=threshold_seconds,
    )

    assert_performance_guardrail_suite(records)
    assert [
        record["performance_guardrail"]["name"]
        for record in records
    ] == list(observed_seconds)


def test_timing_history_producer_is_append_only_and_non_mutating():
    producers = _require_producers()
    previous = [
        {
            "metadata": _metadata(seed=4),
            "timings": {
                "block_construction_seconds": 0.21,
            },
        },
    ]
    new_row = {
        "metadata": _metadata(seed=5),
        "timings": {
            "block_construction_seconds": 0.19,
        },
    }

    next_history = producers.append_timing_history(previous, new_row)

    assert_timing_history_append_only(previous, next_history)
    assert previous == [next_history[0]]
    assert next_history[-1] == new_row
