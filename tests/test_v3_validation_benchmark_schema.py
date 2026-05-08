import copy

import numpy as np
import pytest

from benchmarks.v3_validation.schema_checks import (
    assert_calibration_table,
    assert_calibration_rollup_table,
    assert_benchmark_metadata,
    assert_evidence_calibration_record,
    assert_performance_guardrail_record,
    assert_performance_guardrail_suite,
    assert_posterior_quality_record,
    assert_posterior_wasserstein_record,
    assert_rmse_vs_likelihood_record,
    assert_rollup_separates_evidence_and_posterior,
    assert_timing_history_append_only,
)


def _metadata(seed: int = 11) -> dict:
    return {
        "method_setting": {
            "method": "baseline-race-tree",
            "allocation": "uniform",
        },
        "seed": seed,
        "problem": "three_atom_step",
        "likelihood_evaluations": 256,
        "wall_clock_seconds": 0.25,
        "worker_count": 1,
        "software_version": "3.0.0-test",
    }


def _phantom_metadata(seed: int = 11) -> dict:
    metadata = _metadata(seed=seed)
    metadata["method_setting"] = {
        "method": "phantom-conditioned",
        "allocation": "uniform",
    }
    return metadata


def _evidence_record() -> dict:
    return {
        "metric_family": "evidence_calibration",
        "metadata": _phantom_metadata(),
        "evidence_calibration": {
            "summary_convention": "mean_logZ_samples",
            "sigma_convention": "sample_sd_logZ_samples_ddof1",
            "logZ_ref": 1.0986122886681098,
            "hat_logZ": 1.100,
            "sigma_logZ": 0.05,
            "z_logZ": 0.027754226637803337,
            "bias_logZ": 0.001387711331890291,
            "rmse_logZ": 0.05,
            "reported_uncertainty_logZ": 0.05,
            "empirical_uncertainty_logZ": 0.051,
            "expectation_logZ": 1.099,
            "mc_shrinkage_logZ": 1.100,
            "rho_g": [1.0, 0.8, 0.7],
            "rho_fit": [0.95, 0.82, 0.72],
        },
    }


def _evidence_record_for_group(
        *,
        seed: int,
        problem: str = "three_atom_step",
        method: str = "phantom-conditioned",
        method_setting: dict | None = None,
        z_logZ: float = 0.0,
) -> dict:
    record = _evidence_record()
    record["metadata"]["seed"] = seed
    record["metadata"]["problem"] = problem
    if method_setting is None:
        record["metadata"]["method_setting"]["method"] = method
    else:
        record["metadata"]["method_setting"] = method_setting
    record["evidence_calibration"]["z_logZ"] = z_logZ
    return record


def _baseline_evidence_record_without_rho() -> dict:
    record = _evidence_record()
    record["metadata"] = _metadata()
    del record["evidence_calibration"]["rho_g"]
    del record["evidence_calibration"]["rho_fit"]
    return record


def _posterior_record() -> dict:
    return {
        "metric_family": "posterior_quality",
        "metadata": _metadata(seed=12),
        "posterior_quality": {
            "num_seeds": 2,
            "wasserstein_mc": 0.125,
            "reference_sample_count": 10_000,
            "posterior_sample_count": 1_000,
            "effective_sample_size": 20.48,
            "likelihood_evaluations_per_effective_sample": 12.5,
        },
    }


def _rmse_vs_likelihood_record() -> dict:
    return {
        "metric_family": "rmse_vs_likelihood",
        "metadata": _metadata(seed=13),
        "rmse_vs_likelihood": {
            "likelihood_evaluations": [64, 128, 256],
            "num_seeds": [2, 2, 2],
            "rmse_logZ": [0.20, 0.12, 0.11],
            "mse_times_likelihood_evaluations": [2.56, 1.8432, 3.0976],
            "pareto_efficient": [True, True, False],
            "dominance_summary": {
                "best_index": 1,
                "best_likelihood_evaluations": 128,
                "best_mse_times_likelihood_evaluations": 1.8432,
                "dominated_indices": [0, 2],
            },
        },
    }


def _posterior_wasserstein_record() -> dict:
    return {
        "metric_family": "posterior_wasserstein",
        "metadata": _metadata(seed=14),
        "posterior_wasserstein": {
            "num_seeds": 2,
            "wasserstein": 0.03125,
            "reference_sample_count": 20_000,
            "posterior_sample_count": 2_000,
            "effective_sample_size": 32.0,
            "likelihood_evaluations_per_effective_sample": 8.0,
            "dimension": 2,
        },
    }


def _performance_guardrail_record() -> dict:
    return {
        "metric_family": "performance_guardrail",
        "metadata": _metadata(seed=15),
        "performance_guardrail": {
            "name": "block_construction",
            "observed_seconds": 0.21,
            "threshold_seconds": 0.50,
            "comparison": "<=",
            "passed": True,
            "rationale": "Python baseline guardrail for trend regressions.",
        },
    }


def _timing_row(seed: int, seconds: float) -> dict:
    return {
        "metadata": _metadata(seed=seed),
        "timings": {
            "block_construction_seconds": seconds,
        },
    }


def test_benchmark_metadata_requires_reproducibility_fields():
    metadata = _metadata()
    assert_benchmark_metadata(metadata)

    missing_seed = copy.deepcopy(metadata)
    del missing_seed["seed"]
    with pytest.raises(AssertionError, match="seed"):
        assert_benchmark_metadata(missing_seed)

    missing_version = copy.deepcopy(metadata)
    del missing_version["software_version"]
    with pytest.raises(AssertionError, match="software_version|commit"):
        assert_benchmark_metadata(missing_version)

    bad_cost = copy.deepcopy(metadata)
    bad_cost["likelihood_evaluations"] = 0
    with pytest.raises(AssertionError, match="likelihood_evaluations"):
        assert_benchmark_metadata(bad_cost)


def test_evidence_calibration_schema_rejects_posterior_metrics():
    record = _evidence_record()
    assert_evidence_calibration_record(record)

    mixed_record = copy.deepcopy(record)
    mixed_record["evidence_calibration"]["wasserstein_mc"] = 0.2
    with pytest.raises(AssertionError, match="Forbidden metric keys"):
        assert_evidence_calibration_record(mixed_record)


def test_evidence_calibration_rho_diagnostics_are_method_conditional():
    baseline = _baseline_evidence_record_without_rho()
    assert_evidence_calibration_record(baseline)

    phantom_missing_rho = _evidence_record()
    del phantom_missing_rho["evidence_calibration"]["rho_g"]
    with pytest.raises(AssertionError, match="rho_g"):
        assert_evidence_calibration_record(phantom_missing_rho)

    phantom_bad_rho = _evidence_record()
    phantom_bad_rho["evidence_calibration"]["rho_g"] = [1.0, 0.0]
    with pytest.raises(AssertionError, match="rho_g"):
        assert_evidence_calibration_record(phantom_bad_rho)


def test_calibration_table_scopes_duplicate_seeds_by_problem_and_method():
    dynamic_setting = {
        "method": "phantom-conditioned",
        "allocation": "dynamic",
    }
    table = [
        _evidence_record_for_group(seed=11),
        _evidence_record_for_group(seed=12),
        _evidence_record_for_group(seed=11, problem="gaussian_shell"),
        _evidence_record_for_group(
            seed=11,
            method_setting=dynamic_setting,
        ),
    ]
    assert_calibration_table(table)

    duplicate_seed = [
        _evidence_record_for_group(seed=11),
        _evidence_record_for_group(seed=11),
    ]
    with pytest.raises(
        AssertionError,
        match="Duplicate problem/method_setting/seed",
    ):
        assert_calibration_table(duplicate_seed)


def test_calibration_rollup_requires_mean_and_sd_z_logz_by_full_method_setting():
    uniform_setting = {
        "method": "phantom-conditioned",
        "allocation": "uniform",
    }
    dynamic_setting = {
        "method": "phantom-conditioned",
        "allocation": "dynamic",
    }
    records = [
        _evidence_record_for_group(
            seed=11,
            method_setting=uniform_setting,
            z_logZ=-1.0,
        ),
        _evidence_record_for_group(
            seed=12,
            method_setting=uniform_setting,
            z_logZ=1.0,
        ),
        _evidence_record_for_group(
            seed=11,
            method_setting=dynamic_setting,
            z_logZ=2.0,
        ),
        _evidence_record_for_group(
            seed=12,
            method_setting=dynamic_setting,
            z_logZ=4.0,
        ),
    ]
    rollups = [
        {
            "problem": "three_atom_step",
            "method": "phantom-conditioned",
            "method_setting": uniform_setting,
            "num_seeds": 2,
            "mean_z_logZ": 0.0,
            "sd_z_logZ": np.sqrt(2.0),
        },
        {
            "problem": "three_atom_step",
            "method": "phantom-conditioned",
            "method_setting": dynamic_setting,
            "num_seeds": 2,
            "mean_z_logZ": 3.0,
            "sd_z_logZ": np.sqrt(2.0),
        },
    ]
    assert_calibration_rollup_table(records, rollups)

    wrong_mean = copy.deepcopy(rollups)
    wrong_mean[0]["mean_z_logZ"] = 0.25
    with pytest.raises(AssertionError, match="mean_z_logZ"):
        assert_calibration_rollup_table(records, wrong_mean)

    missing_rollup = rollups[:1]
    with pytest.raises(AssertionError, match="Missing calibration rollups"):
        assert_calibration_rollup_table(records, missing_rollup)

    method_only_rollup = copy.deepcopy(rollups)
    del method_only_rollup[0]["method_setting"]
    with pytest.raises(AssertionError, match="method_setting"):
        assert_calibration_rollup_table(records, method_only_rollup)


def test_rmse_vs_likelihood_schema_requires_pareto_series():
    record = _rmse_vs_likelihood_record()
    assert_rmse_vs_likelihood_record(record)
    assert record["rmse_vs_likelihood"]["mse_times_likelihood_evaluations"] == [
        2.56,
        1.8432,
        3.0976,
    ]
    assert record["rmse_vs_likelihood"]["dominance_summary"][
        "dominated_indices"
    ] == [0, 2]

    unsorted = copy.deepcopy(record)
    unsorted["rmse_vs_likelihood"]["likelihood_evaluations"] = [128, 64, 256]
    with pytest.raises(AssertionError, match="strictly increasing"):
        assert_rmse_vs_likelihood_record(unsorted)

    wrong_length = copy.deepcopy(record)
    wrong_length["rmse_vs_likelihood"]["pareto_efficient"] = [True, False]
    with pytest.raises(AssertionError, match="equal length"):
        assert_rmse_vs_likelihood_record(wrong_length)

    single_seed = copy.deepcopy(record)
    single_seed["rmse_vs_likelihood"]["num_seeds"][1] = 1
    with pytest.raises(AssertionError, match="num_seeds"):
        assert_rmse_vs_likelihood_record(single_seed)

    wrong_mse_cost = copy.deepcopy(record)
    wrong_mse_cost["rmse_vs_likelihood"][
        "mse_times_likelihood_evaluations"
    ][1] = 2.0
    with pytest.raises(AssertionError, match=r"rmse_logZ\*\*2"):
        assert_rmse_vs_likelihood_record(wrong_mse_cost)

    wrong_dominance = copy.deepcopy(record)
    wrong_dominance["rmse_vs_likelihood"]["dominance_summary"][
        "dominated_indices"
    ] = [2]
    with pytest.raises(AssertionError, match="dominated_indices"):
        assert_rmse_vs_likelihood_record(wrong_dominance)


def test_posterior_wasserstein_schema_keeps_posterior_metric_separate():
    record = _posterior_wasserstein_record()
    assert_posterior_wasserstein_record(record)

    single_seed = copy.deepcopy(record)
    single_seed["posterior_wasserstein"]["num_seeds"] = 1
    with pytest.raises(AssertionError, match="num_seeds"):
        assert_posterior_wasserstein_record(single_seed)

    mixed_record = copy.deepcopy(record)
    mixed_record["posterior_wasserstein"]["rho_g"] = [1.0]
    with pytest.raises(AssertionError, match="Forbidden metric keys"):
        assert_posterior_wasserstein_record(mixed_record)


def test_performance_guardrail_schema_records_threshold_and_outcome():
    record = _performance_guardrail_record()
    assert_performance_guardrail_record(record)

    missing_rationale = copy.deepcopy(record)
    del missing_rationale["performance_guardrail"]["rationale"]
    with pytest.raises(AssertionError, match="rationale"):
        assert_performance_guardrail_record(missing_rationale)

    unknown_guardrail = copy.deepcopy(record)
    unknown_guardrail["performance_guardrail"]["name"] = "single_worker_smoke"
    with pytest.raises(AssertionError, match="guardrail name"):
        assert_performance_guardrail_record(unknown_guardrail)


def test_performance_guardrail_suite_covers_ticket_hot_paths():
    names = [
        "block_construction",
        "shrinkage_sampling",
        "phantom_counting",
        "rho_bootstrap",
        "trajectories",
        "serialization",
        "worker_task_latency",
    ]
    records = []
    for idx, name in enumerate(names):
        record = _performance_guardrail_record()
        record["metadata"]["seed"] = idx
        record["performance_guardrail"]["name"] = name
        record["performance_guardrail"]["rationale"] = (
            f"{name} hot-path guardrail for release timing history."
        )
        records.append(record)

    assert_performance_guardrail_suite(records)

    missing = records[:-1]
    with pytest.raises(AssertionError, match="worker_task_latency"):
        assert_performance_guardrail_suite(missing)


def test_timing_history_is_append_only():
    previous = [_timing_row(seed=11, seconds=0.21)]
    next_history = [previous[0], _timing_row(seed=12, seconds=0.19)]
    assert_timing_history_append_only(previous, next_history)

    modified_history = [copy.deepcopy(previous[0])]
    modified_history[0]["timings"]["block_construction_seconds"] = 0.18
    with pytest.raises(AssertionError, match="must not change"):
        assert_timing_history_append_only(previous, modified_history)


def test_posterior_quality_schema_rejects_evidence_calibration_metrics():
    record = _posterior_record()
    assert_posterior_quality_record(record)

    single_seed = copy.deepcopy(record)
    single_seed["posterior_quality"]["num_seeds"] = 1
    with pytest.raises(AssertionError, match="num_seeds"):
        assert_posterior_quality_record(single_seed)

    mixed_record = copy.deepcopy(record)
    mixed_record["posterior_quality"]["z_logZ"] = 0.1
    with pytest.raises(AssertionError, match="Forbidden metric keys"):
        assert_posterior_quality_record(mixed_record)


def test_rollup_keeps_evidence_and_posterior_records_in_separate_sections():
    rollup = {
        "evidence_calibration": [_evidence_record()],
        "posterior_quality": [_posterior_record()],
    }
    assert_rollup_separates_evidence_and_posterior(rollup)

    combined_rollup = {
        "metrics": [_evidence_record(), _posterior_record()],
    }
    with pytest.raises(AssertionError, match="separate"):
        assert_rollup_separates_evidence_and_posterior(combined_rollup)
