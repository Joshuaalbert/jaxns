"""Schema checks for v3 validation benchmark records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


CALIBRATION_LOGZ_CONVENTION = "mean_logZ_samples"
CALIBRATION_SIGMA_CONVENTION = "sample_sd_logZ_samples_ddof1"

REQUIRED_METADATA_FIELDS = frozenset(
    {
        "method_setting",
        "seed",
        "problem",
        "likelihood_evaluations",
        "wall_clock_seconds",
        "worker_count",
    }
)
SOFTWARE_ID_FIELDS = frozenset({"software_version", "commit"})
BASE_EVIDENCE_CALIBRATION_FIELDS = frozenset(
    {
        "summary_convention",
        "sigma_convention",
        "logZ_ref",
        "hat_logZ",
        "sigma_logZ",
        "z_logZ",
        "bias_logZ",
        "rmse_logZ",
        "reported_uncertainty_logZ",
        "empirical_uncertainty_logZ",
        "expectation_logZ",
        "mc_shrinkage_logZ",
    }
)
PHANTOM_DIAGNOSTIC_FIELDS = frozenset(
    {
        "kish_participating_cluster_counts",
        "phantom_gate_active",
        "phantom_A_g",
        "phantom_B_g",
        "phantom_E_g",
        "phantom_R_g",
        "C_min",
    }
)
DEPRECATED_RHO_DIAGNOSTIC_FIELDS = frozenset(
    {
        "rho_g",
        "rho_fit",
        "rho_samples",
        "rho_values",
        "rho_eta_samples",
        "rho_bootstrap",
    }
)
EVIDENCE_CALIBRATION_FIELDS = (
    BASE_EVIDENCE_CALIBRATION_FIELDS | PHANTOM_DIAGNOSTIC_FIELDS
)
CALIBRATION_ROLLUP_FIELDS = frozenset(
    {
        "problem",
        "method",
        "method_setting",
        "num_seeds",
        "mean_z_logZ",
        "sd_z_logZ",
    }
)
PLATEAU_EQUALITY_FIELDS = frozenset(
    {
        "likelihood_level",
        "equality_mass_ref",
        "hat_equality_mass",
        "equality_mass_error",
        "per_sample_equality_mass",
    }
)
POSTERIOR_QUALITY_FIELDS = frozenset(
    {
        "num_seeds",
        "wasserstein_mc",
        "reference_sample_count",
        "posterior_sample_count",
        "effective_sample_size",
        "likelihood_evaluations_per_effective_sample",
    }
)
RMSE_VS_LIKELIHOOD_FIELDS = frozenset(
    {
        "likelihood_evaluations",
        "num_seeds",
        "rmse_logZ",
        "mse_times_likelihood_evaluations",
        "pareto_efficient",
        "dominance_summary",
    }
)
DOMINANCE_SUMMARY_FIELDS = frozenset(
    {
        "best_index",
        "best_likelihood_evaluations",
        "best_mse_times_likelihood_evaluations",
        "dominated_indices",
    }
)
POSTERIOR_WASSERSTEIN_FIELDS = frozenset(
    {
        "num_seeds",
        "wasserstein",
        "reference_sample_count",
        "posterior_sample_count",
        "effective_sample_size",
        "likelihood_evaluations_per_effective_sample",
        "dimension",
    }
)
PERFORMANCE_GUARDRAIL_FIELDS = frozenset(
    {
        "name",
        "observed_seconds",
        "threshold_seconds",
        "comparison",
        "passed",
        "rationale",
    }
)
REQUIRED_PERFORMANCE_GUARDRAIL_NAMES = frozenset(
    {
        "block_construction",
        "shrinkage_sampling",
        "phantom_counting",
        "gamma_phantom_conditioning",
        "trajectories",
        "serialization",
        "worker_task_latency",
    }
)
TIMING_HISTORY_FIELDS = frozenset(
    {
        "metadata",
        "timings",
    }
)
POSTERIOR_ONLY_FIELDS = frozenset(
    {
        "posterior_quality",
        "wasserstein",
        "wasserstein_mc",
        "posterior_wasserstein",
        "reference_posterior_samples",
        "posterior_samples",
    }
)
EVIDENCE_ONLY_FIELDS = frozenset(
    {
        "evidence_calibration",
        "z_logZ",
        "hat_logZ",
        "sigma_logZ",
        "logZ_ref",
        "kish_participating_cluster_counts",
        "phantom_gate_active",
        "phantom_A_g",
        "phantom_B_g",
        "phantom_E_g",
        "phantom_R_g",
        "C_min",
        *DEPRECATED_RHO_DIAGNOSTIC_FIELDS,
    }
)


def compute_calibration_summary(
        log_Z_samples: Sequence[float],
        *,
        logZ_ref: float,
) -> dict[str, Any]:
    """Return the validation convention for `hat logZ` and `sigma`."""
    samples = np.asarray(log_Z_samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("log_Z_samples must be a one-dimensional sequence.")
    if samples.size < 2:
        raise ValueError("At least two log_Z samples are required for sigma.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("log_Z_samples must be finite.")
    hat_logZ = float(np.mean(samples))
    sigma_logZ = float(np.std(samples, ddof=1))
    if not np.isfinite(logZ_ref):
        raise ValueError("logZ_ref must be finite.")
    if sigma_logZ <= 0.0:
        raise ValueError("sigma_logZ must be positive for z-score reporting.")
    bias = hat_logZ - float(logZ_ref)
    return {
        "summary_convention": CALIBRATION_LOGZ_CONVENTION,
        "sigma_convention": CALIBRATION_SIGMA_CONVENTION,
        "logZ_ref": float(logZ_ref),
        "hat_logZ": hat_logZ,
        "sigma_logZ": sigma_logZ,
        "z_logZ": bias / sigma_logZ,
        "bias_logZ": bias,
    }


def assert_benchmark_metadata(metadata: Mapping[str, Any]) -> None:
    _require_mapping(metadata, "metadata")
    _assert_required_keys(metadata, REQUIRED_METADATA_FIELDS, "metadata")
    if not any(metadata.get(field) for field in SOFTWARE_ID_FIELDS):
        raise AssertionError(
            "metadata must include a non-empty software_version or commit."
        )
    _assert_method_setting(metadata["method_setting"])
    _assert_non_empty_string(metadata["problem"], "metadata.problem")
    _assert_integral(metadata["seed"], "metadata.seed", minimum=0)
    _assert_integral(
        metadata["likelihood_evaluations"],
        "metadata.likelihood_evaluations",
        minimum=1,
    )
    wall_clock = metadata["wall_clock_seconds"]
    if wall_clock is not None:
        _assert_real(wall_clock, "metadata.wall_clock_seconds", minimum=0.0)
    _assert_integral(metadata["worker_count"], "metadata.worker_count", 1)


def assert_evidence_calibration_record(record: Mapping[str, Any]) -> None:
    _require_mapping(record, "record")
    _assert_forbidden_keys(record, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    if record.get("metric_family") != "evidence_calibration":
        raise AssertionError(
            "record.metric_family must be evidence_calibration."
        )
    assert_benchmark_metadata(record.get("metadata"))
    calibration = _require_mapping(
        record.get("evidence_calibration"),
        "record.evidence_calibration",
    )
    _assert_required_keys(
        calibration,
        BASE_EVIDENCE_CALIBRATION_FIELDS,
        "record.evidence_calibration",
    )
    if calibration["summary_convention"] != CALIBRATION_LOGZ_CONVENTION:
        raise AssertionError("Unexpected hat logZ summary convention.")
    if calibration["sigma_convention"] != CALIBRATION_SIGMA_CONVENTION:
        raise AssertionError("Unexpected sigma_logZ convention.")
    for key in BASE_EVIDENCE_CALIBRATION_FIELDS - {
        "summary_convention",
        "sigma_convention",
    }:
        _assert_real(calibration[key], f"record.evidence_calibration.{key}")
    has_phantom_diagnostics = any(
        key in calibration
        for key in PHANTOM_DIAGNOSTIC_FIELDS
    )
    if _method_requires_phantom_diagnostics(
        record["metadata"]["method_setting"],
    ):
        _assert_required_keys(
            calibration,
            PHANTOM_DIAGNOSTIC_FIELDS,
            "record.evidence_calibration",
        )
        has_phantom_diagnostics = True
    if has_phantom_diagnostics:
        _assert_required_keys(
            calibration,
            PHANTOM_DIAGNOSTIC_FIELDS,
            "record.evidence_calibration",
        )
        _assert_phantom_diagnostics(calibration)
    _assert_forbidden_keys(record, POSTERIOR_ONLY_FIELDS)


def assert_plateau_equality_record(record: Mapping[str, Any]) -> None:
    assert_evidence_calibration_record(record)
    plateau = _require_mapping(
        record.get("plateau_equality"),
        "record.plateau_equality",
    )
    _assert_required_keys(
        plateau,
        PLATEAU_EQUALITY_FIELDS,
        "record.plateau_equality",
    )
    for key in PLATEAU_EQUALITY_FIELDS:
        _assert_real(plateau[key], f"record.plateau_equality.{key}")
    if not 0.0 <= plateau["equality_mass_ref"] <= 1.0:
        raise AssertionError("equality_mass_ref must be a probability.")
    if not 0.0 <= plateau["hat_equality_mass"] <= 1.0:
        raise AssertionError("hat_equality_mass must be a probability.")
    if not 0.0 <= plateau["per_sample_equality_mass"] <= 1.0:
        raise AssertionError("per_sample_equality_mass must be a probability.")


def assert_posterior_quality_record(record: Mapping[str, Any]) -> None:
    _require_mapping(record, "record")
    _assert_forbidden_keys(record, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    if record.get("metric_family") != "posterior_quality":
        raise AssertionError("record.metric_family must be posterior_quality.")
    assert_benchmark_metadata(record.get("metadata"))
    posterior = _require_mapping(
        record.get("posterior_quality"),
        "record.posterior_quality",
    )
    _assert_required_keys(
        posterior,
        POSTERIOR_QUALITY_FIELDS,
        "record.posterior_quality",
    )
    _assert_integral(
        posterior["num_seeds"],
        "record.posterior_quality.num_seeds",
        minimum=2,
    )
    _assert_real(
        posterior["wasserstein_mc"],
        "record.posterior_quality.wasserstein_mc",
        minimum=0.0,
    )
    _assert_integral(
        posterior["reference_sample_count"],
        "record.posterior_quality.reference_sample_count",
        minimum=1,
    )
    _assert_integral(
        posterior["posterior_sample_count"],
        "record.posterior_quality.posterior_sample_count",
        minimum=1,
    )
    _assert_positive_real(
        posterior["effective_sample_size"],
        "record.posterior_quality.effective_sample_size",
    )
    _assert_positive_real(
        posterior["likelihood_evaluations_per_effective_sample"],
        "record.posterior_quality.likelihood_evaluations_per_effective_sample",
    )
    _assert_likelihood_evaluations_per_ess(
        record["metadata"],
        posterior["effective_sample_size"],
        posterior["likelihood_evaluations_per_effective_sample"],
        "record.posterior_quality.likelihood_evaluations_per_effective_sample",
    )
    _assert_forbidden_keys(record, EVIDENCE_ONLY_FIELDS)


def assert_calibration_table(records: Sequence[Mapping[str, Any]]) -> None:
    rows = _require_sequence(records, "calibration_table")
    if not rows:
        raise AssertionError(
            "calibration_table must contain at least one row."
        )
    seen_keys = set()
    for idx, record in enumerate(rows):
        assert_evidence_calibration_record(record)
        metadata = record["metadata"]
        key = _problem_method_seed_key(metadata)
        if key in seen_keys:
            raise AssertionError(
                "Duplicate problem/method_setting/seed in calibration_table: "
                f"problem={key[0]!r}, method_setting={key[1]!r}, "
                f"seed={key[2]!r}."
            )
        seen_keys.add(key)
        if record["metadata"]["wall_clock_seconds"] is None:
            raise AssertionError(
                f"calibration_table[{idx}] must record wall_clock_seconds."
            )


def assert_calibration_rollup_table(
        records: Sequence[Mapping[str, Any]],
        rollups: Sequence[Mapping[str, Any]],
) -> None:
    rows = _require_sequence(records, "calibration_table")
    assert_calibration_table(rows)
    rollup_rows = _require_sequence(rollups, "calibration_rollups")
    if not rollup_rows:
        raise AssertionError(
            "calibration_rollups must contain at least one row."
        )

    z_scores_by_group: dict[tuple[str, Any], list[float]] = {}
    for record in rows:
        metadata = record["metadata"]
        group_key = _problem_method_key(metadata)
        z_score = float(record["evidence_calibration"]["z_logZ"])
        z_scores_by_group.setdefault(group_key, []).append(z_score)

    seen_groups = set()
    for rollup in rollup_rows:
        _require_mapping(rollup, "calibration_rollup")
        _assert_required_keys(
            rollup,
            CALIBRATION_ROLLUP_FIELDS,
            "calibration_rollup",
        )
        _assert_non_empty_string(rollup["problem"], "calibration_rollup.problem")
        _assert_non_empty_string(rollup["method"], "calibration_rollup.method")
        rollup_method_setting = _require_mapping(
            rollup["method_setting"],
            "calibration_rollup.method_setting",
        )
        _assert_method_setting(rollup_method_setting)
        if rollup["method"] != rollup_method_setting["method"]:
            raise AssertionError(
                "calibration_rollup.method must match "
                "calibration_rollup.method_setting.method."
            )
        _assert_integral(
            rollup["num_seeds"],
            "calibration_rollup.num_seeds",
            minimum=2,
        )
        _assert_real(rollup["mean_z_logZ"], "calibration_rollup.mean_z_logZ")
        _assert_real(
            rollup["sd_z_logZ"],
            "calibration_rollup.sd_z_logZ",
            minimum=0.0,
        )

        group_key = (
            rollup["problem"],
            _method_setting_key(rollup_method_setting),
        )
        if group_key in seen_groups:
            raise AssertionError(
                "Duplicate calibration rollup for "
                f"problem={group_key[0]!r}, method_setting={group_key[1]!r}."
            )
        seen_groups.add(group_key)

        if group_key not in z_scores_by_group:
            raise AssertionError(
                "calibration_rollup has no matching calibration rows: "
                f"problem={group_key[0]!r}, method_setting={group_key[1]!r}."
            )
        z_scores = np.asarray(z_scores_by_group[group_key], dtype=float)
        if z_scores.size < 2:
            raise AssertionError(
                "calibration_rollup requires at least two seeds for "
                f"problem={group_key[0]!r}, method_setting={group_key[1]!r}."
            )

        expected_mean = float(np.mean(z_scores))
        expected_sd = float(np.std(z_scores, ddof=1))
        if rollup["num_seeds"] != z_scores.size:
            raise AssertionError(
                "calibration_rollup.num_seeds does not match calibration rows."
            )
        if not np.isclose(
            float(rollup["mean_z_logZ"]),
            expected_mean,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(
                "calibration_rollup.mean_z_logZ does not match calibration rows."
            )
        if not np.isclose(
            float(rollup["sd_z_logZ"]),
            expected_sd,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(
                "calibration_rollup.sd_z_logZ does not match calibration rows."
            )

    missing_groups = sorted(
        repr(group)
        for group in set(z_scores_by_group).difference(seen_groups)
    )
    if missing_groups:
        raise AssertionError(
            f"Missing calibration rollups for groups: {missing_groups}."
        )


def assert_rmse_vs_likelihood_record(record: Mapping[str, Any]) -> None:
    _require_mapping(record, "record")
    _assert_forbidden_keys(record, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    if record.get("metric_family") != "rmse_vs_likelihood":
        raise AssertionError(
            "record.metric_family must be rmse_vs_likelihood."
        )
    assert_benchmark_metadata(record.get("metadata"))
    body = _require_mapping(
        record.get("rmse_vs_likelihood"),
        "record.rmse_vs_likelihood",
    )
    _assert_required_keys(
        body,
        RMSE_VS_LIKELIHOOD_FIELDS,
        "record.rmse_vs_likelihood",
    )
    evaluations = _assert_real_sequence(
        body["likelihood_evaluations"],
        "record.rmse_vs_likelihood.likelihood_evaluations",
        minimum=1.0,
    )
    num_seeds = _assert_integral_sequence(
        body["num_seeds"],
        "record.rmse_vs_likelihood.num_seeds",
        minimum=2,
    )
    rmse = _assert_real_sequence(
        body["rmse_logZ"],
        "record.rmse_vs_likelihood.rmse_logZ",
        minimum=0.0,
    )
    mse_cost = _assert_real_sequence(
        body["mse_times_likelihood_evaluations"],
        "record.rmse_vs_likelihood.mse_times_likelihood_evaluations",
        minimum=0.0,
    )
    pareto = _assert_bool_sequence(
        body["pareto_efficient"],
        "record.rmse_vs_likelihood.pareto_efficient",
    )
    if not (
        len(evaluations)
        == len(num_seeds)
        == len(rmse)
        == len(mse_cost)
        == len(pareto)
    ):
        raise AssertionError(
            "likelihood_evaluations, num_seeds, rmse_logZ, "
            "mse_times_likelihood_evaluations, and pareto_efficient "
            "must have equal length."
        )
    if any(right <= left for left, right in zip(evaluations, evaluations[1:])):
        raise AssertionError(
            "likelihood_evaluations must be strictly increasing."
        )
    expected_mse_cost = np.square(np.asarray(rmse, dtype=float)) * np.asarray(
        evaluations,
        dtype=float,
    )
    if not np.allclose(
        np.asarray(mse_cost, dtype=float),
        expected_mse_cost,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            "mse_times_likelihood_evaluations must equal "
            "rmse_logZ**2 * likelihood_evaluations."
        )
    _assert_dominance_summary(
        body["dominance_summary"],
        evaluations,
        mse_cost,
    )


def assert_posterior_wasserstein_record(record: Mapping[str, Any]) -> None:
    _require_mapping(record, "record")
    _assert_forbidden_keys(record, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    if record.get("metric_family") != "posterior_wasserstein":
        raise AssertionError(
            "record.metric_family must be posterior_wasserstein."
        )
    assert_benchmark_metadata(record.get("metadata"))
    body = _require_mapping(
        record.get("posterior_wasserstein"),
        "record.posterior_wasserstein",
    )
    _assert_required_keys(
        body,
        POSTERIOR_WASSERSTEIN_FIELDS,
        "record.posterior_wasserstein",
    )
    _assert_integral(
        body["num_seeds"],
        "record.posterior_wasserstein.num_seeds",
        2,
    )
    _assert_real(
        body["wasserstein"],
        "record.posterior_wasserstein.wasserstein",
        0.0,
    )
    _assert_integral(
        body["reference_sample_count"],
        "record.posterior_wasserstein.reference_sample_count",
        1,
    )
    _assert_integral(
        body["posterior_sample_count"],
        "record.posterior_wasserstein.posterior_sample_count",
        1,
    )
    _assert_positive_real(
        body["effective_sample_size"],
        "record.posterior_wasserstein.effective_sample_size",
    )
    _assert_positive_real(
        body["likelihood_evaluations_per_effective_sample"],
        "record.posterior_wasserstein."
        "likelihood_evaluations_per_effective_sample",
    )
    _assert_likelihood_evaluations_per_ess(
        record["metadata"],
        body["effective_sample_size"],
        body["likelihood_evaluations_per_effective_sample"],
        "record.posterior_wasserstein."
        "likelihood_evaluations_per_effective_sample",
    )
    _assert_integral(
        body["dimension"],
        "record.posterior_wasserstein.dimension",
        1,
    )
    _assert_forbidden_keys(record, EVIDENCE_ONLY_FIELDS)


def assert_performance_guardrail_record(record: Mapping[str, Any]) -> None:
    _require_mapping(record, "record")
    _assert_forbidden_keys(record, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    if record.get("metric_family") != "performance_guardrail":
        raise AssertionError(
            "record.metric_family must be performance_guardrail."
        )
    assert_benchmark_metadata(record.get("metadata"))
    body = _require_mapping(
        record.get("performance_guardrail"),
        "record.performance_guardrail",
    )
    _assert_required_keys(
        body,
        PERFORMANCE_GUARDRAIL_FIELDS,
        "record.performance_guardrail",
    )
    _assert_non_empty_string(
        body["name"],
        "record.performance_guardrail.name",
    )
    if body["name"] in DEPRECATED_RHO_DIAGNOSTIC_FIELDS:
        raise AssertionError(
            f"Forbidden stale rho diagnostic {body['name']!r}."
        )
    if body["name"] not in REQUIRED_PERFORMANCE_GUARDRAIL_NAMES:
        raise AssertionError(
            "performance guardrail name must be one of "
            f"{sorted(REQUIRED_PERFORMANCE_GUARDRAIL_NAMES)}."
        )
    _assert_real(
        body["observed_seconds"],
        "record.performance_guardrail.observed_seconds",
        0.0,
    )
    _assert_real(
        body["threshold_seconds"],
        "record.performance_guardrail.threshold_seconds",
        0.0,
    )
    if body["comparison"] not in {"<=", "<"}:
        raise AssertionError(
            "performance guardrail comparison must be <= or <."
        )
    if not isinstance(body["passed"], bool):
        raise AssertionError("performance guardrail passed must be bool.")
    _assert_non_empty_string(
        body["rationale"],
        "record.performance_guardrail.rationale",
    )


def assert_performance_guardrail_suite(
        records: Sequence[Mapping[str, Any]],
) -> None:
    rows = _require_sequence(records, "performance_guardrails")
    seen_names = set()
    for record in rows:
        assert_performance_guardrail_record(record)
        name = record["performance_guardrail"]["name"]
        if name in seen_names:
            raise AssertionError(
                f"Duplicate performance guardrail for {name!r}."
            )
        seen_names.add(name)
    missing = REQUIRED_PERFORMANCE_GUARDRAIL_NAMES.difference(seen_names)
    if missing:
        raise AssertionError(
            f"Missing performance guardrails: {sorted(missing)}."
        )


def assert_timing_history_append_only(
        previous_history: Sequence[Mapping[str, Any]],
        next_history: Sequence[Mapping[str, Any]],
) -> None:
    previous = _require_sequence(previous_history, "previous_history")
    next_rows = _require_sequence(next_history, "next_history")
    if len(next_rows) < len(previous):
        raise AssertionError("Timing history must be append-only.")
    for idx, previous_row in enumerate(previous):
        if next_rows[idx] != previous_row:
            raise AssertionError(
                "Existing timing history rows must not change."
            )
    for row in next_rows:
        _assert_timing_history_row(row)


def assert_rollup_separates_evidence_and_posterior(
        rollup: Mapping[str, Any],
) -> None:
    _require_mapping(rollup, "rollup")
    if "metrics" in rollup:
        raise AssertionError(
            "Use separate evidence_calibration and posterior_quality sections."
        )
    evidence_records = _require_sequence(
        rollup.get("evidence_calibration"),
        "rollup.evidence_calibration",
    )
    posterior_records = _require_sequence(
        rollup.get("posterior_quality"),
        "rollup.posterior_quality",
    )
    for record in evidence_records:
        assert_evidence_calibration_record(record)
    for record in posterior_records:
        assert_posterior_quality_record(record)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{name} must be a mapping.")
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise AssertionError(f"{name} must be a sequence.")
    return value


def _assert_required_keys(
        value: Mapping[str, Any],
        required: frozenset[str],
        context: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise AssertionError(f"{context} missing keys: {sorted(missing)}.")


def _assert_method_setting(value: Any) -> None:
    method_setting = _require_mapping(value, "metadata.method_setting")
    _assert_non_empty_string(
        method_setting.get("method"),
        "metadata.method_setting.method",
    )


def _method_requires_phantom_diagnostics(
        method_setting: Mapping[str, Any],
) -> bool:
    method = str(method_setting.get("method", ""))
    return "phantom" in method


def _problem_method_key(metadata: Mapping[str, Any]) -> tuple[str, Any]:
    return (
        str(metadata["problem"]),
        _method_setting_key(metadata["method_setting"]),
    )


def _problem_method_seed_key(metadata: Mapping[str, Any]) -> tuple[str, Any, int]:
    problem, method_setting = _problem_method_key(metadata)
    return problem, method_setting, int(metadata["seed"])


def _method_setting_key(method_setting: Mapping[str, Any]) -> Any:
    return _freeze_key(method_setting)


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


def _assert_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{name} must be a non-empty string.")


def _assert_integral(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AssertionError(f"{name} must be an integer.")
    if value < minimum:
        raise AssertionError(f"{name} must be >= {minimum}.")


def _assert_real(
        value: Any,
        name: str,
        minimum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AssertionError(f"{name} must be a real number.")
    if not np.isfinite(float(value)):
        raise AssertionError(f"{name} must be finite.")
    if minimum is not None and value < minimum:
        raise AssertionError(f"{name} must be >= {minimum}.")


def _assert_positive_real(value: Any, name: str) -> None:
    _assert_real(value, name, minimum=0.0)
    if value <= 0.0:
        raise AssertionError(f"{name} must be > 0.")


def _assert_likelihood_evaluations_per_ess(
        metadata: Mapping[str, Any],
        effective_sample_size: Any,
        likelihood_evaluations_per_ess: Any,
        name: str,
) -> None:
    expected = (
        float(metadata["likelihood_evaluations"])
        / float(effective_sample_size)
    )
    if not np.isclose(
        float(likelihood_evaluations_per_ess),
        expected,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            f"{name} must equal metadata.likelihood_evaluations / "
            "effective_sample_size."
        )


def _assert_dominance_summary(
        value: Any,
        evaluations: Sequence[Any],
        mse_cost: Sequence[Any],
) -> None:
    summary = _require_mapping(
        value,
        "record.rmse_vs_likelihood.dominance_summary",
    )
    _assert_required_keys(
        summary,
        DOMINANCE_SUMMARY_FIELDS,
        "record.rmse_vs_likelihood.dominance_summary",
    )
    best_index = summary["best_index"]
    _assert_integral(
        best_index,
        "record.rmse_vs_likelihood.dominance_summary.best_index",
        0,
    )
    if best_index >= len(mse_cost):
        raise AssertionError(
            "record.rmse_vs_likelihood.dominance_summary.best_index "
            "is out of range."
        )
    _assert_real(
        summary["best_likelihood_evaluations"],
        "record.rmse_vs_likelihood.dominance_summary."
        "best_likelihood_evaluations",
        minimum=1.0,
    )
    _assert_real(
        summary["best_mse_times_likelihood_evaluations"],
        "record.rmse_vs_likelihood.dominance_summary."
        "best_mse_times_likelihood_evaluations",
        minimum=0.0,
    )
    if not np.isclose(
        summary["best_likelihood_evaluations"],
        evaluations[best_index],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            "dominance_summary.best_likelihood_evaluations must match "
            "the best MSE-cost row."
        )
    if not np.isclose(
        summary["best_mse_times_likelihood_evaluations"],
        mse_cost[best_index],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            "dominance_summary.best_mse_times_likelihood_evaluations "
            "must match the best MSE-cost row."
        )
    dominated_indices = _require_sequence(
        summary["dominated_indices"],
        "record.rmse_vs_likelihood.dominance_summary.dominated_indices",
    )
    for idx, item in enumerate(dominated_indices):
        _assert_integral(
            item,
            "record.rmse_vs_likelihood.dominance_summary."
            f"dominated_indices[{idx}]",
            0,
        )
    expected_dominated = [
        idx
        for idx, value in enumerate(mse_cost)
        if idx != best_index and value > mse_cost[best_index]
    ]
    if list(dominated_indices) != expected_dominated:
        raise AssertionError(
            "dominance_summary.dominated_indices must list rows with larger "
            "MSE * likelihood_evaluations than the best row."
        )


def _assert_phantom_diagnostics(calibration: Mapping[str, Any]) -> None:
    kish = _assert_real_sequence(
        calibration["kish_participating_cluster_counts"],
        "kish_participating_cluster_counts",
        minimum=0.0,
    )
    gate = _assert_bool_sequence(
        calibration["phantom_gate_active"],
        "phantom_gate_active",
    )
    counts_by_name = {
        "phantom_A_g": _assert_real_sequence(
            calibration["phantom_A_g"],
            "phantom_A_g",
            minimum=0.0,
        ),
        "phantom_B_g": _assert_real_sequence(
            calibration["phantom_B_g"],
            "phantom_B_g",
            minimum=0.0,
        ),
        "phantom_E_g": _assert_real_sequence(
            calibration["phantom_E_g"],
            "phantom_E_g",
            minimum=0.0,
        ),
        "phantom_R_g": _assert_real_sequence(
            calibration["phantom_R_g"],
            "phantom_R_g",
            minimum=0.0,
        ),
    }
    C_min = calibration["C_min"]
    _assert_positive_real(C_min, "C_min")
    expected_length = len(kish)
    if len(gate) != expected_length:
        raise AssertionError(
            "phantom_gate_active length must align with "
            "kish_participating_cluster_counts."
        )
    for name, values in counts_by_name.items():
        if len(values) != expected_length:
            raise AssertionError(
                f"{name} length must align with "
                "kish_participating_cluster_counts."
            )

    kish_array = np.asarray(kish, dtype=float)
    gate_array = np.asarray(gate, dtype=bool)
    expected_gate = kish_array >= float(C_min)
    if not np.array_equal(gate_array, expected_gate):
        raise AssertionError(
            "phantom_gate_active must match kish_participating_cluster_counts "
            ">= C_min."
        )
    A = np.asarray(counts_by_name["phantom_A_g"], dtype=float)
    B = np.asarray(counts_by_name["phantom_B_g"], dtype=float)
    E = np.asarray(counts_by_name["phantom_E_g"], dtype=float)
    R = np.asarray(counts_by_name["phantom_R_g"], dtype=float)
    if not np.allclose(R, A - B - E, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            "phantom_R_g must equal A_g - B_g - E_g; R_g derives from A."
        )


def _assert_rho_curve(value: Any, name: str) -> None:
    curve = _require_sequence(value, name)
    if not curve:
        raise AssertionError(f"{name} must not be empty.")
    for idx, rho in enumerate(curve):
        _assert_real(rho, f"{name}[{idx}]", minimum=0.0)
        if rho <= 0.0 or rho > 1.0:
            raise AssertionError(f"{name}[{idx}] must satisfy 0 < rho <= 1.")


def _assert_real_sequence(
        value: Any,
        name: str,
        minimum: float | None = None,
) -> Sequence[Any]:
    sequence = _require_sequence(value, name)
    if not sequence:
        raise AssertionError(f"{name} must not be empty.")
    for idx, item in enumerate(sequence):
        _assert_real(item, f"{name}[{idx}]", minimum=minimum)
    return sequence


def _assert_integral_sequence(
        value: Any,
        name: str,
        minimum: int,
) -> Sequence[Any]:
    sequence = _require_sequence(value, name)
    if not sequence:
        raise AssertionError(f"{name} must not be empty.")
    for idx, item in enumerate(sequence):
        _assert_integral(item, f"{name}[{idx}]", minimum)
    return sequence


def _assert_bool_sequence(value: Any, name: str) -> Sequence[Any]:
    sequence = _require_sequence(value, name)
    if not sequence:
        raise AssertionError(f"{name} must not be empty.")
    for idx, item in enumerate(sequence):
        if not isinstance(item, bool):
            raise AssertionError(f"{name}[{idx}] must be bool.")
    return sequence


def _assert_timing_history_row(row: Mapping[str, Any]) -> None:
    _require_mapping(row, "timing_history row")
    _assert_forbidden_keys(row, DEPRECATED_RHO_DIAGNOSTIC_FIELDS)
    _assert_required_keys(row, TIMING_HISTORY_FIELDS, "timing_history row")
    assert_benchmark_metadata(row["metadata"])
    timings = _require_mapping(row["timings"], "timing_history row.timings")
    if not timings:
        raise AssertionError("timing_history row.timings must not be empty.")
    for key, value in timings.items():
        _assert_non_empty_string(key, "timing name")
        _assert_real(value, f"timing_history row.timings.{key}", 0.0)


def _assert_forbidden_keys(
        value: Mapping[str, Any],
        forbidden: frozenset[str],
) -> None:
    present = sorted(key for key in forbidden if _contains_key(value, key))
    if present:
        raise AssertionError(f"Forbidden metric keys present: {present}.")


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(
            _contains_key(child, key)
            for child in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return any(_contains_key(child, key) for child in value)
    return False
