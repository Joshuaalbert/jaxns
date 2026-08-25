import json

import numpy as np
import pytest

from benchmarks.v2_v3.io import completed_seeds
from benchmarks.v2_v3.posterior import REFERENCES, posterior_diagnostics
from benchmarks.v2_v3.schema import STANDARD_CASES, validate_release_matrix


def _release_record(
        implementation: str,
        case: str,
        conditioning: str,
        seed: int,
) -> dict:
    phantoms = conditioning == "phantom"
    case_index = STANDARD_CASES.index(case)
    return {
        "implementation": implementation,
        "source_id": ("2" if implementation == "v2" else "3") * 40,
        "case": case,
        "phantoms": phantoms,
        "conditioning": conditioning,
        "seed": seed,
        "truth_log_Z": -float(case_index),
        "ndims": float(case_index + 1),
        "root_degree": float(30 * (case_index + 1)),
        "replacement_width": float(10 * (case_index + 1)),
        "num_slices": float(5 * (case_index + 1)),
        "num_retained_phantoms": float(case_index + 1) if phantoms else 0.0,
        "dlogZ": float(np.log1p(1e-3)),
        "lower_s": 0.3,
        "compile_s": 0.4,
        "log_Z_mean": -float(case_index),
        "log_Z_uncert": 0.1,
        "log_Z_error": 0.0,
        "mc_log_Z_mean": -float(case_index),
        "mc_log_Z_std": 0.1,
        "mc_log_Z_error": 0.0,
        "mc_z_score": 0.0,
        "run_s": 1.0,
        "result_s": 0.1,
        "mc_s": 0.2,
        "classic_samples": 90.0,
        "phantom_samples": 10.0 if phantoms else 0.0,
        "likelihood_evaluations": 100.0,
        "ess": 20.0,
        "process_peak_rss_kib": 1000.0,
        "environment": {
            "jaxns_distribution_version": (
                "2.6.9" if implementation == "v2" else "3.0.0"
            ),
            "jaxns_module": f"/source-{implementation}/jaxns/__init__.py",
            "jax_version": "0.10.0",
            "jaxlib_version": "0.10.0",
            "backend": "cpu",
            "device": "TFRT_CPU_0",
            "x64": True,
            "python": "3.12.9",
            "platform": "test-platform",
        },
        **({
            "posterior_mean_rmse": 0.01,
        } if case in {
            "basic_mvn",
            "spike_slab",
            "spike_slab10",
            "weak_curved_mvn8",
            "weak_curved_spike_slab8",
            "weak_curved_spike_slab10",
        } else {}),
        **({
            "posterior_mode_weights": [0.4, 0.6],
            "posterior_mode_weights_true": [0.4, 0.6],
            "posterior_mode_weight_max_abs_error": 0.0,
            "posterior_missed_mode_count": 0,
            "posterior_incorrect_mode_weight_count": 0,
        } if case in {
            "spike_slab",
            "spike_slab10",
            "weak_curved_spike_slab8",
            "weak_curved_spike_slab10",
        } else {}),
        **({
            "phantom_gate_active_blocks": 1 if phantoms else 0,
            "phantom_gate_active_fraction": 0.5 if phantoms else 0.0,
            "phantom_kish_median_active": 20.0 if phantoms else 0.0,
        } if implementation == "v3" else {}),
    }


def _one_seed_release_matrix() -> list[dict]:
    return [
        _release_record(implementation, case, conditioning, seed=0)
        for implementation in ("v2", "v3")
        for case in STANDARD_CASES
        for conditioning in ("classic", "phantom")
    ]


def test_release_benchmark_schema_accepts_complete_matched_matrix() -> None:
    validate_release_matrix(
        _one_seed_release_matrix(),
        expected_seeds={0},
    )


def test_release_benchmark_schema_rejects_duplicates_and_setting_drift() -> None:
    records = _one_seed_release_matrix()
    with pytest.raises(ValueError, match="duplicate seeds"):
        validate_release_matrix(
            records + [dict(records[0])],
            expected_seeds={0},
        )

    drifted = [dict(record) for record in records]
    drifted[-1]["root_degree"] += 1
    with pytest.raises(ValueError, match="Matched field root_degree differs"):
        validate_release_matrix(drifted, expected_seeds={0})

    mixed_source = [dict(record) for record in records]
    mixed_source[-1]["source_id"] = "4" * 40
    with pytest.raises(ValueError, match="v3 mixes source commits"):
        validate_release_matrix(mixed_source, expected_seeds={0})


def test_release_benchmark_output_resumes_only_matching_groups(tmp_path) -> None:
    output = tmp_path / "v3_basic_classic.jsonl"
    identity = {
        "implementation": "v3",
        "source_id": "3" * 40,
        "case": "basic",
        "phantoms": False,
        "seed": 4,
    }
    output.write_text(json.dumps(identity) + "\n", encoding="utf-8")
    assert completed_seeds(
        output,
        implementation="v3",
        source_id="3" * 40,
        case="basic",
        phantoms=False,
    ) == {4}
    with pytest.raises(ValueError, match="belongs to"):
        completed_seeds(
            output,
            implementation="v2",
            source_id="2" * 40,
            case="basic",
            phantoms=False,
        )


def test_release_benchmark_posterior_reference_reports_mode_errors() -> None:
    reference = REFERENCES["spike_slab"]
    samples = reference.component_means
    diagnostics = posterior_diagnostics(
        "spike_slab",
        {"x": samples},
        np.log(np.asarray([0.5, 0.5])),
    )
    assert np.isfinite(diagnostics["posterior_mean_rmse"])
    assert np.isfinite(
        diagnostics["posterior_mode_weight_max_abs_error"]
    )
    np.testing.assert_allclose(
        np.sum(diagnostics["posterior_mode_weights"]),
        1.0,
    )
    np.testing.assert_allclose(
        np.sum(diagnostics["posterior_mode_weights_true"]),
        1.0,
    )
