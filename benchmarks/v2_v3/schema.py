"""Validation for maintained release-benchmark JSONL artifacts."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

STANDARD_CASES = (
    "basic",
    "basic2",
    "basic3",
    "plateau",
    "basic_mvn",
    "spike_slab",
    "spike_slab10",
    "weak_curved_mvn8",
    "weak_curved_spike_slab8",
    "weak_curved_spike_slab10",
)

REQUIRED_FIELDS = (
    "implementation",
    "source_id",
    "case",
    "phantoms",
    "conditioning",
    "seed",
    "truth_log_Z",
    "ndims",
    "root_degree",
    "replacement_width",
    "num_slices",
    "num_retained_phantoms",
    "dlogZ",
    "lower_s",
    "compile_s",
    "log_Z_mean",
    "log_Z_uncert",
    "log_Z_error",
    "mc_log_Z_mean",
    "mc_log_Z_std",
    "mc_log_Z_error",
    "mc_z_score",
    "run_s",
    "result_s",
    "mc_s",
    "classic_samples",
    "phantom_samples",
    "likelihood_evaluations",
    "ess",
    "process_peak_rss_kib",
    "environment",
)

FINITE_FIELDS = (
    "truth_log_Z",
    "ndims",
    "root_degree",
    "replacement_width",
    "num_slices",
    "num_retained_phantoms",
    "dlogZ",
    "lower_s",
    "compile_s",
    "log_Z_mean",
    "log_Z_uncert",
    "log_Z_error",
    "mc_log_Z_mean",
    "mc_log_Z_std",
    "mc_log_Z_error",
    "mc_z_score",
    "run_s",
    "result_s",
    "mc_s",
    "classic_samples",
    "phantom_samples",
    "likelihood_evaluations",
    "ess",
    "process_peak_rss_kib",
)

ENVIRONMENT_FIELDS = (
    "jaxns_distribution_version",
    "jaxns_module",
    "jax_version",
    "jaxlib_version",
    "backend",
    "device",
    "x64",
    "python",
    "platform",
)

POSTERIOR_CASES = {
    "basic_mvn",
    "spike_slab",
    "spike_slab10",
    "weak_curved_mvn8",
    "weak_curved_spike_slab8",
    "weak_curved_spike_slab10",
}

MULTIMODAL_CASES = {
    "spike_slab",
    "spike_slab10",
    "weak_curved_spike_slab8",
    "weak_curved_spike_slab10",
}


def load_records(paths: list[Path]) -> list[dict]:
    """Load JSON records while retaining file/line context on errors."""
    records = []
    for path in paths:
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}."
                ) from error
    return records


def validate_record(record: dict, *, exact_source: bool) -> None:
    """Validate one record before it contributes to release evidence."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Benchmark record is missing fields: {missing}.")
    if record["implementation"] not in ("v2", "v3"):
        raise ValueError("implementation must be 'v2' or 'v3'.")
    expected_conditioning = "phantom" if record["phantoms"] else "classic"
    if record["conditioning"] != expected_conditioning:
        raise ValueError(
            "conditioning must agree with the recorded phantom configuration."
        )
    if record["case"] not in STANDARD_CASES:
        raise ValueError(f"Unknown standard problem {record['case']!r}.")
    if not isinstance(record["phantoms"], bool):
        raise TypeError("phantoms must be a boolean.")
    if isinstance(record["seed"], bool) or int(record["seed"]) != record["seed"]:
        raise ValueError("seed must be an integer.")
    for field in FINITE_FIELDS:
        if not math.isfinite(float(record[field])):
            raise ValueError(f"{field} must be finite for every release run.")
    if float(record["log_Z_uncert"]) < 0 or float(record["mc_log_Z_std"]) < 0:
        raise ValueError("Reported uncertainties must be non-negative.")
    if int(record["ndims"]) <= 0 or int(record["root_degree"]) <= 0:
        raise ValueError("ndims and root_degree must be positive.")
    if int(record["num_slices"]) <= 0 or float(record["dlogZ"]) <= 0:
        raise ValueError("num_slices and dlogZ must be positive.")
    environment = record["environment"]
    if not isinstance(environment, dict):
        raise TypeError("environment must be a metadata object.")
    missing_environment = [
        field for field in ENVIRONMENT_FIELDS if field not in environment
    ]
    if missing_environment:
        raise ValueError(
            "Benchmark environment is missing fields: "
            f"{missing_environment}."
        )
    if record["case"] in POSTERIOR_CASES and (
            "posterior_mean_rmse" not in record
            or not math.isfinite(float(record["posterior_mean_rmse"]))
    ):
        raise ValueError(
            "Gaussian reference cases require finite posterior-mean RMSE."
        )
    if record["case"] in MULTIMODAL_CASES:
        mode_fields = (
            "posterior_mode_weights",
            "posterior_mode_weights_true",
            "posterior_mode_weight_max_abs_error",
            "posterior_missed_mode_count",
            "posterior_incorrect_mode_weight_count",
        )
        missing_modes = [field for field in mode_fields if field not in record]
        if missing_modes:
            raise ValueError(
                f"Multimodal reference case is missing fields: {missing_modes}."
            )
        mode_weights = (
            list(record["posterior_mode_weights"])
            + list(record["posterior_mode_weights_true"])
        )
        if not mode_weights or not all(
                math.isfinite(float(weight)) and float(weight) >= 0.0
                for weight in mode_weights
        ):
            raise ValueError("Posterior mode weights must be finite and non-negative.")
        if not math.isfinite(
                float(record["posterior_mode_weight_max_abs_error"])
        ):
            raise ValueError("Posterior mode-weight error must be finite.")
    if record["implementation"] == "v3":
        phantom_fields = (
            "phantom_gate_active_blocks",
            "phantom_gate_active_fraction",
            "phantom_kish_median_active",
        )
        missing_phantom = [
            field for field in phantom_fields if field not in record
        ]
        if missing_phantom:
            raise ValueError(
                "V3 release record is missing phantom eligibility fields: "
                f"{missing_phantom}."
            )
        gate_fraction = float(record["phantom_gate_active_fraction"])
        kish_count = float(record["phantom_kish_median_active"])
        if not math.isfinite(gate_fraction) or not 0.0 <= gate_fraction <= 1.0:
            raise ValueError("Phantom gate fraction must be finite and in [0, 1].")
        if not math.isfinite(kish_count) or kish_count < 0.0:
            raise ValueError("Phantom Kish count must be finite and non-negative.")
    if exact_source and re.fullmatch(r"[0-9a-f]{40}", record["source_id"]) is None:
        raise ValueError(
            "Release records require a full 40-character source commit."
        )


def validate_release_matrix(
        records: list[dict],
        *,
        expected_seeds: set[int] | None = None,
) -> None:
    """Validate complete v2/v3, problem, conditioning, and seed coverage."""
    if expected_seeds is None:
        expected_seeds = set(range(30))
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        validate_record(record, exact_source=True)
        key = (
            record["implementation"],
            record["case"],
            record["conditioning"],
        )
        groups[key].append(record)

    expected_groups = {
        (implementation, case, conditioning)
        for implementation in ("v2", "v3")
        for case in STANDARD_CASES
        for conditioning in ("classic", "phantom")
    }
    if set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        extra = sorted(set(groups) - expected_groups)
        raise ValueError(
            f"Release matrix group mismatch; missing={missing}, extra={extra}."
        )

    for key, rows in groups.items():
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Release group {key} contains duplicate seeds.")
        if set(seeds) != expected_seeds:
            raise ValueError(
                f"Release group {key} has seeds {sorted(seeds)}, "
                f"expected {sorted(expected_seeds)}."
            )
    # A release line has one source commit across the entire matrix, not one
    # independently chosen commit per problem. Otherwise a complete-looking
    # matrix could silently combine behavior from multiple candidates.
    for implementation in ("v2", "v3"):
        source_ids = {
            row["source_id"]
            for key, rows in groups.items()
            if key[0] == implementation
            for row in rows
        }
        if len(source_ids) != 1:
            raise ValueError(
                f"Release line {implementation} mixes source commits "
                f"{source_ids}."
            )

    # The scientific stopping condition and root/slice effort are the matched
    # contract. Replacement width is intentionally reported rather than forced
    # equal because v2 and v3 express parallel work differently.
    for case in STANDARD_CASES:
        rows = [
            row
            for key, group_rows in groups.items()
            if key[1] == case
            for row in group_rows
        ]
        for field in ("truth_log_Z", "root_degree", "num_slices", "dlogZ"):
            values = {float(row[field]) for row in rows}
            if len(values) != 1:
                raise ValueError(
                    f"Matched field {field} differs for {case}: "
                    f"{sorted(values)}."
                )

    # Hardware, numeric precision, and the JAX runtime are part of the matched
    # experiment. Source paths and JAXNS versions intentionally differ.
    for field in (
            "jax_version",
            "jaxlib_version",
            "backend",
            "device",
            "x64",
            "python",
            "platform",
    ):
        values = {
            json.dumps(record["environment"][field], sort_keys=True)
            for record in records
        }
        if len(values) != 1:
            raise ValueError(
                f"Matched environment field {field} differs: {sorted(values)}."
            )
