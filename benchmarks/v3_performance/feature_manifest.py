"""Shared Ticket 0020 feature manifest for v3 performance suites."""

from __future__ import annotations

import dataclasses


DEFAULT_ALLOCATION_TARGETS = (
    "uniform",
    "evidence_improving",
    "posterior_improving",
)
ALLOCATION_TARGETS = DEFAULT_ALLOCATION_TARGETS
STANDARD_PROBLEM_FIXTURES = (
    "basic_mvn",
    "spike_slab",
    "plateau",
)
STANDARD_PROBLEM_ROW_NAMES = STANDARD_PROBLEM_FIXTURES

SUITE_PURE_CORE = "pure_core"
SUITE_DISTRIBUTED = "distributed"
FEATURE_MANIFEST_SUITES = (
    SUITE_PURE_CORE,
    SUITE_DISTRIBUTED,
)

ROW_KIND_ACCURACY = "accuracy"
ROW_KIND_BENCHMARK = "benchmark"

@dataclasses.dataclass(frozen=True, slots=True)
class V3PerformanceFeatureRow:
    """One statistical feature row shared by pure-core and distributed suites."""

    row_id: str
    problem_fixture: str
    dimension: int
    reference_evidence: str
    reference_posterior: str
    seeds: tuple[int, ...]
    allocation_target: str
    depth_condition: str
    goal_condition: str
    phantom_enabled: bool
    c_min: int | None
    direction_kernel: str
    trajectory_mode: str
    resume_pattern: str
    expected_diagnostic_families: tuple[str, ...]
    usage: tuple[str, ...]
    accuracy_criterion: str
    performance_gate_seconds: float | None = None
    suites: tuple[str, ...] = FEATURE_MANIFEST_SUITES


_COMMON_DIAGNOSTICS = (
    "timing",
    "accuracy",
    "allocation",
    "depth",
    "result",
)
_PURE_AND_DISTRIBUTED_DIAGNOSTICS = _COMMON_DIAGNOSTICS + (
    "pure_core_jax_cache_static_shape",
    "distributed_worker_topology",
)


def _standard_problem_row(
        *,
        problem_fixture: str,
        allocation_target: str,
        dimension: int,
        reference_evidence: str,
        reference_posterior: str,
        phantom_enabled: bool,
        c_min: int | None,
        direction_kernel: str = "isotropic",
        trajectory_mode: str = "straight_line",
        resume_pattern: str = "run_until_goal",
        usage: tuple[str, ...] = ("accuracy",),
        diagnostic_families: tuple[str, ...] = _COMMON_DIAGNOSTICS,
        performance_gate_seconds: float | None = None,
) -> V3PerformanceFeatureRow:
    return V3PerformanceFeatureRow(
        row_id=f"standard_{problem_fixture}_{allocation_target}",
        problem_fixture=problem_fixture,
        dimension=dimension,
        reference_evidence=reference_evidence,
        reference_posterior=reference_posterior,
        seeds=(0, 17, 29),
        allocation_target=allocation_target,
        depth_condition="max_samples_standard_gate",
        goal_condition="host_goal_checked_between_compiled_epochs",
        phantom_enabled=phantom_enabled,
        c_min=c_min,
        direction_kernel=direction_kernel,
        trajectory_mode=trajectory_mode,
        resume_pattern=resume_pattern,
        expected_diagnostic_families=diagnostic_families,
        usage=usage,
        accuracy_criterion="abs(mean(log_Z_samples) - logZ_ref) <= 3 * std(log_Z_samples)",
        performance_gate_seconds=performance_gate_seconds,
    )


DEFAULT_V3_PERFORMANCE_FEATURE_ROWS = (
    _standard_problem_row(
        problem_fixture="basic_mvn",
        allocation_target="uniform",
        dimension=8,
        reference_evidence="analytic_8d_gaussian_logZ",
        reference_posterior="analytic_8d_gaussian_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="isotropic",
        usage=("accuracy", "benchmark"),
        diagnostic_families=_PURE_AND_DISTRIBUTED_DIAGNOSTICS,
        performance_gate_seconds=60.0,
    ),
    _standard_problem_row(
        problem_fixture="basic_mvn",
        allocation_target="evidence_improving",
        dimension=8,
        reference_evidence="analytic_8d_gaussian_logZ",
        reference_posterior="analytic_8d_gaussian_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="ellipsoidal",
        usage=("accuracy", "benchmark"),
        diagnostic_families=_PURE_AND_DISTRIBUTED_DIAGNOSTICS,
    ),
    _standard_problem_row(
        problem_fixture="basic_mvn",
        allocation_target="posterior_improving",
        dimension=8,
        reference_evidence="analytic_8d_gaussian_logZ",
        reference_posterior="analytic_8d_gaussian_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="gmm",
        resume_pattern="resume_until_goal",
        usage=("accuracy", "benchmark"),
        diagnostic_families=_PURE_AND_DISTRIBUTED_DIAGNOSTICS + (
            "direction_adaptation",
            "resume",
        ),
    ),
    _standard_problem_row(
        problem_fixture="spike_slab",
        allocation_target="uniform",
        dimension=8,
        reference_evidence="standard_problem_reference_logZ",
        reference_posterior="standard_problem_reference_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="ellipsoidal",
    ),
    _standard_problem_row(
        problem_fixture="spike_slab",
        allocation_target="evidence_improving",
        dimension=8,
        reference_evidence="standard_problem_reference_logZ",
        reference_posterior="standard_problem_reference_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="ellipsoidal",
    ),
    _standard_problem_row(
        problem_fixture="spike_slab",
        allocation_target="posterior_improving",
        dimension=8,
        reference_evidence="standard_problem_reference_logZ",
        reference_posterior="standard_problem_reference_posterior",
        phantom_enabled=True,
        c_min=20,
        direction_kernel="gmm",
        diagnostic_families=_COMMON_DIAGNOSTICS + (
            "direction_adaptation",
            "phantom_gate",
        ),
    ),
    _standard_problem_row(
        problem_fixture="plateau",
        allocation_target="uniform",
        dimension=2,
        reference_evidence="analytic_plateau_equality_logZ",
        reference_posterior="analytic_plateau_equality_atom_mass",
        phantom_enabled=False,
        c_min=None,
        diagnostic_families=_COMMON_DIAGNOSTICS + ("plateau_equality",),
    ),
    _standard_problem_row(
        problem_fixture="plateau",
        allocation_target="evidence_improving",
        dimension=2,
        reference_evidence="analytic_plateau_equality_logZ",
        reference_posterior="analytic_plateau_equality_atom_mass",
        phantom_enabled=False,
        c_min=None,
        diagnostic_families=_COMMON_DIAGNOSTICS + ("plateau_equality",),
    ),
    _standard_problem_row(
        problem_fixture="plateau",
        allocation_target="posterior_improving",
        dimension=2,
        reference_evidence="analytic_plateau_equality_logZ",
        reference_posterior="analytic_plateau_equality_atom_mass",
        phantom_enabled=True,
        c_min=20,
        diagnostic_families=_COMMON_DIAGNOSTICS + (
            "plateau_equality",
            "phantom_gate",
        ),
    ),
)
V3_FEATURE_MANIFEST = DEFAULT_V3_PERFORMANCE_FEATURE_ROWS


def v3_performance_feature_rows() -> tuple[V3PerformanceFeatureRow, ...]:
    """Return the immutable shared feature manifest rows."""
    return DEFAULT_V3_PERFORMANCE_FEATURE_ROWS


def row_ids_for_usage(usage: str) -> tuple[str, ...]:
    """Return manifest row ids tagged with an accuracy or benchmark usage."""
    return tuple(
        row.row_id
        for row in DEFAULT_V3_PERFORMANCE_FEATURE_ROWS
        if usage in row.usage
    )


def feature_rows_for_suite(
        suite: str,
        *,
        row_kind: str = ROW_KIND_ACCURACY,
) -> tuple[V3PerformanceFeatureRow, ...]:
    """Return manifest rows used to parametrize one Ticket 0020 suite.

    Pure-core and distributed suites intentionally draw from the same manifest.
    Suite-specific tests may vary worker counts and timing metadata, but row
    selection must not fork away from this source of truth.
    """
    if suite not in FEATURE_MANIFEST_SUITES:
        supported = ", ".join(FEATURE_MANIFEST_SUITES)
        raise ValueError(f"Unknown feature suite {suite!r}; expected {supported}.")
    return tuple(
        row
        for row in DEFAULT_V3_PERFORMANCE_FEATURE_ROWS
        if suite in row.suites and row_kind in row.usage
    )


def row_ids_for_suite(
        suite: str,
        *,
        row_kind: str = ROW_KIND_ACCURACY,
) -> tuple[str, ...]:
    """Return row ids used to parametrize one Ticket 0020 suite."""
    return tuple(
        row.row_id
        for row in feature_rows_for_suite(suite, row_kind=row_kind)
    )

def row_by_id(row_id: str) -> V3PerformanceFeatureRow:
    """Look up one manifest row by id.

    Args:
        row_id: Feature row id.

    Returns:
        The matching feature row.

    Raises:
        KeyError: If the row id is not part of the shared manifest.
    """
    for row in DEFAULT_V3_PERFORMANCE_FEATURE_ROWS:
        if row.row_id == row_id:
            return row
    raise KeyError(row_id)
