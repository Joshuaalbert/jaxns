from __future__ import annotations

from collections import Counter

import pytest

from benchmarks.v3_performance.feature_manifest import ALLOCATION_TARGETS
from benchmarks.v3_performance.feature_manifest import (
    FEATURE_MANIFEST_SUITES,
)
from benchmarks.v3_performance.feature_manifest import ROW_KIND_ACCURACY
from benchmarks.v3_performance.feature_manifest import (
    STANDARD_PROBLEM_ROW_NAMES,
)
from benchmarks.v3_performance.feature_manifest import SUITE_DISTRIBUTED
from benchmarks.v3_performance.feature_manifest import SUITE_PURE_CORE
from benchmarks.v3_performance.feature_manifest import V3_FEATURE_MANIFEST
from benchmarks.v3_performance.feature_manifest import feature_rows_for_suite
from benchmarks.v3_performance.feature_manifest import row_ids_for_suite
from benchmarks.v3_performance.feature_manifest import row_ids_for_usage
from tests import test_v3_manifest_accuracy


def test_v3_feature_manifest_has_unique_row_ids() -> None:
    row_id_counts = Counter(row.row_id for row in V3_FEATURE_MANIFEST)

    assert all(count == 1 for count in row_id_counts.values())


def _parametrized_feature_rows(test_function) -> tuple[object, ...]:
    rows = []
    for mark in getattr(test_function, "pytestmark", ()):
        if mark.name != "parametrize":
            continue
        argnames = mark.args[0]
        argvalues = mark.args[1]
        names = (
            [name.strip() for name in argnames.split(",")]
            if isinstance(argnames, str)
            else list(argnames)
        )
        if "feature_row" not in names:
            continue
        feature_row_index = names.index("feature_row")
        for raw_value in argvalues:
            values = getattr(raw_value, "values", raw_value)
            if len(names) == 1:
                rows.append(values[0] if hasattr(raw_value, "values") else values)
            else:
                rows.append(values[feature_row_index])
    return tuple(rows)


def _parametrized_feature_row_ids(test_function) -> tuple[str, ...]:
    rows = _parametrized_feature_rows(test_function)

    assert rows, (
        f"{test_function.__name__} must be parametrized over executable "
        "feature_row values from the shared manifest."
    )
    assert all(row in V3_FEATURE_MANIFEST for row in rows)
    return tuple(row.row_id for row in rows)


def test_executable_accuracy_tests_cover_the_shared_manifest_rows() -> None:
    pure_core_row_ids = set(
        _parametrized_feature_row_ids(
            test_v3_manifest_accuracy.test_pure_core_standard_problem_accuracy,
        )
    )
    distributed_row_ids = set(
        _parametrized_feature_row_ids(
            test_v3_manifest_accuracy.test_distributed_standard_problem_accuracy,
        )
    )
    manifest_row_ids = set(row_ids_for_usage(ROW_KIND_ACCURACY))

    assert pure_core_row_ids == distributed_row_ids
    assert pure_core_row_ids
    assert pure_core_row_ids == manifest_row_ids


@pytest.mark.parametrize("suite", FEATURE_MANIFEST_SUITES)
def test_accuracy_parametrization_rows_come_from_shared_manifest(suite) -> None:
    manifest_ids = tuple(row.row_id for row in V3_FEATURE_MANIFEST)
    suite_rows = feature_rows_for_suite(suite, row_kind=ROW_KIND_ACCURACY)

    assert suite_rows
    assert tuple(row.row_id for row in suite_rows) == row_ids_for_suite(
        suite,
        row_kind=ROW_KIND_ACCURACY,
    )
    assert all(row in V3_FEATURE_MANIFEST for row in suite_rows)
    assert set(row.row_id for row in suite_rows) <= set(manifest_ids)


def test_shared_manifest_covers_required_ticket_0020_feature_axes() -> None:
    accuracy_rows = [
        row for row in V3_FEATURE_MANIFEST
        if ROW_KIND_ACCURACY in row.usage
    ]

    assert {row.allocation_target for row in accuracy_rows} == set(
        ALLOCATION_TARGETS
    )
    assert {row.phantom_enabled for row in accuracy_rows} == {False, True}
    assert {row.direction_kernel for row in accuracy_rows} >= {
        "isotropic",
        "gmm",
    }
    assert {row.trajectory_mode for row in accuracy_rows} >= {"straight_line"}
    assert {row.resume_pattern for row in accuracy_rows} >= {
        "run_until_goal",
        "resume_until_goal",
    }
    assert {
        row.problem_fixture for row in accuracy_rows
        if row.problem_fixture in STANDARD_PROBLEM_ROW_NAMES
    } == set(STANDARD_PROBLEM_ROW_NAMES)

    standard_problem_pairs = {
        (row.problem_fixture, row.allocation_target)
        for row in accuracy_rows
        if row.problem_fixture in STANDARD_PROBLEM_ROW_NAMES
    }
    assert standard_problem_pairs == {
        (problem, allocation_target)
        for problem in STANDARD_PROBLEM_ROW_NAMES
        for allocation_target in ALLOCATION_TARGETS
    }


def test_accuracy_rows_share_the_same_acceptance_rule_across_suites() -> None:
    for row in V3_FEATURE_MANIFEST:
        if ROW_KIND_ACCURACY not in row.usage:
            continue
        assert set(row.suites) == {SUITE_PURE_CORE, SUITE_DISTRIBUTED}
        assert row.accuracy_criterion == (
            "abs(mean(log_Z_samples) - logZ_ref) <= 3 * std(log_Z_samples)"
        )


def test_primary_basic_mvn_benchmark_is_isotropic_sub_60_gate() -> None:
    rows_by_id = {row.row_id: row for row in V3_FEATURE_MANIFEST}
    row = rows_by_id["standard_basic_mvn_uniform"]

    assert "benchmark" in row.usage
    assert row.problem_fixture == "basic_mvn"
    assert row.dimension == 8
    assert row.allocation_target == "uniform"
    assert row.direction_kernel == "isotropic"
    assert row.performance_gate_seconds == pytest.approx(60.0)
