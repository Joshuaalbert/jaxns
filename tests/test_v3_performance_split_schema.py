from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import pytest

from benchmarks.v3_performance import standard_problem_speed
from benchmarks.v3_performance.feature_manifest import (
    DEFAULT_ALLOCATION_TARGETS,
    ROW_KIND_BENCHMARK,
    SUITE_DISTRIBUTED,
    SUITE_PURE_CORE,
    STANDARD_PROBLEM_FIXTURES,
    feature_rows_for_suite,
    row_ids_for_usage,
    v3_performance_feature_rows,
)
from benchmarks.v3_performance.split_schema import (
    assert_split_benchmark_coverage,
    assert_split_benchmark_markdown_report,
    assert_split_benchmark_record,
    assert_split_benchmark_records,
)


def _timings() -> dict[str, float]:
    return {
        "setup_seconds": 1.0,
        "compile_seconds": 2.0,
        "run_seconds": 3.0,
        "result_conversion_seconds": 4.0,
        "mc_shrinkage_seconds": 5.0,
        "total_seconds": 15.0,
    }


def _results() -> dict[str, Any]:
    return {
        "likelihood_evaluations": 123,
        "total_samples": 45,
        "log_Z_mean": -24.5,
        "log_Z_uncert": 0.25,
        "logZ_ref": -24.606462553878423,
        "accuracy_passed": True,
    }


def _metadata(row_id: str = "standard_basic_mvn_uniform") -> dict[str, Any]:
    row = {
        feature_row.row_id: feature_row
        for feature_row in v3_performance_feature_rows()
    }[row_id]
    return {
        "feature_row_id": row.row_id,
        "problem_fixture": row.problem_fixture,
        "dimension": row.dimension,
        "allocation_target": row.allocation_target,
        "seed": row.seeds[0],
        "target_num_live_points": 30,
        "max_samples": 1200,
        "shell_size": 15,
        "num_slices": 24,
        "phantom_burn_in": 4,
        "mc_sample_count": 1000,
        "depth_condition": row.depth_condition,
        "goal_condition": row.goal_condition,
        "phantom_enabled": row.phantom_enabled,
        "c_min": row.c_min,
        "direction_kernel": row.direction_kernel,
        "trajectory_mode": row.trajectory_mode,
        "resume_pattern": row.resume_pattern,
    }


def _pure_core_diagnostics() -> dict[str, Any]:
    return {
        "jax_cache_hit_count": 7,
        "jax_cache_miss_count": 1,
        "jit_compile_count": 1,
        "static_shape_signature": "basic_mvn:D8:shell15:live400",
        "static_shape_cache_key": "sha256:0123456789abcdef",
        "rejected_dynamic_shape_count": 0,
    }


def _distributed_diagnostics() -> dict[str, Any]:
    return {
        "requested_worker_spec": ["cpu:*:4"],
        "observed_node_count": 1,
        "live_node_ingress_process_count": 1,
        "live_node_coordinator_count": 1,
        "observed_worker_process_count": 4,
        "worker_utilization": 0.75,
        "queue_diagnostics": {
            "load_balancer_queue_length": 0,
            "node_queue_length": 1,
        },
        "compile_cache_diagnostics": {
            "compile_count": 1,
            "cache_hit_count": 7,
            "rejected_shape_cache_count": 0,
        },
    }


def _record(
        *,
        execution_mode: str,
        row_id: str = "standard_basic_mvn_uniform",
) -> dict[str, Any]:
    return {
        "schema_version": "v3_performance_split_v1",
        "metric_family": "v3_performance_split",
        "execution_mode": execution_mode,
        "metadata": _metadata(row_id=row_id),
        "timings": _timings(),
        "diagnostics": (
            _pure_core_diagnostics()
            if execution_mode == "pure_core"
            else _distributed_diagnostics()
        ),
        "results": _results(),
    }


def _feature_row_ids_by_fixture_and_allocation() -> set[tuple[str, str]]:
    return {
        (row.problem_fixture, row.allocation_target)
        for row in v3_performance_feature_rows()
    }


def test_shared_manifest_covers_required_standard_problem_feature_rows():
    covered = _feature_row_ids_by_fixture_and_allocation()
    expected = {
        (problem_fixture, allocation_target)
        for problem_fixture in STANDARD_PROBLEM_FIXTURES
        for allocation_target in DEFAULT_ALLOCATION_TARGETS
    }

    assert expected <= covered
    benchmark_ids = set(row_ids_for_usage("benchmark"))
    assert {
        f"standard_basic_mvn_{allocation_target}"
        for allocation_target in DEFAULT_ALLOCATION_TARGETS
    } <= benchmark_ids
    for row in v3_performance_feature_rows():
        assert row.problem_fixture
        assert row.reference_evidence
        assert row.reference_posterior
        assert row.seeds
        assert row.depth_condition
        assert row.goal_condition
        assert row.direction_kernel
        assert row.trajectory_mode
        assert row.resume_pattern in {"run_until_goal", "resume_until_goal"}
        assert row.expected_diagnostic_families
        assert row.usage
        assert row.accuracy_criterion == (
            "abs(mean(log_Z_samples) - logZ_ref) <= 3 * std(log_Z_samples)"
        )
        if row.phantom_enabled:
            assert row.c_min is not None


@pytest.mark.parametrize("execution_mode", ["pure_core", "distributed"])
def test_split_benchmark_schema_accepts_mode_specific_records(execution_mode):
    assert_split_benchmark_record(_record(execution_mode=execution_mode))


def test_split_record_set_requires_pure_core_and_distributed_coverage_match():
    row_id = "standard_basic_mvn_uniform"
    assert_split_benchmark_records([
        _record(execution_mode="pure_core", row_id=row_id),
        _record(execution_mode="distributed", row_id=row_id),
    ])

    with pytest.raises(AssertionError, match="coverage must match|pure_only"):
        assert_split_benchmark_coverage(
            pure_core_feature_row_ids=(row_id,),
            distributed_feature_row_ids=("standard_basic_mvn_evidence_improving",),
        )


def test_split_benchmark_parametrization_comes_from_shared_manifest():
    pure_core_rows = tuple(row.row_id for row in feature_rows_for_suite(
        SUITE_PURE_CORE,
        row_kind=ROW_KIND_BENCHMARK,
    ))
    distributed_rows = tuple(row.row_id for row in feature_rows_for_suite(
        SUITE_DISTRIBUTED,
        row_kind=ROW_KIND_BENCHMARK,
    ))
    benchmark_rows = row_ids_for_usage(ROW_KIND_BENCHMARK)

    assert pure_core_rows == benchmark_rows
    assert distributed_rows == benchmark_rows
    assert_split_benchmark_coverage(
        pure_core_feature_row_ids=pure_core_rows,
        distributed_feature_row_ids=distributed_rows,
    )


def test_executable_benchmark_config_is_manifest_parametrized():
    benchmark_rows = row_ids_for_usage(ROW_KIND_BENCHMARK)
    config = standard_problem_speed.default_standard_problem_speed_config()

    feature_row_ids = getattr(config, "feature_row_ids", None)
    if feature_row_ids is None:
        pytest.fail(
            "The executable v3 standard-problem benchmark config must expose "
            "manifest feature_row_ids instead of duplicating allocation-target "
            "or row-id constants in tests."
        )
    assert tuple(feature_row_ids) == benchmark_rows

    if hasattr(standard_problem_speed, "benchmark_feature_rows"):
        producer_rows = tuple(
            row.row_id
            for row in standard_problem_speed.benchmark_feature_rows()
        )
    else:
        producer_rows = tuple(
            standard_problem_speed.row_ids_for_benchmark_config(config)
        )
    assert producer_rows == benchmark_rows


def test_pure_core_schema_requires_jax_cache_and_static_shape_diagnostics():
    record = _record(execution_mode="pure_core")
    bad_record = copy.deepcopy(record)
    bad_record["diagnostics"].pop("static_shape_cache_key")
    with pytest.raises(AssertionError, match="static_shape_cache_key"):
        assert_split_benchmark_record(bad_record)

    bad_record = copy.deepcopy(record)
    bad_record["diagnostics"]["rejected_dynamic_shape_count"] = -1
    with pytest.raises(AssertionError, match="rejected_dynamic_shape_count"):
        assert_split_benchmark_record(bad_record)


def test_split_schema_requires_full_benchmark_setting_metadata():
    record = _record(execution_mode="pure_core")
    for field_name in (
            "target_num_live_points",
            "max_samples",
            "shell_size",
            "num_slices",
            "phantom_burn_in",
            "mc_sample_count",
    ):
        bad_record = copy.deepcopy(record)
        bad_record["metadata"].pop(field_name)
        with pytest.raises(AssertionError, match=field_name):
            assert_split_benchmark_record(bad_record)

    bad_record = copy.deepcopy(record)
    bad_record["metadata"]["max_samples"] = 0
    with pytest.raises(AssertionError, match="max_samples"):
        assert_split_benchmark_record(bad_record)


def test_distributed_schema_requires_worker_topology_and_cache_diagnostics():
    record = _record(execution_mode="distributed")
    bad_record = copy.deepcopy(record)
    bad_record["diagnostics"].pop("live_node_coordinator_count")
    with pytest.raises(AssertionError, match="live_node_coordinator_count"):
        assert_split_benchmark_record(bad_record)

    bad_record = copy.deepcopy(record)
    bad_record["diagnostics"]["live_node_ingress_process_count"] = 2
    with pytest.raises(AssertionError, match="one live ingress per node"):
        assert_split_benchmark_record(bad_record)

    bad_record = copy.deepcopy(record)
    bad_record["diagnostics"]["worker_utilization"] = 1.25
    with pytest.raises(AssertionError, match="worker_utilization"):
        assert_split_benchmark_record(bad_record)


def test_split_schema_requires_separate_compile_result_and_mc_timings():
    record = _record(execution_mode="pure_core")
    bad_record = copy.deepcopy(record)
    bad_record["timings"].pop("compile_seconds")
    with pytest.raises(AssertionError, match="compile_seconds"):
        assert_split_benchmark_record(bad_record)

    bad_record = copy.deepcopy(record)
    bad_record["timings"]["total_seconds"] = 14.0
    with pytest.raises(AssertionError, match="setup \\+ compile \\+ run"):
        assert_split_benchmark_record(bad_record)


def test_split_schema_rejects_failed_accuracy_rows():
    record = _record(execution_mode="pure_core")
    bad_record = copy.deepcopy(record)
    bad_record["results"]["accuracy_passed"] = False

    with pytest.raises(AssertionError, match="accuracy_passed"):
        assert_split_benchmark_record(bad_record)


def test_split_markdown_report_is_dated_and_mentions_required_sections():
    records: list[Mapping[str, Any]] = [
        _record(execution_mode="pure_core"),
        _record(execution_mode="distributed"),
    ]
    report_text = "\n".join([
        "# V3 Performance Split Benchmark Report",
        "",
        "Date: 2026-05-11",
        "",
        "## Pure-Core Results",
        "",
        "| feature row | mode | setup s | compile s | run s | result s | MC s | likelihood evals |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| standard_basic_mvn_uniform | pure_core | 1 | 2 | 3 | 4 | 5 | 123 |",
        "",
        "## Distributed Results",
        "",
        "| feature row | mode | setup s | compile s | run s | result s | MC s | likelihood evals |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| standard_basic_mvn_uniform | distributed | 1 | 2 | 3 | 4 | 5 | 123 |",
        "",
        "## Diagnostics",
        "",
        "- JAX cache/static-shape diagnostics are reported for pure_core rows.",
        "- worker topology diagnostics are reported for distributed rows.",
    ])

    assert_split_benchmark_markdown_report(
        path="v3_performance_split_2026-05-11.md",
        text=report_text,
        records=records,
    )

    with pytest.raises(AssertionError, match="dated"):
        assert_split_benchmark_markdown_report(
            path="v3_performance_split.md",
            text=report_text,
            records=records,
        )
