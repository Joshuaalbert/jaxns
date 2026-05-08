from __future__ import annotations

import importlib
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import NamedTuple

import numpy as np

from benchmarks.v3_validation.schema_checks import (
    assert_calibration_rollup_table,
    assert_calibration_table,
    assert_performance_guardrail_suite,
    assert_plateau_equality_record,
    assert_posterior_quality_record,
    assert_posterior_wasserstein_record,
    assert_rmse_vs_likelihood_record,
    assert_rollup_separates_evidence_and_posterior,
    assert_timing_history_append_only,
)
from jaxns.core import NestedSampler


SEEDS = (0, 1)
MAX_SAMPLES = 8
EVIDENCE_SAMPLE_COUNT = 4


class PublicRunEvent(NamedTuple):
    api_name: str
    allocation_target: object
    max_samples: int
    sampler_type: str


def _require_collector_api():
    try:
        collector_module = importlib.import_module(
            "benchmarks.v3_validation.collector"
        )
    except ModuleNotFoundError as error:
        if error.name == "benchmarks.v3_validation.collector":
            raise AssertionError(
                "Ticket 0010 final acceptance requires "
                "benchmarks.v3_validation.collector with a public "
                "collect_minimal_v3_validation_records(...) API. The "
                "collector must run cheap real public v3 sampler/result/"
                "diagnostics paths and emit benchmark schema records."
            ) from error
        raise

    collector = getattr(
        collector_module,
        "collect_minimal_v3_validation_records",
        None,
    )
    assert callable(collector), (
        "benchmarks.v3_validation.collector must expose callable "
        "collect_minimal_v3_validation_records(...)."
    )
    return collector


def _install_public_run_spy(monkeypatch) -> list[PublicRunEvent]:
    events: list[PublicRunEvent] = []
    original_run_until_goal = NestedSampler.run_until_goal
    original_run = NestedSampler.run

    def tracked_run_until_goal(self, *args, **kwargs):
        events.append(
            PublicRunEvent(
                api_name="run_until_goal",
                allocation_target=kwargs.get("allocation_target", "uniform"),
                max_samples=int(self.max_samples),
                sampler_type=type(self.sampler).__name__,
            )
        )
        return original_run_until_goal(self, *args, **kwargs)

    def tracked_run(self, *args, **kwargs):
        events.append(
            PublicRunEvent(
                api_name="run",
                allocation_target=None,
                max_samples=int(self.max_samples),
                sampler_type=type(self.sampler).__name__,
            )
        )
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(
        NestedSampler,
        "run_until_goal",
        tracked_run_until_goal,
    )
    monkeypatch.setattr(NestedSampler, "run", tracked_run)
    return events


def _records_section(
        rollup: Mapping[str, Any],
        name: str,
) -> Sequence[Mapping[str, Any]]:
    records = rollup.get(name)
    assert isinstance(records, Sequence) and not isinstance(records, str), (
        f"collector rollup must contain a non-string sequence at {name!r}."
    )
    assert records, f"collector rollup section {name!r} must not be empty."
    for record in records:
        assert isinstance(record, Mapping), (
            f"collector rollup section {name!r} must contain mapping records."
        )
    return records


def _method_label(method_setting: Mapping[str, Any]) -> str | None:
    method = str(method_setting.get("method", "")).lower()
    allocation = method_setting.get("allocation")
    trajectory = str(method_setting.get("trajectory", "")).lower()
    sampler = str(method_setting.get("sampler", "")).lower()

    if method == "baseline-race-tree":
        return "baseline race-tree"
    if (
            "galilean" in method
            or "galilean" in trajectory
            or "galilean" in sampler
    ):
        return "Galilean"
    if "dynamic" in method or _allocation_is_dynamic(allocation):
        return "dynamic allocation"
    if "phantom" in method:
        return "phantom-conditioned"
    return None


def _allocation_is_dynamic(allocation: object) -> bool:
    if isinstance(allocation, Mapping):
        return str(allocation.get("mode", "")).lower() == "dynamic"
    return str(allocation).lower() == "dynamic"


def _assert_method_settings_cover_ticket_methods(
        calibration_records: Sequence[Mapping[str, Any]],
) -> None:
    seeds_by_label: dict[str, set[int]] = {
        "baseline race-tree": set(),
        "phantom-conditioned": set(),
        "dynamic allocation": set(),
        "Galilean": set(),
    }
    for record in calibration_records:
        metadata = record["metadata"]
        label = _method_label(metadata["method_setting"])
        if label in seeds_by_label:
            seeds_by_label[label].add(int(metadata["seed"]))

    for label, seen_seeds in seeds_by_label.items():
        assert seen_seeds == set(SEEDS), (
            "collector evidence calibration must cover two deterministic "
            f"seeds for the {label} method setting; saw {sorted(seen_seeds)}."
        )


def _assert_phantom_rows_include_rho(
        calibration_records: Sequence[Mapping[str, Any]],
) -> None:
    for record in calibration_records:
        method = str(record["metadata"]["method_setting"]["method"]).lower()
        if "phantom" not in method:
            continue
        calibration = record["evidence_calibration"]
        for key in ("rho_g", "rho_fit"):
            assert key in calibration, (
                f"phantom calibration rows must include {key} diagnostics."
            )
            values = np.asarray(calibration[key], dtype=float)
            assert values.ndim == 1 and values.size > 0
            assert np.all((values > 0.0) & (values <= 1.0))


def _assert_result_diagnostics_metadata(record: Mapping[str, Any]) -> None:
    diagnostics = record["metadata"].get("result_diagnostics")
    assert isinstance(diagnostics, Mapping), (
        "collector metadata must include result_diagnostics from the public "
        "NestedSamplerResults/diagnostics API, not only producer inputs."
    )
    for key in (
            "total_num_samples",
            "log_Z_mean",
            "log_Z_uncert",
            "ess",
            "block_count",
    ):
        assert key in diagnostics, (
            f"result_diagnostics metadata must include {key!r}."
        )
    assert 1 <= int(diagnostics["total_num_samples"]) <= MAX_SAMPLES
    assert int(diagnostics["block_count"]) >= 1
    assert np.isfinite(float(diagnostics["log_Z_mean"]))
    assert np.isfinite(float(diagnostics["log_Z_uncert"]))
    assert float(diagnostics["ess"]) > 0.0


def _assert_plateau_equality_comes_from_result_blocks(
        record: Mapping[str, Any],
) -> None:
    assert_plateau_equality_record(record)
    plateau = record["plateau_equality"]
    assert plateau.get("source") == "result_blocks", (
        "plateau equality recovery must be derived from actual result/block "
        "output, not only deterministic fixture math."
    )
    assert int(plateau.get("result_block_count", 0)) >= 1
    assert int(plateau.get("equality_sample_count", 0)) >= 1
    diagnostics = record["metadata"]["result_diagnostics"]
    assert int(plateau["result_block_count"]) == int(
        diagnostics["block_count"]
    )


def _assert_public_runs_were_executed(
        events: Sequence[PublicRunEvent],
) -> None:
    assert len(events) >= len(SEEDS) * 4, (
        "collector must execute real public v3 runs for every seed and each "
        "ticket method setting."
    )
    assert {event.api_name for event in events}.intersection(
        {"run_until_goal", "run"}
    )
    assert all(event.max_samples <= MAX_SAMPLES for event in events)
    assert any(
        event.api_name == "run_until_goal"
        and event.allocation_target in {
            "uniform",
            "evidence_improving",
            "posterior_improving",
        }
        for event in events
    ), (
        "collector must use the public v3 NestedSampler.run_until_goal path "
        "for cheap toy validation runs."
    )


def test_minimal_collector_runs_public_v3_apis_and_emits_schema_records(
        monkeypatch,
):
    collector = _require_collector_api()
    public_run_events = _install_public_run_spy(monkeypatch)

    rollup = collector(
        seeds=SEEDS,
        max_samples=MAX_SAMPLES,
        evidence_sample_count=EVIDENCE_SAMPLE_COUNT,
    )

    assert isinstance(rollup, Mapping)
    assert "metrics" not in rollup, (
        "collector must keep evidence and posterior records in separate "
        "sections."
    )
    _assert_public_runs_were_executed(public_run_events)

    calibration_records = _records_section(rollup, "evidence_calibration")
    assert_calibration_table(calibration_records)
    _assert_method_settings_cover_ticket_methods(calibration_records)
    _assert_phantom_rows_include_rho(calibration_records)
    for record in calibration_records:
        _assert_result_diagnostics_metadata(record)

    calibration_rollups = _records_section(rollup, "calibration_rollups")
    assert_calibration_rollup_table(calibration_records, calibration_rollups)

    plateau_records = _records_section(rollup, "plateau_equality")
    for record in plateau_records:
        _assert_plateau_equality_comes_from_result_blocks(record)

    posterior_quality_records = _records_section(rollup, "posterior_quality")
    for record in posterior_quality_records:
        assert_posterior_quality_record(record)
    assert_rollup_separates_evidence_and_posterior(rollup)

    wasserstein_records = _records_section(rollup, "posterior_wasserstein")
    for record in wasserstein_records:
        assert_posterior_wasserstein_record(record)

    rmse_records = _records_section(rollup, "rmse_vs_likelihood")
    for record in rmse_records:
        assert_rmse_vs_likelihood_record(record)
        assert any(record["rmse_vs_likelihood"]["pareto_efficient"])

    guardrail_records = _records_section(rollup, "performance_guardrails")
    assert_performance_guardrail_suite(guardrail_records)

    timing_history = _records_section(rollup, "timing_history")
    timing_rows = _records_section(rollup, "timing_rows")
    assert_timing_history_append_only([], timing_history)
    assert_timing_history_append_only([], timing_rows)
    for row in timing_rows:
        assert row in timing_history
