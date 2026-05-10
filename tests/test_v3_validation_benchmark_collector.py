from __future__ import annotations

import importlib
from types import SimpleNamespace
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

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


class GammaTimingCounts(NamedTuple):
    A_cg: object
    B_cg: object
    E_cg: object


class GammaTimingResult(NamedTuple):
    log_L_blocks: object
    block_first_idx: object
    block_size: object
    block_incoming_K: object
    block_out_degree: object
    block_start: object
    block_stop: object
    block_sample_indices: object
    counts: GammaTimingCounts

    def phantom_conditioning_diagnostics(self, C_min: float = 20):
        del C_min
        return self.counts


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


def _assert_phantom_rows_include_kish_gate_diagnostics(
        calibration_records: Sequence[Mapping[str, Any]],
) -> None:
    for record in calibration_records:
        method = str(record["metadata"]["method_setting"]["method"]).lower()
        if "phantom" not in method:
            continue
        calibration = record["evidence_calibration"]
        for key in (
                "kish_participating_cluster_counts",
                "phantom_gate_active",
                "phantom_A_g",
                "phantom_B_g",
                "phantom_E_g",
                "phantom_R_g",
                "C_min",
        ):
            assert key in calibration, (
                f"phantom calibration rows must include {key} diagnostics."
            )
        assert "rho_g" not in calibration
        assert "rho_fit" not in calibration
        kish = np.asarray(
            calibration["kish_participating_cluster_counts"],
            dtype=float,
        )
        gate = np.asarray(calibration["phantom_gate_active"], dtype=bool)
        assert kish.ndim == 1 and kish.size > 0, "kish diagnostics must be 1D."
        assert gate.shape == kish.shape, "phantom_gate_active length must align."
        assert np.all(kish >= 0.0), "kish diagnostics must be non-negative."
        c_min = float(calibration["C_min"])
        assert np.isfinite(c_min) and c_min > 0.0, "C_min must be positive."
        np.testing.assert_array_equal(gate, kish >= c_min)
        aggregate_counts = {}
        for key in ("phantom_A_g", "phantom_B_g", "phantom_E_g", "phantom_R_g"):
            values = np.asarray(calibration[key], dtype=float)
            assert values.shape == kish.shape, f"{key} length must align."
            assert np.all(values >= 0.0), f"{key} counts must be non-negative."
            aggregate_counts[key] = values
        np.testing.assert_allclose(
            aggregate_counts["phantom_R_g"],
            (
                aggregate_counts["phantom_A_g"]
                - aggregate_counts["phantom_B_g"]
                - aggregate_counts["phantom_E_g"]
            ),
        )


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


def test_gamma_phantom_conditioning_timing_runs_probability_sampler(
        monkeypatch,
):
    _require_collector_api()
    collector_module = importlib.import_module("benchmarks.v3_validation.collector")
    calls = []

    def fake_sampler(**kwargs):
        calls.append(kwargs)
        num_samples = int(kwargs["num_samples"])
        num_blocks = int(kwargs["block_state"].log_L_blocks.shape[0])
        samples = jnp.ones((num_samples, num_blocks))
        return SimpleNamespace(
            p_gt_samples=samples,
            p_eq_samples=samples,
            p_lt_samples=samples,
        )

    monkeypatch.setattr(
        collector_module,
        "sample_gamma_weighted_phantom_probabilities",
        fake_sampler,
    )
    result = GammaTimingResult(
        log_L_blocks=jnp.asarray([0.0, 1.0]),
        block_first_idx=jnp.asarray([0, 1], dtype=jnp.int32),
        block_size=jnp.asarray([1, 1], dtype=jnp.int32),
        block_incoming_K=jnp.asarray([5, 4], dtype=jnp.int32),
        block_out_degree=jnp.asarray([0, 0], dtype=jnp.int32),
        block_start=None,
        block_stop=None,
        block_sample_indices=jnp.asarray([[0], [1]], dtype=jnp.int32),
        counts=GammaTimingCounts(
            A_cg=jnp.ones((20, 2)),
            B_cg=jnp.zeros((20, 2)),
            E_cg=jnp.zeros((20, 2)),
        ),
    )

    seconds = collector_module._measure_gamma_phantom_conditioning_seconds(
        result,
        seed=123,
        evidence_sample_count=3,
        require=True,
    )

    assert seconds >= 0.0
    assert len(calls) == 1
    assert calls[0]["num_samples"] == 3
    assert calls[0]["C_min"] == 20


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


def _phantom_calibration_record(**updates) -> dict[str, Any]:
    calibration = {
        "kish_participating_cluster_counts": [20.0, 12.0],
        "phantom_gate_active": [True, False],
        "phantom_A_g": [20.0, 18.0],
        "phantom_B_g": [8.0, 6.0],
        "phantom_E_g": [3.0, 2.0],
        "phantom_R_g": [9.0, 10.0],
        "C_min": 20,
    }
    calibration.update(updates)
    return {
        "metadata": {
            "method_setting": {
                "method": "phantom-conditioned",
            },
        },
        "evidence_calibration": calibration,
    }


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"phantom_R_g": [8.0, 10.0]}, "Not equal|Mismatched|ACTUAL"),
        (
            {
                "kish_participating_cluster_counts": [20.0, 21.0],
                "phantom_gate_active": [True, False],
            },
            "Arrays are not equal|Mismatched|ACTUAL",
        ),
        ({"C_min": 0}, "C_min"),
        ({"phantom_A_g": [20.0]}, "align"),
    ],
    ids=["bad-R", "bad-gate", "bad-C-min", "bad-length"],
)
def test_collector_phantom_row_check_rejects_incoherent_diagnostics(
        updates,
        match,
):
    with pytest.raises(AssertionError, match=match):
        _assert_phantom_rows_include_kish_gate_diagnostics(
            [_phantom_calibration_record(**updates)]
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
    _assert_phantom_rows_include_kish_gate_diagnostics(calibration_records)
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
    guardrail_names = {
        record["performance_guardrail"]["name"]
        for record in guardrail_records
    }
    assert "gamma_phantom_conditioning" in guardrail_names
    assert "rho_bootstrap" not in guardrail_names

    timing_history = _records_section(rollup, "timing_history")
    timing_rows = _records_section(rollup, "timing_rows")
    assert_timing_history_append_only([], timing_history)
    assert_timing_history_append_only([], timing_rows)
    for row in timing_rows:
        assert row in timing_history
        collector_timings = row["metadata"].get("collector_timings", {})
        assert "gamma_phantom_conditioning" in collector_timings
        assert "rho_bootstrap" not in collector_timings
