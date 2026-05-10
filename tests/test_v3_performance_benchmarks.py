from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


DEFAULT_ALLOCATION_TARGETS = (
    "uniform",
    "evidence_improving",
    "posterior_improving",
)
REQUIRED_TIMING_FIELDS = (
    "setup_seconds",
    "run_seconds",
    "result_conversion_seconds",
    "mc_shrinkage_seconds",
    "total_seconds",
)
REQUIRED_FRACTION_FIELDS = (
    "setup_fraction",
    "run_fraction",
    "result_conversion_fraction",
    "mc_shrinkage_fraction",
)
REQUIRED_CONFIG_FIELDS = (
    "seed",
    "worker_specs",
    "target_num_live_points",
    "live_points_per_dimension",
    "max_samples",
    "shell_size",
    "num_slices",
    "phantom_burn_in",
    "direction_kernel",
    "mc_sample_count",
)


def _require_speed_api():
    try:
        module = importlib.import_module(
            "benchmarks.v3_performance.standard_problem_speed"
        )
    except ModuleNotFoundError as error:
        if error.name in {
            "benchmarks.v3_performance",
            "benchmarks.v3_performance.standard_problem_speed",
        }:
            raise AssertionError(
                "Ticket 0016 requires a public "
                "benchmarks.v3_performance.standard_problem_speed module."
            ) from error
        raise

    required_callables = (
        "default_standard_problem_speed_config",
        "collect_standard_problem_speed_records",
        "assert_standard_problem_speed_record",
        "compute_timing_fractions",
        "main",
    )
    for name in required_callables:
        assert callable(getattr(module, name, None)), (
            "benchmarks.v3_performance.standard_problem_speed must expose a "
            f"callable {name}(...)."
        )
    return module


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    assert isinstance(value, Mapping), (
        "benchmark config/records must be mappings or dataclasses that expose "
        "their public fields."
    )
    return value


def _speed_config(api, **overrides):
    return api.default_standard_problem_speed_config(**overrides)


def _complete_speed_record(**updates) -> dict[str, Any]:
    record = {
        "schema_version": "v3_standard_problem_speed_v1",
        "metric_family": "standard_problem_speed",
        "metadata": {
            "problem": "basic_mvn",
            "dimension": 8,
            "allocation_target": "uniform",
            "seed": 7,
            "worker_specs": ["cpu:*:2"],
            "target_num_live_points": 400,
            "live_points_per_dimension": 50.0,
            "live_point_policy": "50_per_dimension",
            "max_samples": 12,
            "shell_size": 15,
            "num_slices": 24,
            "phantom_burn_in": 4,
            "direction_kernel": "ellipsoidal",
            "mc_sample_count": 5,
            "software_version": "3.0.0-test",
        },
        "timings": {
            "setup_seconds": 1.0,
            "run_seconds": 2.0,
            "result_conversion_seconds": 4.0,
            "mc_shrinkage_seconds": 5.0,
            "total_seconds": 12.0,
        },
        "timing_fractions": {
            "setup_fraction": 1.0 / 12.0,
            "run_fraction": 2.0 / 12.0,
            "result_conversion_fraction": 4.0 / 12.0,
            "mc_shrinkage_fraction": 5.0 / 12.0,
        },
        "diagnostics": {
            "worker_sampler_latency_seconds": 3.5,
            "actual_worker_count": 2,
        },
        "results": {
            "likelihood_evaluations": 123,
            "total_samples": 12,
            "log_Z_mean": -50.125,
            "log_Z_uncert": 0.25,
            "logZ_ref": -50.0,
        },
    }
    for key, value in updates.items():
        record[key] = value
    return record


def test_standard_problem_speed_schema_accepts_complete_records():
    api = _require_speed_api()
    record = _complete_speed_record()

    api.assert_standard_problem_speed_record(record)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda record: record["timings"].pop("run_seconds"),
            "run_seconds|timing",
        ),
        (
            lambda record: record["timings"].__setitem__(
                "mc_shrinkage_seconds",
                -0.01,
            ),
            "mc_shrinkage_seconds|non-negative|>= 0",
        ),
        (
            lambda record: record["timings"].__setitem__(
                "run_seconds",
                np.inf,
            ),
            "run_seconds|finite",
        ),
        (
            lambda record: record["timing_fractions"].__setitem__(
                "run_fraction",
                0.99,
            ),
            "run_fraction|fraction|timing",
        ),
        (
            lambda record: record["metadata"].pop("live_point_policy"),
            "live_point_policy|metadata",
        ),
        (
            lambda record: record["results"].pop("logZ_ref"),
            "logZ_ref|results",
        ),
        (
            lambda record: record["results"].__setitem__(
                "likelihood_evaluations",
                0,
            ),
            "likelihood_evaluations|positive",
        ),
        (
            lambda record: record["metadata"].__setitem__(
                "worker_specs",
                "cpu:*:2",
            ),
            "worker_specs|string|sequence",
        ),
        (
            lambda record: record["metadata"].__setitem__(
                "worker_specs",
                ["cpu:*:2", ""],
            ),
            "worker_specs|non-empty|string",
        ),
        (
            lambda record: record["diagnostics"].pop("actual_worker_count"),
            "actual_worker_count|diagnostics",
        ),
    ],
    ids=[
        "missing-run-timing",
        "negative-mc-timing",
        "nonfinite-run-timing",
        "fraction-not-derived",
        "missing-live-point-policy",
        "missing-reference-evidence",
        "nonpositive-likelihood-evaluations",
        "bare-string-worker-specs",
        "empty-worker-spec",
        "missing-actual-worker-count",
    ],
)
def test_standard_problem_speed_schema_rejects_bad_timing_fields(
        mutate,
        match,
):
    api = _require_speed_api()
    record = _complete_speed_record()
    mutate(record)

    with pytest.raises(AssertionError, match=match):
        api.assert_standard_problem_speed_record(record)


def test_timing_fractions_are_derived_from_measured_seconds():
    api = _require_speed_api()
    timings = {
        "setup_seconds": 0.5,
        "run_seconds": 1.0,
        "result_conversion_seconds": 0.25,
        "mc_shrinkage_seconds": 1.25,
    }

    fractions = api.compute_timing_fractions(timings)

    total = sum(timings.values())
    assert set(REQUIRED_FRACTION_FIELDS).issubset(fractions)
    for timing_name, seconds in timings.items():
        fraction_name = timing_name.removesuffix("_seconds") + "_fraction"
        assert fractions[fraction_name] == pytest.approx(seconds / total)
    assert sum(fractions[name] for name in REQUIRED_FRACTION_FIELDS) == (
        pytest.approx(1.0)
    )


def test_worker_sampler_latency_is_not_part_of_wall_clock_fractions():
    api = _require_speed_api()
    record = _complete_speed_record()
    record["diagnostics"]["worker_sampler_latency_seconds"] = 500.0

    api.assert_standard_problem_speed_record(record)

    total = record["timings"]["total_seconds"]
    assert total == pytest.approx(12.0)
    assert record["timing_fractions"]["run_fraction"] == pytest.approx(
        record["timings"]["run_seconds"] / total
    )


def test_default_speed_config_covers_8d_basic_mvn_and_all_allocations():
    api = _require_speed_api()
    config = _as_mapping(_speed_config(api))

    assert config["problem"] == "basic_mvn"
    assert config["dimension"] == 8
    assert tuple(config["allocation_targets"]) == DEFAULT_ALLOCATION_TARGETS
    for field_name in REQUIRED_CONFIG_FIELDS:
        assert field_name in config, (
            "default standard-problem speed config must make "
            f"{field_name!r} explicit."
        )
    assert config["target_num_live_points"] > 0
    assert config["live_points_per_dimension"] == pytest.approx(
        config["target_num_live_points"] / 8.0
    )
    assert config["live_point_policy"] in {
        "accepted_standard_gate",
        "50_per_dimension",
    }


def test_speed_config_can_express_historical_50d_live_point_policy():
    api = _require_speed_api()
    config = _as_mapping(_speed_config(
        api,
        live_points_per_dimension=50.0,
    ))

    assert config["dimension"] == 8
    assert config["target_num_live_points"] == 400
    assert config["live_points_per_dimension"] == pytest.approx(50.0)
    assert config["live_point_policy"] == "50_per_dimension"


def test_speed_config_rejects_invalid_phantom_burn_in_for_slice_count():
    api = _require_speed_api()

    with pytest.raises(ValueError, match="phantom_burn_in"):
        _speed_config(api, num_slices=2, phantom_burn_in=4)


def test_real_basic_mvn_builder_is_full_8d_reference():
    api = _require_speed_api()

    model, log_z_ref = api.build_basic_mvn_problem()

    assert model.U_ndims() == 8
    prior_mu, prior_cov, data_mu, data_cov = api._basic_mvn_parameters()
    expected_log_z_ref = api._log_normal(
        data_mu,
        prior_mu,
        prior_cov + data_cov,
    )
    assert log_z_ref == pytest.approx(expected_log_z_ref)
    assert log_z_ref == pytest.approx(-24.606462553878423)


def test_collect_speed_records_uses_one_local_load_balancer_per_target(
        monkeypatch,
):
    api = _require_speed_api()
    created_clients = _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)
    config = _speed_config(
        api,
        seed=17,
        max_samples=9,
        mc_sample_count=4,
        target_num_live_points=16,
        worker_specs=("cpu:*:2",),
    )

    records = api.collect_standard_problem_speed_records(config=config)

    allocation_targets = [record["metadata"]["allocation_target"] for record in records]
    assert tuple(allocation_targets) == DEFAULT_ALLOCATION_TARGETS
    assert len(created_clients) == len(DEFAULT_ALLOCATION_TARGETS)
    for client, allocation_target in zip(created_clients, DEFAULT_ALLOCATION_TARGETS):
        assert client.address == "local"
        assert client.entered
        assert client.exited
        assert client.worker_specs == [("cpu:*:2",)]
        assert client.run_allocation_targets == [allocation_target]
        assert len(client.get_nested_sampler_calls) == 1
        call = client.get_nested_sampler_calls[0]
        assert call["target_num_live_points"] == 16
        assert call["max_samples"] == 9
        assert call["shell_size"] == config.shell_size
        assert bool(call["collect_phantoms"])
        sampler = call["sampler"]
        assert sampler.num_slices == config.num_slices
        assert sampler.phantom_burn_in == config.phantom_burn_in
        assert sampler.direction_kernel == config.direction_kernel


def test_collect_speed_records_include_full_metadata_and_worker_inner_time(
        monkeypatch,
):
    api = _require_speed_api()
    _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)
    config = _speed_config(
        api,
        allocation_targets=("posterior_improving",),
        seed=19,
        max_samples=11,
        mc_sample_count=6,
        target_num_live_points=40,
        worker_specs=("cpu:*:3",),
    )

    records = api.collect_standard_problem_speed_records(config=config)

    assert len(records) == 1
    record = records[0]
    api.assert_standard_problem_speed_record(record)
    metadata = record["metadata"]
    assert metadata["problem"] == "basic_mvn"
    assert metadata["dimension"] == 8
    assert metadata["allocation_target"] == "posterior_improving"
    assert metadata["seed"] == 19
    assert metadata["worker_specs"] == ["cpu:*:3"]
    assert metadata["target_num_live_points"] == 40
    assert metadata["live_points_per_dimension"] == pytest.approx(5.0)
    assert metadata["live_point_policy"] == "custom"
    assert metadata["max_samples"] == 11
    assert metadata["mc_sample_count"] == 6
    assert metadata["shell_size"] > 0
    assert metadata["num_slices"] > 0
    assert metadata["phantom_burn_in"] >= 0
    assert metadata["direction_kernel"]
    assert record["timings"]["run_seconds"] == pytest.approx(1.0)
    assert record["timings"]["total_seconds"] == pytest.approx(4.0)
    assert record["diagnostics"]["worker_sampler_latency_seconds"] == (
        pytest.approx(1.25)
    )


def test_worker_scaling_records_vary_specs_and_report_observed_workers(
        monkeypatch,
):
    api = _require_speed_api()
    collect_worker_scaling = getattr(
        api,
        "collect_worker_scaling_speed_records",
        None,
    )
    assert callable(collect_worker_scaling), (
        "Ticket 0016 worker-scaling mode requires a public "
        "collect_worker_scaling_speed_records(...) API."
    )
    created_clients = _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)
    worker_specs_grid = (
        ("cpu:*:1",),
        ("cpu:*:3",),
    )
    config = _speed_config(
        api,
        allocation_targets=("uniform",),
        seed=29,
        max_samples=7,
        mc_sample_count=3,
        target_num_live_points=24,
    )

    records = collect_worker_scaling(
        config=config,
        worker_specs_grid=worker_specs_grid,
    )

    assert len(records) == len(worker_specs_grid)
    assert len(created_clients) == len(worker_specs_grid)
    assert [
        record["metadata"]["worker_specs"]
        for record in records
    ] == [list(worker_specs) for worker_specs in worker_specs_grid]
    assert [
        _observed_worker_count(record)
        for record in records
    ] == [
        _fake_compute_sector_worker_count(worker_specs)
        for worker_specs in worker_specs_grid
    ]
    for client, worker_specs, record in zip(
            created_clients,
            worker_specs_grid,
            records,
    ):
        assert client.address == "local"
        assert client.entered
        assert client.exited
        assert client.worker_specs == [tuple(worker_specs)]
        assert client.run_allocation_targets == ["uniform"]
        api.assert_standard_problem_speed_record(record)

    encoded = json.dumps(records, sort_keys=True)
    assert json.loads(encoded) == records


def test_collect_speed_records_are_deterministic_and_json_serializable(
        monkeypatch,
):
    api = _require_speed_api()
    _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)
    config = _speed_config(
        api,
        allocation_targets=("uniform",),
        seed=23,
        max_samples=7,
        mc_sample_count=3,
        target_num_live_points=24,
    )

    first_records = api.collect_standard_problem_speed_records(config=config)
    second_records = api.collect_standard_problem_speed_records(config=config)

    assert first_records == second_records
    encoded = json.dumps(first_records, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == first_records


def test_main_writes_json_records_with_fake_runtime(monkeypatch, capsys):
    api = _require_speed_api()
    _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)

    exit_code = api.main([
        "--allocation-target",
        "uniform",
        "--max-samples",
        "7",
        "--target-num-live-points",
        "24",
        "--mc-sample-count",
        "3",
    ])

    assert exit_code == 0
    decoded = json.loads(capsys.readouterr().out)
    assert isinstance(decoded, list)
    assert len(decoded) == 1
    assert decoded[0]["metadata"]["allocation_target"] == "uniform"
    api.assert_standard_problem_speed_record(decoded[0])


def test_main_worker_scaling_writes_json_records(monkeypatch, capsys):
    api = _require_speed_api()
    _install_fake_runtime(monkeypatch, api)
    _install_fake_clock(monkeypatch, api)

    exit_code = api.main([
        "--worker-scaling",
        "--allocation-target",
        "uniform",
        "--worker-spec",
        "cpu:*:1",
        "--worker-spec",
        "cpu:*:3",
        "--max-samples",
        "7",
        "--target-num-live-points",
        "24",
        "--mc-sample-count",
        "3",
    ])

    assert exit_code == 0
    decoded = json.loads(capsys.readouterr().out)
    assert [
        record["metadata"]["worker_specs"]
        for record in decoded
    ] == [["cpu:*:1"], ["cpu:*:3"]]
    assert [_observed_worker_count(record) for record in decoded] == [
        _fake_compute_sector_worker_count(("cpu:*:1",)),
        _fake_compute_sector_worker_count(("cpu:*:3",)),
    ]
    for record in decoded:
        api.assert_standard_problem_speed_record(record)


def test_mc_shrinkage_timing_synchronizes_samples_before_timer_stops(
        monkeypatch,
):
    api = _require_speed_api()
    clock = _McSynchronizationClock()
    returned_samples: list[_FakeMcShrinkageSamples] = []

    def sample_factory(num_samples: int) -> _FakeMcShrinkageSamples:
        samples = _FakeMcShrinkageSamples(num_samples=num_samples)
        returned_samples.append(samples)
        clock.require_next_timer_after_sync(samples)
        return samples

    _install_fake_runtime(
        monkeypatch,
        api,
        mc_shrinkage_sample_factory=sample_factory,
    )
    _install_fake_clock(monkeypatch, api, clock=clock)
    config = _speed_config(
        api,
        allocation_targets=("uniform",),
        seed=31,
        max_samples=7,
        mc_sample_count=3,
        target_num_live_points=24,
    )

    records = api.collect_standard_problem_speed_records(config=config)

    assert len(records) == 1
    assert len(returned_samples) == 1
    assert returned_samples[0].synchronized
    assert returned_samples[0].events, (
        "MC shrinkage timing must materialize or block returned samples before "
        "stopping the timer."
    )
    api.assert_standard_problem_speed_record(records[0])


class _StepClock:
    def __init__(self):
        self.current = 0.0

    def __call__(self) -> float:
        value = self.current
        self.current += 1.0
        return value


class _McSynchronizationClock(_StepClock):
    def __init__(self):
        super().__init__()
        self._pending_sample: _FakeMcShrinkageSamples | None = None

    def require_next_timer_after_sync(
            self,
            samples: "_FakeMcShrinkageSamples",
    ) -> None:
        self._pending_sample = samples

    def __call__(self) -> float:
        if self._pending_sample is not None:
            samples = self._pending_sample
            self._pending_sample = None
            assert samples.synchronized, (
                "mc_shrinkage_seconds stopped before returned samples were "
                "materialized or synchronized."
            )
        return super().__call__()


class _FakeMcShrinkageSamples:
    def __init__(self, *, num_samples: int):
        self.events: list[str] = []
        self.log_Z_samples = _FakeMaterializableSamples(
            events=self.events,
            values=[-50.1 + 0.01 * idx for idx in range(num_samples)],
        )

    @property
    def synchronized(self) -> bool:
        return bool(self.events)

    def block_until_ready(self):
        self.events.append("samples.block_until_ready")
        return self


class _FakeMaterializableSamples:
    def __init__(self, *, events: list[str], values: list[float]):
        self._events = events
        self._values = values

    def block_until_ready(self):
        self._events.append("log_Z_samples.block_until_ready")
        return self

    def __array__(self, dtype=None):
        self._events.append("log_Z_samples.__array__")
        return np.asarray(self._values, dtype=dtype)

    def tolist(self) -> list[float]:
        self._events.append("log_Z_samples.tolist")
        return list(self._values)


class _FakeLoadBalancerClient:
    def __init__(
            self,
            address: str,
            created_clients: list[Any],
            mc_shrinkage_sample_factory: Any = None,
    ):
        self.address = address
        self.created_clients = created_clients
        self.mc_shrinkage_sample_factory = mc_shrinkage_sample_factory
        self.entered = False
        self.exited = False
        self.worker_specs: list[tuple[str, ...]] = []
        self.get_nested_sampler_calls: list[dict[str, Any]] = []
        self.run_allocation_targets: list[str] = []
        self.compute_sectors = [SimpleNamespace(num_workers=0)]
        created_clients.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False

    def add_workers(self, worker_specs):
        worker_specs = tuple(worker_specs)
        self.worker_specs.append(worker_specs)
        self.compute_sectors = [
            SimpleNamespace(
                num_workers=_fake_compute_sector_worker_count(worker_specs),
            )
        ]

    def get_nested_sampler(self, **kwargs):
        self.get_nested_sampler_calls.append(copy.deepcopy(kwargs))
        return _FakeNestedSampler(self, kwargs)


class _FakeNestedSampler:
    def __init__(self, client: _FakeLoadBalancerClient, kwargs: Mapping[str, Any]):
        self.client = client
        self.max_samples = int(kwargs["max_samples"])

    def run_until_goal(self, *args, **kwargs):
        del args
        allocation_target = kwargs["allocation_target"]
        self.client.run_allocation_targets.append(allocation_target)
        return _FakeState(
            allocation_target=allocation_target,
            max_samples=self.max_samples,
            mc_shrinkage_sample_factory=(
                self.client.mc_shrinkage_sample_factory
            ),
        )


class _FakeState:
    def __init__(
            self,
            *,
            allocation_target: str,
            max_samples: int,
            mc_shrinkage_sample_factory: Any = None,
    ):
        self.allocation_target = allocation_target
        self.max_samples = max_samples
        self.mc_shrinkage_sample_factory = mc_shrinkage_sample_factory

    def to_result(self):
        return _FakeResult(
            allocation_target=self.allocation_target,
            max_samples=self.max_samples,
            mc_shrinkage_sample_factory=self.mc_shrinkage_sample_factory,
        )


class _FakeResult:
    def __init__(
            self,
            *,
            allocation_target: str,
            max_samples: int,
            mc_shrinkage_sample_factory: Any = None,
    ):
        self.allocation_target = allocation_target
        self.mc_shrinkage_sample_factory = mc_shrinkage_sample_factory
        self.total_num_samples = max_samples
        self.total_num_likelihood_evaluations = max_samples * 3
        self.log_Z_mean = -50.0
        self.log_Z_uncert = 0.25
        self.execution_diagnostics = SimpleNamespace(
            worker_runtime=SimpleNamespace(
                dispatch_records=(
                    {"sampler_execution_latency_seconds": 0.75},
                    SimpleNamespace(sampler_execution_latency_seconds=0.50),
                ),
            ),
        )

    def trim(self):
        return self

    def sample_mc_shrinkage(self, *, num_samples: int, **kwargs):
        del kwargs
        if self.mc_shrinkage_sample_factory is not None:
            return self.mc_shrinkage_sample_factory(num_samples)
        return SimpleNamespace(
            log_Z_samples=[-50.1 + 0.01 * idx for idx in range(num_samples)],
        )

    def get_diagnostics(self):
        return self.execution_diagnostics


def _observed_worker_count(record: Mapping[str, Any]) -> int:
    for section_name in ("metadata", "diagnostics"):
        section = record.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for field_name in ("actual_worker_count", "worker_count"):
            if field_name in section:
                worker_count = section[field_name]
                assert isinstance(worker_count, int)
                assert worker_count > 0
                return worker_count
    raise AssertionError(
        "Speed records must report the actual worker count observed from "
        "load-balancer compute sectors in metadata or diagnostics."
    )


def _fake_compute_sector_worker_count(worker_specs: tuple[str, ...]) -> int:
    return 10 + sum(_requested_worker_count(worker_spec)
                    for worker_spec in worker_specs)


def _requested_worker_count(worker_spec: str) -> int:
    match = re.search(r":(\d+)$", worker_spec)
    if match is None:
        return 1
    return int(match.group(1))


def _install_fake_runtime(
        monkeypatch,
        api,
        *,
        mc_shrinkage_sample_factory: Any = None,
) -> list[_FakeLoadBalancerClient]:
    created_clients: list[_FakeLoadBalancerClient] = []

    def fake_client_factory(address: str):
        return _FakeLoadBalancerClient(
            address=address,
            created_clients=created_clients,
            mc_shrinkage_sample_factory=mc_shrinkage_sample_factory,
        )

    monkeypatch.setattr(api, "LoadBalancerClient", fake_client_factory)
    monkeypatch.setattr(
        api,
        "build_basic_mvn_problem",
        lambda: (object(), -50.0),
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "UniDimSliceSampler",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        api,
        "TerminationCondition",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    return created_clients


def _install_fake_clock(monkeypatch, api, *, clock: Any = None) -> None:
    if clock is None:
        clock = _StepClock()
    if hasattr(api, "perf_counter"):
        monkeypatch.setattr(api, "perf_counter", clock)
    if hasattr(api, "time"):
        monkeypatch.setattr(api.time, "perf_counter", clock, raising=False)
        monkeypatch.setattr(api.time, "time", clock, raising=False)
