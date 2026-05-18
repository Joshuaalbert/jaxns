from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxns.core_distributed as core_distributed
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.runtime import LoadBalancerClient
from jaxns.samples import SeedPoint
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from tests.test_v3_pure_core_contract import _discover_boundary_schema_types
from tests.test_v3_pure_core_contract import _field_names
from tests.test_v3_pure_core_contract import _schema_pair_by_role
from tests.distributed_support import QuadraticEvaluator, make_toy_model


_MISSING = object()


def _run_until_sample_count(
        runner: NestedSampler,
        *,
        max_samples: int,
        key,
) -> State:
    state = runner.run_until_goal(
        goal_cond=lambda state: int(state.num_samples) >= max_samples,
        depth_cond=TerminationCondition(max_samples=max_samples),
        allocation_target="uniform",
        key=key,
        max_goal_iterations=8,
    )
    assert isinstance(state, State)
    assert int(state.num_samples) == max_samples
    return state


def _assert_valid_state_and_result(
        state: State,
        *,
        expected_samples: int,
) -> None:
    result = state.to_result()
    num_samples = int(state.num_samples)

    assert num_samples == expected_samples
    assert int(state.root_out_degree + jnp.sum(state.samples.out_degree)) == (
        num_samples
    )
    assert bool(
        jnp.all(jnp.diff(state.samples.log_likelihoods[:num_samples]) >= 0.0)
    )
    assert int(result.total_num_samples) == expected_samples
    assert int(result.total_num_likelihood_evaluations) >= expected_samples
    assert bool(jnp.isfinite(result.log_Z_mean))
    assert bool(jnp.isfinite(result.log_Z_uncert))
    assert bool(
        jnp.all(state.samples.log_likelihoods[:num_samples] > -jnp.inf)
    )


def _accepted_runtime_records_for_runner(
        runner: NestedSampler,
        lb: LoadBalancerClient,
) -> tuple[object, ...]:
    runner_id = runner.runtime_runner_identity.runner_id
    records = tuple(
        record
        for record in lb.coordinator_dispatch_records
        if record.runner_id == runner_id and record.status == "accepted"
    )
    assert records
    return records


def _runtime_core_boundary_records(
        runner: NestedSampler,
        lb: LoadBalancerClient,
) -> tuple[object, ...]:
    accessor_names = (
        "core_boundary_records",
        "core_work_result_boundary_records",
        "runtime_core_boundary_records",
        "pure_core_boundary_records",
        "core_boundary_trace",
        "work_result_boundary_trace",
    )
    for owner in (runner, lb):
        for accessor_name in accessor_names:
            value = getattr(owner, accessor_name, None)
            if value is None:
                continue
            records = value() if callable(value) else value
            if isinstance(records, (str, bytes)):
                continue
            try:
                records = tuple(records)
            except TypeError:
                records = (records,)
            if records:
                return records
    pytest.fail(
        "Local-LB distributed execution must expose the fixed-shape "
        "pure-core work/result boundary it consumed, through a public runner "
        "or load-balancer boundary trace. Coordinator dispatch records alone "
        "are raw runtime records and do not prove Ticket 0020 core boundary "
        "consumption."
    )


def _read_field(obj: object, *field_names: str, default=_MISSING):
    for field_name in field_names:
        if isinstance(obj, Mapping) and field_name in obj:
            return obj[field_name]
        if hasattr(obj, field_name):
            return getattr(obj, field_name)
    if default is not _MISSING:
        return default
    raise AssertionError(
        f"Object {type(obj).__name__} missing any of fields {field_names}."
    )


def _fields_from_boundary_value(value: object) -> frozenset[str]:
    schema_type = _read_field(
        value,
        "schema",
        "schema_type",
        "boundary_schema",
        default=None,
    )
    if schema_type is not None:
        fields = _field_names(schema_type)
        if fields:
            return fields
    if dataclasses.is_dataclass(value):
        return frozenset(field.name for field in dataclasses.fields(value))
    annotations = getattr(value, "__annotations__", None)
    if annotations is not None:
        return frozenset(annotations)
    if isinstance(value, Mapping):
        return frozenset(str(key) for key in value)
    named_tuple_fields = getattr(value, "_fields", None)
    if named_tuple_fields is not None:
        return frozenset(named_tuple_fields)
    return frozenset()


def _record_boundary_fields(
        record: object,
        *,
        role: str,
) -> frozenset[str]:
    if role == "work":
        value = _read_field(
            record,
            "work",
            "work_buffer",
            "core_work",
            "core_work_buffer",
            "work_schema",
            "core_work_schema",
        )
    else:
        value = _read_field(
            record,
            "result",
            "result_buffer",
            "core_result",
            "core_result_buffer",
            "result_schema",
            "core_result_schema",
        )
    fields = _fields_from_boundary_value(value)
    if not fields:
        raise AssertionError(
            f"Core boundary record {type(record).__name__} exposes {role} "
            "but not field/schema metadata."
        )
    return fields


def _accepted_core_result_parent_work_ids(records: tuple[object, ...]) -> tuple[int, ...]:
    parent_work_ids: list[int] = []
    for record in records:
        result = _read_field(
            record,
            "result",
            "result_buffer",
            "core_result",
            "core_result_buffer",
        )
        ids = np.asarray(_read_field(result, "parent_work_id", "work_id"))
        valid_mask = np.asarray(
            _read_field(
                result,
                "valid_mask",
                "result_valid_mask",
                default=np.ones(ids.shape, dtype=bool),
            ),
            dtype=bool,
        )
        statuses = _read_field(
            result,
            "status",
            "status_code",
            "status_mask",
            default=np.ones(ids.shape, dtype=bool),
        )
        statuses = np.asarray(statuses)
        accepted_mask = valid_mask & _accepted_status_mask(statuses)
        parent_work_ids.extend(int(value) for value in ids[accepted_mask])
    if not parent_work_ids:
        raise AssertionError(
            "Core result buffers must expose accepted parent_work_id/work_id "
            "values through valid result masks."
        )
    return tuple(parent_work_ids)


def _accepted_status_mask(statuses: np.ndarray) -> np.ndarray:
    if statuses.dtype == np.bool_:
        return statuses
    if np.issubdtype(statuses.dtype, np.integer):
        return statuses > 0
    vectorized = np.vectorize(lambda value: str(value).lower() == "accepted")
    return vectorized(statuses)


def _accepted_record_parent_work_ids(records: tuple[object, ...]) -> tuple[int, ...]:
    ids = []
    for record in records:
        if str(_read_field(record, "status", default="accepted")).lower() != "accepted":
            continue
        ids.append(
            int(
                _read_field(
                    record,
                    "parent_work_id",
                    "core_parent_work_id",
                    "core_result_parent_work_id",
                )
            )
        )
    if not ids:
        raise AssertionError(
            "Accepted runtime records must carry the parent_work_id from the "
            "core result buffer; task_id/attempt_id alone are runtime "
            "completion identities."
        )
    return tuple(ids)


def test_distributed_slice_sampler_preserves_sampler_contract():
    model = make_toy_model()
    sampler = DistributedUniDimSliceSampler(
        model=model,
        evaluator=QuadraticEvaluator(),
        num_slices=5,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    (
        u_sample,
        log_likelihood,
        num_likelihood_evaluations,
        phantom_samples,
    ) = sampler.get_sample(
        key=jax.random.PRNGKey(0),
        log_L_constraint=jnp.asarray(-0.05),
        seed_point=SeedPoint(U0=jnp.asarray(0.25), log_L0=jnp.asarray(0.0)),
    )

    assert sampler.num_phantom() == 3
    assert 0.0 <= float(u_sample) <= 1.0
    assert float(log_likelihood) > -0.05
    assert int(num_likelihood_evaluations) >= 1
    assert phantom_samples.U_samples is None
    assert phantom_samples.log_L.shape == (sampler.num_phantom(),)
    assert phantom_samples.valid_mask.shape == (sampler.num_phantom(),)
    assert bool(jnp.all(phantom_samples.valid_mask))
    assert bool(jnp.all(phantom_samples.log_L > -0.05))


def test_load_balanced_nested_sampler_run_returns_valid_state_and_result():
    model = make_toy_model()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            target_num_live_points=8,
            max_samples=24,
            shell_size=4,
            termination_condition=TerminationCondition(max_samples=24),
            store_phantom_samples=False,
            batch_size=None,
        )
        state = _run_until_sample_count(
            runner,
            max_samples=24,
            key=jax.random.PRNGKey(1),
        )
        accepted_records = _accepted_runtime_records_for_runner(runner, lb)

    _assert_valid_state_and_result(state, expected_samples=24)
    assert state.samples.phantom_samples.U_samples is None
    assert state.samples.phantom_samples.log_L is not None
    assert {record.sector_id for record in accepted_records} == {
        "sector-000001"
    }
    assert {
        record.task_id
        for record in accepted_records
    } == set(runner.runtime_acceptance_ledger.accepted_task_ids)


def test_load_balanced_nested_sampler_matches_direct_v3_result_invariants():
    model = make_toy_model()
    direct_sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    runtime_sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    direct_runner = NestedSampler(
        model=model,
        sampler=direct_sampler,
        target_num_live_points=8,
        max_samples=24,
        shell_size=4,
        termination_condition=TerminationCondition(max_samples=24),
        store_phantom_samples=True,
        batch_size=None,
    )
    key = jax.random.PRNGKey(7)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runtime_runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            sampler=runtime_sampler,
            target_num_live_points=8,
            max_samples=24,
            shell_size=4,
            termination_condition=TerminationCondition(max_samples=24),
            store_phantom_samples=True,
            batch_size=None,
        )
        runtime_state = _run_until_sample_count(
            runtime_runner,
            max_samples=24,
            key=key,
        )
        runtime_records = _accepted_runtime_records_for_runner(
            runtime_runner,
            lb,
        )

    direct_state = _run_until_sample_count(
        direct_runner,
        max_samples=24,
        key=key,
    )
    direct_result = direct_state.to_result()
    runtime_result = runtime_state.to_result()
    log_z_tolerance = float(
        direct_result.log_Z_uncert + runtime_result.log_Z_uncert
    )
    supremum_tolerance = 0.2

    _assert_valid_state_and_result(runtime_state, expected_samples=24)
    _assert_valid_state_and_result(direct_state, expected_samples=24)
    assert int(runtime_state.num_samples) == int(direct_state.num_samples)
    assert len(runtime_records) >= (
        int(runtime_state.num_samples)
        - int(runtime_runner.target_num_live_points)
    )
    assert runtime_state.samples.phantom_samples.log_L is not None
    assert runtime_state.samples.phantom_samples.valid_mask.shape[1] > 0
    np.testing.assert_allclose(
        np.asarray(runtime_result.log_Z_mean),
        np.asarray(direct_result.log_Z_mean),
        atol=log_z_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(runtime_result.log_Z_uncert),
        np.asarray(direct_result.log_Z_uncert),
        atol=0.5 * log_z_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(runtime_result.log_L_supremum),
        np.asarray(direct_result.log_L_supremum),
        atol=supremum_tolerance,
        rtol=0.0,
    )
    assert float(runtime_result.ess) > 0.0
    assert float(direct_result.ess) > 0.0


def test_local_lb_consumes_pure_core_work_result_boundary_schemas():
    export_name, schema_pairs = _discover_boundary_schema_types()
    schemas_by_role = _schema_pair_by_role(schema_pairs)
    assert "work_schema" in schemas_by_role, (
        f"{export_name} must identify the fixed-shape core work schema."
    )
    assert "result_schema" in schemas_by_role, (
        f"{export_name} must identify the fixed-shape core result schema."
    )
    _, expected_work_type = schemas_by_role["work_schema"]
    _, expected_result_type = schemas_by_role["result_schema"]
    expected_work_fields = _field_names(expected_work_type)
    expected_result_fields = _field_names(expected_result_type)
    model = make_toy_model()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            target_num_live_points=6,
            max_samples=12,
            shell_size=3,
            termination_condition=TerminationCondition(max_samples=12),
            store_phantom_samples=False,
            batch_size=None,
        )
        _run_until_sample_count(
            runner,
            max_samples=12,
            key=jax.random.PRNGKey(31),
        )
        boundary_records = _runtime_core_boundary_records(runner, lb)

    for record in boundary_records:
        work_fields = _record_boundary_fields(record, role="work")
        result_fields = _record_boundary_fields(record, role="result")
        assert expected_work_fields <= work_fields, (
            "Local-LB runtime must consume the same fixed-shape core work "
            f"schema fields as pure core. Missing "
            f"{sorted(expected_work_fields - work_fields)}."
        )
        assert expected_result_fields <= result_fields, (
            "Local-LB runtime must produce the same fixed-shape core result "
            f"schema fields as pure core. Missing "
            f"{sorted(expected_result_fields - result_fields)}."
        )


def test_local_lb_acceptance_order_uses_core_result_parent_work_ids():
    model = make_toy_model()

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            target_num_live_points=6,
            max_samples=12,
            shell_size=3,
            termination_condition=TerminationCondition(max_samples=12),
            store_phantom_samples=False,
            batch_size=None,
        )
        _run_until_sample_count(
            runner,
            max_samples=12,
            key=jax.random.PRNGKey(37),
        )
        boundary_records = _runtime_core_boundary_records(runner, lb)
        accepted_runtime_records = _accepted_runtime_records_for_runner(
            runner,
            lb,
        )

    core_parent_work_ids = _accepted_core_result_parent_work_ids(
        boundary_records
    )
    runtime_parent_work_ids = _accepted_record_parent_work_ids(
        accepted_runtime_records
    )
    assert runtime_parent_work_ids == core_parent_work_ids, (
        "Ordered acceptance must be keyed by parent_work_id/work_id values "
        "from core result buffers. Raw runtime completion identities such as "
        "task_id or attempt_id must not define race-tree mutation order."
    )


@pytest.mark.parametrize(
    "api_name",
    ["NestedSamplerDistributed", "DistributedNestedSampler"],
)
def test_legacy_distributed_nested_sampler_directs_to_load_balancer(api_name):
    legacy_cls = getattr(core_distributed, api_name)
    model = make_toy_model()
    sampler = DistributedUniDimSliceSampler(
        model=model,
        evaluator=QuadraticEvaluator(),
        num_slices=3,
        no_step_out=True,
    )

    try:
        legacy_runner = legacy_cls(
            model=model,
            sampler=sampler,
            target_num_live_points=4,
            max_samples=8,
            shell_size=2,
            termination_condition=TerminationCondition(max_samples=8),
        )
    except (DeprecationWarning, NotImplementedError, RuntimeError) as exc:
        assert "jaxns.runtime.LoadBalancerClient" in str(exc)
        return

    with pytest.raises(
            (DeprecationWarning, NotImplementedError, RuntimeError),
            match=r"jaxns\.runtime\.LoadBalancerClient",
    ):
        legacy_runner.run(jax.random.PRNGKey(2))
