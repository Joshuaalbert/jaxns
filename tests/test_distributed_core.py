from __future__ import annotations

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
from tests.distributed_support import QuadraticEvaluator, make_toy_model


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
