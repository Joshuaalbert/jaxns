from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.random
import numpy as np

from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.core_distributed import NestedSamplerDistributed
from jaxns.fabric.node import RemoteNodeEvaluator, build_scheduler_process_manager, build_worker_process_manager
from jaxns.fabric.process_manager import create_random_ack_address, create_random_control_address
from jaxns.fabric.zmq_p2p import get_free_port
from jaxns.samples import SeedPoint
from jaxns.termination_condition import TerminationCondition
from tests.distributed_support import QuadraticEvaluator, make_quadratic_node, make_toy_model


class FabricAddresses(NamedTuple):
    ctl_pub_addr: str
    ack_rep_addr: str
    frontend_addr: str
    backend_addr: str


def make_fabric_addresses() -> FabricAddresses:
    frontend_port = get_free_port()
    backend_port = get_free_port()
    return FabricAddresses(
        ctl_pub_addr=create_random_control_address(),
        ack_rep_addr=create_random_ack_address(),
        frontend_addr=f"tcp://127.0.0.1:{frontend_port}",
        backend_addr=f"tcp://127.0.0.1:{backend_port}",
    )


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

    u_sample, log_likelihood, num_likelihood_evaluations, phantom_samples = sampler.get_sample(
        key=jax.random.PRNGKey(0),
        log_L_constraint=jnp.asarray(-0.05),
        seed_point=SeedPoint(U0=jnp.asarray(0.25), log_L0=jnp.asarray(0.0)),
    )

    assert sampler.num_phantom() == 3
    assert 0.0 <= float(u_sample) <= 1.0
    assert float(log_likelihood) > -0.05
    assert int(num_likelihood_evaluations) >= 1
    assert phantom_samples.log_L.shape == (sampler.num_phantom(),)
    assert phantom_samples.valid_mask.shape == (sampler.num_phantom(),)
    assert bool(jnp.all(phantom_samples.valid_mask))
    assert bool(jnp.all(phantom_samples.log_L > -0.05))


def test_distributed_nested_sampler_run_returns_valid_state_and_result():
    model = make_toy_model()
    sampler = DistributedUniDimSliceSampler(
        model=model,
        evaluator=QuadraticEvaluator(),
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    ns = NestedSamplerDistributed(
        model=model,
        sampler=sampler,
        target_num_live_points=8,
        max_samples=24,
        shell_size=4,
        termination_condition=TerminationCondition(max_samples=24),
        store_phantom_samples=True,
    )

    state = ns.run(jax.random.PRNGKey(1))
    result = state.to_result()
    num_samples = int(state.num_samples)

    assert num_samples == 24
    assert (int(state.termination_reason) & 1) == 1
    assert int(state.root_out_degree + jnp.sum(state.samples.out_degree)) == num_samples
    assert bool(jnp.all(jnp.diff(state.samples.log_likelihoods[:num_samples]) >= 0.0))
    assert int(result.total_num_samples) == 24
    assert int(result.total_num_likelihood_evaluations) >= 24
    assert state.samples.phantom_samples.U_samples is not None
    assert bool(jnp.isfinite(result.log_Z_mean))
    assert bool(jnp.all(state.samples.log_likelihoods[:num_samples] > -jnp.inf))


def test_distributed_nested_sampler_matches_non_distributed_results():
    model = make_toy_model()
    local_sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    distributed_sampler = DistributedUniDimSliceSampler(
        model=model,
        evaluator=QuadraticEvaluator(),
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    local_ns = NestedSampler(
        model=model,
        sampler=local_sampler,
        target_num_live_points=8,
        max_samples=24,
        shell_size=4,
        termination_condition=TerminationCondition(max_samples=24),
        store_phantom_samples=True,
    )
    distributed_ns = NestedSamplerDistributed(
        model=model,
        sampler=distributed_sampler,
        target_num_live_points=8,
        max_samples=24,
        shell_size=4,
        termination_condition=TerminationCondition(max_samples=24),
        store_phantom_samples=True,
    )

    key = jax.random.PRNGKey(7)
    local_state = local_ns.run(key)
    distributed_state = distributed_ns.run(key)
    local_result = local_state.to_result()
    distributed_result = distributed_state.to_result()
    log_z_tolerance = float(local_result.log_Z_uncert + distributed_result.log_Z_uncert)
    supremum_tolerance = 0.2

    assert int(distributed_state.num_samples) == int(local_state.num_samples)
    np.testing.assert_allclose(
        np.asarray(distributed_result.log_Z_mean),
        np.asarray(local_result.log_Z_mean),
        atol=log_z_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(distributed_result.log_Z_uncert),
        np.asarray(local_result.log_Z_uncert),
        atol=0.5 * log_z_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(distributed_result.log_L_supremum),
        np.asarray(local_result.log_L_supremum),
        atol=supremum_tolerance,
        rtol=0.0,
    )
    assert float(distributed_result.ess) > 0.0
    assert float(local_result.ess) > 0.0


def test_distributed_nested_sampler_runs_over_remote_node_evaluator():
    addresses = make_fabric_addresses()
    scheduler_mgr = build_scheduler_process_manager(addresses=addresses, profile=False)
    worker_mgr = build_worker_process_manager(
        addresses=addresses,
        service_factory=make_quadratic_node,
        num_workers=2,
        profile=False,
    )

    scheduler_mgr.start_all()
    worker_mgr.start_all()
    try:
        model = make_toy_model()
        with RemoteNodeEvaluator(frontend_addr=addresses.frontend_addr, ident_prefix="distributed-ns") as evaluator:
            sampler = DistributedUniDimSliceSampler(
                model=model,
                evaluator=evaluator,
                num_slices=3,
                no_step_out=True,
            )
            ns = NestedSamplerDistributed(
                model=model,
                sampler=sampler,
                target_num_live_points=4,
                max_samples=8,
                shell_size=2,
                termination_condition=TerminationCondition(max_samples=8),
            )
            state = ns.run(jax.random.PRNGKey(2))

        result = state.to_result()
        num_samples = int(state.num_samples)
        assert num_samples == 8
        assert (int(state.termination_reason) & 1) == 1
        assert int(state.root_out_degree + jnp.sum(state.samples.out_degree)) == num_samples
        assert bool(jnp.isfinite(result.log_Z_mean))
        assert bool(jnp.all(state.samples.log_likelihoods[:num_samples] > -jnp.inf))
    finally:
        worker_mgr.stop_all()
        worker_mgr.print_tracebacks()
        scheduler_mgr.stop_all()
        scheduler_mgr.print_tracebacks()
