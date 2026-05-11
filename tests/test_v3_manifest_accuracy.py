from __future__ import annotations

import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import jax
import numpy as np
import pytest

from benchmarks.v3_performance.feature_manifest import ROW_KIND_ACCURACY
from benchmarks.v3_performance.feature_manifest import SUITE_DISTRIBUTED
from benchmarks.v3_performance.feature_manifest import SUITE_PURE_CORE
from benchmarks.v3_performance.feature_manifest import V3PerformanceFeatureRow
from benchmarks.v3_performance.feature_manifest import feature_rows_for_suite
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.runtime import LoadBalancerClient
from jaxns.termination_condition import TerminationCondition
from tests.test_ns_standard_problems import STANDARD_MAX_SAMPLES
from tests.test_ns_standard_problems import STANDARD_NUM_SLICES
from tests.test_ns_standard_problems import STANDARD_PHANTOM_BURN_IN
from tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES_BY_NAME
from tests.test_ns_standard_problems import STANDARD_SHELL_SIZE
from tests.test_ns_standard_problems import STANDARD_TARGET_NUM_LIVE_POINTS


MC_SHRINKAGE_NUM_SAMPLES = 1000
DISTRIBUTED_WORKER_SPECS = ("cpu:*:2",)


def _accuracy_row_params(suite: str) -> tuple[pytest.ParameterSet, ...]:
    return tuple(
        pytest.param(row, id=row.row_id)
        for row in feature_rows_for_suite(suite, row_kind=ROW_KIND_ACCURACY)
    )


def _build_standard_problem_sampler(row: V3PerformanceFeatureRow):
    case = STANDARD_PROBLEM_CASES_BY_NAME[row.problem_fixture]
    model, log_z_ref = case.build_case()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=STANDARD_NUM_SLICES,
        phantom_burn_in=(
            STANDARD_PHANTOM_BURN_IN if row.phantom_enabled else None
        ),
        collect_phantom_samples=row.phantom_enabled,
        direction_kernel=row.direction_kernel,
        trajectory=row.trajectory_mode,
    )
    return model, log_z_ref, sampler


def _run_row_to_depth(
        runner: NestedSampler,
        row: V3PerformanceFeatureRow,
        *,
        key,
):
    depth_cond = TerminationCondition(max_samples=STANDARD_MAX_SAMPLES)
    if row.resume_pattern == "run_until_goal":
        return runner.run_until_goal(
            goal_cond=lambda state: False,
            depth_cond=depth_cond,
            allocation_target=row.allocation_target,
            key=key,
        )
    if row.resume_pattern == "resume_until_goal":
        initial_key, resume_key = jax.random.split(key)
        initial_max_samples = (
            STANDARD_TARGET_NUM_LIVE_POINTS + STANDARD_SHELL_SIZE
        )
        initial_state = runner.run_until_goal(
            goal_cond=lambda state: int(state.num_samples) >= initial_max_samples,
            depth_cond=TerminationCondition(max_samples=initial_max_samples),
            allocation_target=row.allocation_target,
            key=initial_key,
            max_goal_iterations=8,
        )
        return runner.resume_until_goal(
            state=initial_state,
            goal_cond=lambda state: False,
            depth_cond=depth_cond,
            allocation_target=row.allocation_target,
            key=resume_key,
        )
    raise ValueError(f"Unsupported resume pattern {row.resume_pattern!r}.")


def _assert_standard_problem_accuracy(state, log_z_ref):
    result = state.to_result().trim()
    mc_shrinkage_samples = result.sample_mc_shrinkage(
        num_samples=MC_SHRINKAGE_NUM_SAMPLES
    )
    log_z_samples = np.asarray(mc_shrinkage_samples.log_Z_samples)
    log_z_ensemble_mean = float(np.mean(log_z_samples))
    log_z_sample_std = float(np.std(log_z_samples))
    log_z_error = abs(log_z_ensemble_mean - float(log_z_ref))

    assert np.isfinite(float(result.log_Z_mean))
    assert np.isfinite(float(result.log_Z_uncert))
    assert int(result.total_num_samples) == STANDARD_MAX_SAMPLES
    assert log_z_sample_std > 0.0
    assert log_z_error <= 3.0 * log_z_sample_std
    return result


def _assert_row_diagnostics(result, row: V3PerformanceFeatureRow) -> None:
    diagnostics = getattr(result, "execution_diagnostics", None)

    assert diagnostics is not None
    assert diagnostics.sampler.direction_kernel_mode == row.direction_kernel
    trajectory_mode = diagnostics.sampler.trajectory_mode
    if row.trajectory_mode == "straight_line":
        assert trajectory_mode in {
            "straight_line",
            "straight_line_perfect_bracketing",
        }
    else:
        assert trajectory_mode == row.trajectory_mode
    if row.phantom_enabled:
        assert result.log_L_phantom.shape[1] > 0
        phantom_diag = result.phantom_conditioning_diagnostics(C_min=row.c_min)
        assert phantom_diag.phantom_gate_active.shape == result.log_L_blocks.shape
    else:
        assert result.log_L_phantom.shape[1] == 0


@pytest.mark.parametrize(
    "feature_row",
    _accuracy_row_params(SUITE_PURE_CORE),
)
def test_pure_core_standard_problem_accuracy(
        feature_row: V3PerformanceFeatureRow,
) -> None:
    model, log_z_ref, sampler = _build_standard_problem_sampler(feature_row)
    runner = NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=STANDARD_TARGET_NUM_LIVE_POINTS,
        max_samples=STANDARD_MAX_SAMPLES,
        shell_size=STANDARD_SHELL_SIZE,
        collect_phantom_samples=feature_row.phantom_enabled,
        store_phantom_samples=feature_row.phantom_enabled,
    )

    state = _run_row_to_depth(
        runner,
        feature_row,
        key=jax.random.PRNGKey(feature_row.seeds[0]),
    )

    result = _assert_standard_problem_accuracy(state, log_z_ref)
    _assert_row_diagnostics(result, feature_row)


@pytest.mark.parametrize(
    "feature_row",
    _accuracy_row_params(SUITE_DISTRIBUTED),
)
def test_distributed_standard_problem_accuracy(
        feature_row: V3PerformanceFeatureRow,
) -> None:
    model, log_z_ref, sampler = _build_standard_problem_sampler(feature_row)

    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(list(DISTRIBUTED_WORKER_SPECS))
        runner = lb.get_nested_sampler(
            model=model,
            collect_phantoms=feature_row.phantom_enabled,
            sampler=sampler,
            target_num_live_points=STANDARD_TARGET_NUM_LIVE_POINTS,
            max_samples=STANDARD_MAX_SAMPLES,
            shell_size=STANDARD_SHELL_SIZE,
        )
        state = _run_row_to_depth(
            runner,
            feature_row,
            key=jax.random.PRNGKey(feature_row.seeds[0]),
        )

    result = _assert_standard_problem_accuracy(state, log_z_ref)
    _assert_row_diagnostics(result, feature_row)
