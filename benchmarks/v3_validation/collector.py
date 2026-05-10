"""Minimal public v3 validation collector.

This module is intentionally small: it runs cheap toy models through public
``NestedSampler`` v3 APIs and turns public result objects into the benchmark
records produced by :mod:`benchmarks.v3_validation.producers`.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.v3_validation import producers
from benchmarks.v3_validation.deterministic_fixtures import (
    analytic_evidence_fixtures,
)
from benchmarks.v3_validation.schema_checks import (
    REQUIRED_PERFORMANCE_GUARDRAIL_NAMES,
)
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState
from jaxns.termination_condition import TerminationCondition
from jaxns.v3_shrinkage import sample_gamma_weighted_phantom_probabilities


QUADRATIC_CENTRE = 0.25
QUADRATIC_PROBLEM_NAME = "quadratic_unit_interval"
SOFTWARE_COMMIT = "ticket-0010-collector-skeleton"


@dataclasses.dataclass(frozen=True, slots=True)
class QuadraticUnitIntervalModel(PureDataclassPytree):
    """One-dimensional toy model with analytic evidence."""

    centre: float = QUADRATIC_CENTRE

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        return -jnp.square(jnp.asarray(U) - self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


@dataclasses.dataclass(frozen=True, slots=True)
class PlateauStepModel(PureDataclassPytree):
    """Three-atom likelihood surface used for plateau block diagnostics."""

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        u = jnp.asarray(U)
        likelihood = jnp.where(
            u < 0.5,
            1.0,
            jnp.where(u < 0.8, 5.0, 10.0),
        )
        return jnp.log(likelihood)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


QuadraticUnitIntervalModel.register_pytree()
PlateauStepModel.register_pytree()


def collect_minimal_v3_validation_records(
        seeds: Sequence[int],
        max_samples: int,
        evidence_sample_count: int,
) -> dict[str, list[dict[str, Any]]]:
    """Run cheap public v3 samplers and collect benchmark schema records."""
    if len(tuple(seeds)) < 2:
        raise ValueError("At least two seeds are required for rollups.")
    if int(max_samples) < 4:
        raise ValueError("max_samples must be at least four.")
    if int(evidence_sample_count) < 2:
        raise ValueError("evidence_sample_count must be at least two.")

    seeds = tuple(int(seed) for seed in seeds)
    model = QuadraticUnitIntervalModel()
    logZ_ref = _quadratic_logZ_ref(model.centre)

    run_rows = []
    posterior_runs = []
    for method_index, setting in enumerate(_method_settings()):
        for seed in seeds:
            run = _execute_method_run(
                model=model,
                seed=seed,
                method_index=method_index,
                max_samples=int(max_samples),
                evidence_sample_count=int(evidence_sample_count),
                setting=setting,
                logZ_ref=logZ_ref,
            )
            run_rows.append(run["calibration_run"])
            posterior_runs.append(run["posterior_run"])

    evidence_calibration = producers.produce_multi_seed_evidence_calibration(
        run_rows,
        logZ_ref=logZ_ref,
    )
    calibration_rollups = producers.produce_multi_seed_calibration_rollups(
        evidence_calibration,
    )
    rmse_vs_likelihood = producers.produce_grouped_rmse_vs_likelihood_pareto(
        _records_for_rmse_buckets(evidence_calibration),
    )
    posterior_wasserstein = producers.produce_grouped_posterior_wasserstein(
        posterior_runs,
    )
    posterior_quality = producers.produce_grouped_posterior_quality(
        posterior_runs,
    )
    plateau_equality = [
        _collect_plateau_equality_record(
            seed=seeds[0],
            max_samples=int(max_samples),
            evidence_sample_count=int(evidence_sample_count),
        )
    ]
    performance_guardrails = _collect_performance_guardrails(
        evidence_calibration[0]["metadata"],
    )
    timing_rows = _collect_timing_rows(evidence_calibration)
    timing_history: list[dict[str, Any]] = []
    for row in timing_rows:
        timing_history = producers.append_timing_history(timing_history, row)

    return {
        "evidence_calibration": evidence_calibration,
        "calibration_rollups": calibration_rollups,
        "plateau_equality": plateau_equality,
        "posterior_quality": posterior_quality,
        "posterior_wasserstein": posterior_wasserstein,
        "rmse_vs_likelihood": rmse_vs_likelihood,
        "performance_guardrails": performance_guardrails,
        "timing_history": timing_history,
        "timing_rows": timing_rows,
    }


def _method_settings() -> tuple[dict[str, Any], ...]:
    return (
        {
            "method_setting": {
                "method": "baseline-race-tree",
                "allocation": "uniform",
                "sampler": "slice",
            },
            "allocation_target": "uniform",
            "collect_phantom_samples": False,
            "trajectory": "straight_line",
        },
        {
            "method_setting": {
                "method": "phantom-conditioned",
                "allocation": "uniform",
                "sampler": "slice",
            },
            "allocation_target": "uniform",
            "collect_phantom_samples": True,
            "trajectory": "straight_line",
        },
        {
            "method_setting": {
                "method": "dynamic-allocation",
                "allocation": {
                    "mode": "dynamic",
                    "target": "evidence_improving",
                },
                "sampler": "slice",
            },
            "allocation_target": "evidence_improving",
            "collect_phantom_samples": False,
            "trajectory": "straight_line",
        },
        {
            "method_setting": {
                "method": "galilean",
                "allocation": "uniform",
                "sampler": "slice",
                "trajectory": "galilean",
            },
            "allocation_target": "uniform",
            "collect_phantom_samples": False,
            "trajectory": "galilean",
        },
    )


def _execute_method_run(
        *,
        model: QuadraticUnitIntervalModel,
        seed: int,
        method_index: int,
        max_samples: int,
        evidence_sample_count: int,
        setting: dict[str, Any],
        logZ_ref: float,
) -> dict[str, dict[str, Any]]:
    sampler = _make_sampler(model=model, setting=setting)
    nested_sampler = _make_nested_sampler(
        model=model,
        sampler=sampler,
        max_samples=max_samples,
    )
    key = jax.random.PRNGKey(seed + 10_000 * (method_index + 1))
    start = time.perf_counter()
    state = nested_sampler.run_until_goal(
        goal_cond=lambda current: int(current.num_samples) >= max_samples,
        depth_cond=TerminationCondition(max_samples=max_samples),
        allocation_target=setting["allocation_target"],
        key=key,
        delta_K=1,
    )
    wall_clock_seconds = time.perf_counter() - start
    block_start = time.perf_counter()
    result = state.to_result()
    block_seconds = time.perf_counter() - block_start
    evidence_samples = _sample_evidence_from_result(
        result,
        seed=seed + 101 * (method_index + 1),
        evidence_sample_count=evidence_sample_count,
        require_phantom_diagnostics=(
                setting["method_setting"]["method"]
                == "phantom-conditioned"
        ),
    )
    metadata = _metadata(
        seed=seed,
        problem=QUADRATIC_PROBLEM_NAME,
        method_setting=setting["method_setting"],
        result=result,
        wall_clock_seconds=wall_clock_seconds,
        collector_timings=_collector_timings(
            result=result,
            wall_clock_seconds=wall_clock_seconds,
            block_seconds=block_seconds,
            evidence_samples=evidence_samples,
        ),
    )
    calibration_run = {
        "metadata": metadata,
        "log_Z_samples": evidence_samples["log_Z_samples"],
        "reported_uncertainty_logZ": _positive_uncertainty(
            float(result.log_Z_uncert),
            evidence_samples["log_Z_samples"],
        ),
        "empirical_uncertainty_logZ": float(np.std(
            evidence_samples["log_Z_samples"],
            ddof=1,
        )),
        "expectation_logZ": float(result.log_Z_mean),
        "mc_shrinkage_logZ": float(np.mean(evidence_samples["log_Z_samples"])),
    }
    if setting["method_setting"]["method"] == "phantom-conditioned":
        for key in (
                "kish_participating_cluster_counts",
                "phantom_gate_active",
                "phantom_A_g",
                "phantom_B_g",
                "phantom_E_g",
                "phantom_R_g",
                "C_min",
        ):
            calibration_run[key] = evidence_samples[key]

    posterior_run = _posterior_run_from_result(
        result=result,
        metadata=metadata,
        seed=seed + 211 * (method_index + 1),
        logZ_ref=logZ_ref,
    )
    return {
        "calibration_run": calibration_run,
        "posterior_run": posterior_run,
    }


def _make_nested_sampler(
        *,
        model,
        sampler: UniDimSliceSampler,
        max_samples: int,
) -> NestedSampler:
    return NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=2,
        max_samples=max_samples,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=max_samples),
        batch_size=1,
    )


def _make_sampler(*, model, setting: dict[str, Any]) -> UniDimSliceSampler:
    if setting["trajectory"] == "galilean":
        return UniDimSliceSampler(
            model=model,
            num_slices=2,
            no_step_out=True,
            collect_phantom_samples=False,
            trajectory="galilean",
            galilean_initial_step_size=0.05,
            max_galilean_reflections=16,
            max_galilean_step_halvings=16,
            max_galilean_step_doublings=16,
        )
    return UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=bool(setting["collect_phantom_samples"]),
        phantom_burn_in=1 if setting["collect_phantom_samples"] else None,
    )


def _sample_evidence_from_result(
        result,
        *,
        seed: int,
        evidence_sample_count: int,
        require_phantom_diagnostics: bool,
) -> dict[str, Any]:
    reducer_result = dataclasses.replace(result, execution_diagnostics=None)
    key = jax.random.PRNGKey(seed)
    shrinkage_start = time.perf_counter()
    mc = reducer_result.sample_mc_shrinkage(
        num_samples=evidence_sample_count,
        key=key,
        C_min=20,
    )
    shrinkage_seconds = time.perf_counter() - shrinkage_start
    log_Z_samples = np.asarray(mc.log_Z_samples, dtype=float)
    sample_sd = np.std(log_Z_samples, ddof=1)
    if (
            log_Z_samples.ndim != 1
            or log_Z_samples.size < 2
            or not np.all(np.isfinite(log_Z_samples))
            or sample_sd <= 0.0
    ):
        raise ValueError(
            "MC shrinkage must produce finite non-degenerate log_Z samples "
            "for validation calibration."
        )
    phantom_diagnostics = _phantom_diagnostics_from_mc(
        mc,
        require=require_phantom_diagnostics,
    )
    gamma_seconds = _measure_gamma_phantom_conditioning_seconds(
        reducer_result,
        seed=seed + 17_171,
        evidence_sample_count=evidence_sample_count,
        require=require_phantom_diagnostics,
    )
    return {
        "log_Z_samples": log_Z_samples.tolist(),
        "shrinkage_seconds": float(shrinkage_seconds),
        "gamma_phantom_conditioning_seconds": float(gamma_seconds),
        **phantom_diagnostics,
    }


def _measure_gamma_phantom_conditioning_seconds(
        result,
        *,
        seed: int,
        evidence_sample_count: int,
        require: bool,
) -> float:
    block_state = _block_state_for_gamma_timing(result, require=require)
    if block_state is None:
        return 0.0
    counts = result.phantom_conditioning_diagnostics(C_min=20)
    gamma_start = time.perf_counter()
    probability_samples = sample_gamma_weighted_phantom_probabilities(
        key=jax.random.PRNGKey(seed),
        block_state=block_state,
        A_cg=counts.A_cg,
        B_cg=counts.B_cg,
        E_cg=counts.E_cg,
        num_samples=evidence_sample_count,
        C_min=20,
    )
    jax.block_until_ready(
        (
            probability_samples.p_gt_samples,
            probability_samples.p_eq_samples,
            probability_samples.p_lt_samples,
        )
    )
    return time.perf_counter() - gamma_start


def _block_state_for_gamma_timing(result, *, require: bool) -> BlockState | None:
    if result.block_size is None or result.block_incoming_K is None:
        if require:
            raise ValueError(
                "Gamma phantom-conditioning timing requires block_size and "
                "block_incoming_K on phantom-conditioned validation rows."
            )
        return None
    log_L_blocks = jnp.asarray(result.log_L_blocks)
    valid = jnp.isfinite(log_L_blocks)
    block_first_idx = result.block_first_idx
    if block_first_idx is None:
        block_first_idx = jnp.arange(log_L_blocks.shape[0], dtype=jnp.int32)
        block_first_idx = jnp.where(valid, block_first_idx, -1)
    block_out_degree = result.block_out_degree
    if block_out_degree is None:
        block_out_degree = jnp.zeros_like(result.block_size)
    return BlockState(
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=result.block_size,
        incoming_K=result.block_incoming_K,
        block_out_degree=block_out_degree,
        valid=valid,
        block_start=result.block_start,
        block_stop=result.block_stop,
        block_sample_indices=result.block_sample_indices,
    )


def _phantom_diagnostics_from_mc(
        mc,
        *,
        require: bool,
) -> dict[str, Any]:
    required_fields = (
        "kish_participating_cluster_counts",
        "phantom_gate_active",
        "phantom_A",
        "phantom_B",
        "phantom_E",
        "phantom_R",
    )
    missing = [field for field in required_fields if not hasattr(mc, field)]
    if missing:
        if require:
            raise ValueError(
                "MC shrinkage must expose Kish/gate/gamma diagnostics for "
                f"phantom-conditioned validation rows; missing {missing}."
            )
        return {}
    kish = np.asarray(mc.kish_participating_cluster_counts, dtype=float)
    gate = np.asarray(mc.phantom_gate_active, dtype=bool)
    A = np.asarray(mc.phantom_A, dtype=float)
    B = np.asarray(mc.phantom_B, dtype=float)
    E = np.asarray(mc.phantom_E, dtype=float)
    R = np.asarray(mc.phantom_R, dtype=float)
    valid = np.isfinite(kish)
    if require and not np.any(valid):
        raise ValueError(
            "MC shrinkage must expose finite Kish diagnostics for "
            "phantom-conditioned validation rows."
        )
    valid = valid & np.isfinite(A) & np.isfinite(B) & np.isfinite(E) & np.isfinite(R)
    return {
        "kish_participating_cluster_counts": [
            float(value)
            for value in kish[valid]
        ],
        "phantom_gate_active": [
            bool(value)
            for value in gate[valid]
        ],
        "phantom_A_g": [
            float(value)
            for value in A[valid]
        ],
        "phantom_B_g": [
            float(value)
            for value in B[valid]
        ],
        "phantom_E_g": [
            float(value)
            for value in E[valid]
        ],
        "phantom_R_g": [
            float(value)
            for value in R[valid]
        ],
        "C_min": 20,
    }


def _collector_timings(
        *,
        result,
        wall_clock_seconds: float,
        block_seconds: float,
        evidence_samples: dict[str, Any],
) -> dict[str, float]:
    phantom_start = time.perf_counter()
    valid_phantom = np.asarray(result.valid_phantom, dtype=bool)
    block_phantom_A = np.asarray(
        [] if result.block_phantom_A is None else result.block_phantom_A,
        dtype=float,
    )
    int(np.sum(valid_phantom))
    int(np.sum(np.isfinite(block_phantom_A)))
    phantom_seconds = time.perf_counter() - phantom_start

    serialization_start = time.perf_counter()
    payload = {
        "result_diagnostics": _result_diagnostics(result),
        "method": "collector_guardrail_payload",
    }
    pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    serialization_seconds = time.perf_counter() - serialization_start

    total_samples = max(1, int(result.total_num_samples))
    return {
        "block_construction": float(block_seconds),
        "shrinkage_sampling": float(evidence_samples["shrinkage_seconds"]),
        "phantom_counting": float(phantom_seconds),
        "gamma_phantom_conditioning": float(
            evidence_samples["gamma_phantom_conditioning_seconds"]
        ),
        "trajectories": float(wall_clock_seconds),
        "serialization": float(serialization_seconds),
        "worker_task_latency": float(wall_clock_seconds / total_samples),
    }


def _metadata(
        *,
        seed: int,
        problem: str,
        method_setting: dict[str, Any],
        result,
        wall_clock_seconds: float,
        collector_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    metadata = {
        "method_setting": dict(method_setting),
        "seed": int(seed),
        "problem": problem,
        "likelihood_evaluations": int(result.total_num_likelihood_evaluations),
        "wall_clock_seconds": float(wall_clock_seconds),
        "worker_count": 1,
        "commit": SOFTWARE_COMMIT,
        "result_diagnostics": _result_diagnostics(result),
    }
    if collector_timings is not None:
        metadata["collector_timings"] = dict(collector_timings)
    return metadata


def _result_diagnostics(result) -> dict[str, Any]:
    valid_blocks = np.isfinite(np.asarray(result.log_L_blocks, dtype=float))
    diagnostics = result.get_diagnostics()
    result_diagnostics = {
        "total_num_samples": int(result.total_num_samples),
        "total_num_likelihood_evaluations": int(
            result.total_num_likelihood_evaluations
        ),
        "log_Z_mean": float(result.log_Z_mean),
        "log_Z_uncert": float(result.log_Z_uncert),
        "ess": float(result.ess),
        "block_count": int(np.sum(valid_blocks)),
    }
    if diagnostics is not None:
        result_diagnostics["allocation_mode"] = diagnostics.allocation.mode
        result_diagnostics["sampler_mode"] = diagnostics.sampler.mode
        result_diagnostics["trajectory_mode"] = (
            diagnostics.sampler.trajectory_mode
        )
        result_diagnostics["retained_phantom_capacity"] = int(
            diagnostics.sampler.retained_phantom_capacity
        )
    return result_diagnostics


def _posterior_run_from_result(
        *,
        result,
        metadata: dict[str, Any],
        seed: int,
        logZ_ref: float,
) -> dict[str, Any]:
    del logZ_ref
    reducer_result = dataclasses.replace(result, execution_diagnostics=None)
    posterior_count = max(1, min(8, int(reducer_result.total_num_samples)))
    posterior = reducer_result.resample(
        num_samples=posterior_count,
        replace=True,
        key=jax.random.PRNGKey(seed),
    )
    posterior_samples = _as_column(
        np.asarray(posterior.X_samples[:posterior_count], dtype=float)
    )
    reference_samples = _reference_posterior_samples(sample_count=32)
    return {
        "metadata": metadata,
        "reference_samples": reference_samples,
        "posterior_samples": posterior_samples,
        "effective_sample_size": max(float(reducer_result.ess), 1e-6),
        "wasserstein_mc": _wasserstein_1d(
            reference_samples,
            posterior_samples,
        ),
        "reference_sample_count": int(reference_samples.shape[0]),
        "posterior_sample_count": int(posterior_samples.shape[0]),
    }


def _reference_posterior_samples(*, sample_count: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 2048)
    weights = np.exp(-np.square(grid - QUADRATIC_CENTRE))
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    quantiles = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    samples = np.interp(quantiles, cdf, grid)
    return samples[:, None]


def _as_column(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    return array.reshape((array.shape[0], -1))


def _wasserstein_1d(reference: np.ndarray, posterior: np.ndarray) -> float:
    reference_values = np.sort(np.ravel(reference))
    posterior_values = np.sort(np.ravel(posterior))
    paired_count = min(reference_values.size, posterior_values.size)
    ref_idx = np.linspace(0, reference_values.size - 1, paired_count)
    post_idx = np.linspace(0, posterior_values.size - 1, paired_count)
    ref = reference_values[np.round(ref_idx).astype(int)]
    post = posterior_values[np.round(post_idx).astype(int)]
    return float(np.mean(np.abs(ref - post)))


def _collect_plateau_equality_record(
        *,
        seed: int,
        max_samples: int,
        evidence_sample_count: int,
) -> dict[str, Any]:
    fixture = analytic_evidence_fixtures()[1]
    model = PlateauStepModel()
    method_setting = {
        "method": "baseline-race-tree",
        "allocation": "uniform",
        "sampler": "slice",
    }
    nested_sampler = _make_nested_sampler(
        model=model,
        sampler=UniDimSliceSampler(
            model=model,
            num_slices=3,
            no_step_out=True,
            collect_phantom_samples=False,
        ),
        max_samples=max_samples,
    )
    start = time.perf_counter()
    state = nested_sampler.run_until_goal(
        goal_cond=lambda current: int(current.num_samples) >= max_samples,
        depth_cond=TerminationCondition(max_samples=max_samples),
        allocation_target="uniform",
        key=jax.random.PRNGKey(seed + 99_001),
        delta_K=1,
    )
    wall_clock_seconds = time.perf_counter() - start
    result = state.to_result()
    evidence_samples = _sample_evidence_from_result(
        result,
        seed=seed + 99_101,
        evidence_sample_count=evidence_sample_count,
        require_phantom_diagnostics=False,
    )
    metadata = _metadata(
        seed=seed,
        problem=fixture.name,
        method_setting=method_setting,
        result=result,
        wall_clock_seconds=wall_clock_seconds,
        collector_timings=_collector_timings(
            result=result,
            wall_clock_seconds=wall_clock_seconds,
            block_seconds=0.0,
            evidence_samples=evidence_samples,
        ),
    )
    record = producers.produce_multi_seed_evidence_calibration(
        [
            {
                "metadata": metadata,
                "log_Z_samples": evidence_samples["log_Z_samples"],
                "reported_uncertainty_logZ": _positive_uncertainty(
                    float(result.log_Z_uncert),
                    evidence_samples["log_Z_samples"],
                ),
                "empirical_uncertainty_logZ": float(np.std(
                    evidence_samples["log_Z_samples"],
                    ddof=1,
                )),
                "expectation_logZ": float(result.log_Z_mean),
                "mc_shrinkage_logZ": float(np.mean(
                    evidence_samples["log_Z_samples"]
                )),
            }
        ],
        logZ_ref=fixture.logZ_ref,
    )[0]
    plateau = _plateau_from_result_blocks(result)
    record["plateau_equality"] = plateau
    return record


def _plateau_from_result_blocks(result) -> dict[str, Any]:
    log_L_blocks = np.asarray(result.log_L_blocks, dtype=float)
    valid = np.isfinite(log_L_blocks)
    block_sizes = np.asarray(result.block_size, dtype=int)
    p_eq = np.asarray(result.block_classic_p_eq_mean, dtype=float)
    finite_indices = np.where(valid)[0]
    if finite_indices.size == 0:
        raise ValueError(
            "Plateau equality requires at least one result block."
        )
    target_log_likelihood = math.log(5.0)
    selected = int(finite_indices[np.argmin(
        np.abs(log_L_blocks[finite_indices] - target_log_likelihood)
    )])
    level = float(np.exp(log_L_blocks[selected]))
    ref_by_level = {
        1.0: 0.5,
        5.0: 0.3,
        10.0: 0.2,
    }
    nearest_level = min(ref_by_level, key=lambda item: abs(item - level))
    equality_mass_ref = ref_by_level[nearest_level]
    equality_sample_count = max(1, int(block_sizes[selected]))
    hat_equality_mass = float(np.clip(p_eq[selected], 0.0, 1.0))
    return {
        "likelihood_level": float(nearest_level),
        "equality_mass_ref": float(equality_mass_ref),
        "hat_equality_mass": hat_equality_mass,
        "equality_mass_error": float(hat_equality_mass - equality_mass_ref),
        "per_sample_equality_mass": float(
            hat_equality_mass / equality_sample_count
        ),
        "source": "result_blocks",
        "result_block_count": int(np.sum(valid)),
        "equality_sample_count": int(equality_sample_count),
    }


def _collect_performance_guardrails(
        metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    timings = metadata.get("collector_timings")
    if not isinstance(timings, dict):
        raise ValueError(
            "Performance guardrails require measured collector_timings."
        )
    observed = _guardrail_observations_from_timings(timings)
    thresholds = {
        "block_construction": 1.0,
        "shrinkage_sampling": 10.0,
        "phantom_counting": 1.0,
        "gamma_phantom_conditioning": 1.0,
        "trajectories": 10.0,
        "serialization": 1.0,
        "worker_task_latency": 10.0,
    }
    rationales = {
        name: (
            f"{name} measured during the minimal public collector run; "
            "threshold is a coarse smoke-test ceiling, not a final benchmark "
            "claim."
        )
        for name in observed
    }
    return producers.produce_performance_guardrail_suite(
        metadata=metadata,
        observed_seconds_by_name=observed,
        threshold_seconds_by_name=thresholds,
        rationales_by_name=rationales,
    )


def _guardrail_observations_from_timings(
        timings: dict[str, float],
) -> dict[str, float]:
    observed = {
        "block_construction": float(timings["block_construction"]),
        "shrinkage_sampling": float(timings["shrinkage_sampling"]),
        "phantom_counting": float(timings["phantom_counting"]),
        "gamma_phantom_conditioning": float(
            timings["gamma_phantom_conditioning"]
        ),
        "trajectories": float(timings["trajectories"]),
        "serialization": float(timings["serialization"]),
        "worker_task_latency": float(timings["worker_task_latency"]),
    }
    missing = REQUIRED_PERFORMANCE_GUARDRAIL_NAMES.difference(observed)
    if missing:
        raise ValueError(f"Missing measured guardrail timings: {missing}.")
    for name, seconds in observed.items():
        if seconds < 0.0 or not np.isfinite(seconds):
            raise ValueError(
                f"Measured guardrail timing {name!r} must be finite and "
                "non-negative."
            )
    return observed


def _records_for_rmse_buckets(
        calibration_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = copy.deepcopy(list(calibration_records))
    buckets: dict[tuple[str, Any], int] = {}
    for record in records:
        metadata = record["metadata"]
        key = (
            metadata["problem"],
            _freeze_method_setting(metadata["method_setting"]),
        )
        buckets[key] = max(
            buckets.get(key, 0),
            int(metadata["likelihood_evaluations"]),
        )
    for record in records:
        metadata = record["metadata"]
        key = (
            metadata["problem"],
            _freeze_method_setting(metadata["method_setting"]),
        )
        metadata["likelihood_evaluations"] = buckets[key]
    return records


def _freeze_method_setting(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_method_setting(child))
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze_method_setting(child) for child in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_timing_rows(
        calibration_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in calibration_records:
        metadata = record["metadata"]
        diagnostics = metadata["result_diagnostics"]
        rows.append(
            {
                "metadata": metadata,
                "timings": {
                    "wall_clock_seconds": float(
                        metadata["wall_clock_seconds"]
                    ),
                    "likelihood_evaluations_per_second": (
                        float(metadata["likelihood_evaluations"])
                        / max(float(metadata["wall_clock_seconds"]), 1e-12)
                    ),
                    "blocks_per_second": (
                        float(diagnostics["block_count"])
                        / max(float(metadata["wall_clock_seconds"]), 1e-12)
                    ),
                },
            }
        )
    return rows


def _quadratic_logZ_ref(centre: float) -> float:
    integral = 0.5 * math.sqrt(math.pi) * (
        math.erf(1.0 - centre) - math.erf(-centre)
    )
    return math.log(integral)


def _positive_uncertainty(
        reported: float,
        log_Z_samples: Sequence[float],
) -> float:
    sample_sd = float(np.std(np.asarray(log_Z_samples, dtype=float), ddof=1))
    return max(float(reported), sample_sd, 1e-12)
