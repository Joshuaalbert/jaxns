"""Issue 247 current-core accuracy and performance runner."""

import argparse
import dataclasses
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from jax import numpy as jnp

import jaxns
from cicd.tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES_BY_NAME
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler, _run_depth


def _environment():
    return {
        "jaxns_distribution_version": importlib.metadata.version("jaxns"),
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--phantoms", action="store_true")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in range(30)),
    )
    parser.add_argument("--root-multiplier", type=int)
    parser.add_argument("--shell-multiplier", type=int, default=10)
    parser.add_argument("--slice-multiplier", type=int, default=5)
    parser.add_argument("--mc-draws", type=int, default=1000)
    parser.add_argument("--mc-key", type=int)
    parser.add_argument("--c-min", type=float, default=20.0)
    parser.add_argument("--measure-depth-program", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    output_path = None if args.output is None else Path(args.output)
    if output_path is not None:
        output_path.touch(exist_ok=False)

    case = STANDARD_PROBLEM_CASES_BY_NAME[args.case]
    model, truth = case.build_case()
    ndims = int(model.U_ndims())
    root_multiplier = args.root_multiplier
    if root_multiplier is None:
        root_multiplier = 30
    root_degree = root_multiplier * ndims
    shell_size = min(
        root_degree,
        max(1, args.shell_multiplier * ndims),
    )
    num_slices = args.slice_multiplier * ndims
    retained_phantoms = ndims if args.phantoms else 0
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        no_step_out=True,
        gradient_guided=False,
        collect_phantom_samples=args.phantoms,
        phantom_burn_in=num_slices - 1 - retained_phantoms,
    )
    ns = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        shell_size=shell_size,
        max_samples=100 * root_degree,
        collect_phantom_samples=args.phantoms,
        sampler=sampler,
    )

    lower_s = None
    compile_s = None
    warm_times = []
    hlo_text_bytes = None
    hlo_operation_counts = None
    memory_stats = None
    if args.measure_depth_program:
        # Measure the compiled depth program once on a representative case,
        # independently from the multi-seed end-to-end matrix.
        example_key = jax.random.PRNGKey(0)
        depth_state = dataclasses.replace(
            ns.initialise(example_key),
            goal_loop_iter=jnp.asarray(1, dtype=jnp.int32),
        )
        lower_start = time.perf_counter()
        lowered = _run_depth.lower(
            example_key,
            depth_state,
            ns.sampler,
            ns.termination_condition,
            shell_size=shell_size,
            allocation_target=ns.allocation_target,
            root_degree=root_degree,
            delta_K=ns.delta_K,
            max_samples=ns.max_samples,
        )
        lower_s = time.perf_counter() - lower_start
        compile_start = time.perf_counter()
        compiled = lowered.compile()
        compile_s = time.perf_counter() - compile_start
        for warm_idx in range(2):
            warm_start = time.perf_counter()
            warm_state = compiled(
                jax.random.fold_in(example_key, warm_idx),
                depth_state,
                ns.sampler,
                ns.termination_condition,
            )
            jax.block_until_ready(warm_state)
            warm_times.append(time.perf_counter() - warm_start)

        hlo_text = lowered.as_text()
        hlo_text_bytes = len(hlo_text.encode())
        hlo_operation_counts = {
            operation: hlo_text.count(f"stablehlo.{operation}")
            for operation in (
                "while",
                "sort",
                "scatter",
                "dynamic_slice",
                "dynamic_update_slice",
                "custom_call",
                "map",
            )
        }
        memory = compiled.memory_analysis()
        if memory is not None:
            memory_stats = {
                name: int(getattr(memory, name))
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
                if hasattr(memory, name)
            }
    base = {
        "implementation": "current",
        "source_id": args.source_id,
        "case": args.case,
        "phantoms": args.phantoms,
        "truth_log_Z": float(truth),
        "ndims": ndims,
        "root_degree": root_degree,
        "replacement_width": shell_size,
        "allocation_increment": int(ns.delta_K),
        "num_slices": int(ns.sampler.num_slices),
        "num_retained_phantoms": int(ns.sampler.num_phantom()),
        "dlogZ": float(ns.termination_condition.dlogZ),
        "lower_s": lower_s,
        "compile_s": compile_s,
        "warmed_depth_s": warm_times[-1] if warm_times else None,
        "warmed_depth_s_all": warm_times,
        "hlo_text_bytes": hlo_text_bytes,
        "hlo_operation_counts": hlo_operation_counts,
        "memory_analysis": memory_stats,
        "environment": _environment(),
    }
    conditioning = "phantom" if args.phantoms else "classic"
    for seed in [int(value) for value in args.seeds.split(",")]:
        key = jax.random.PRNGKey(seed)
        run_start = time.perf_counter()
        state = ns.run(key)
        jax.block_until_ready(state)
        run_s = time.perf_counter() - run_start

        result_start = time.perf_counter()
        results = state.to_result().trim()
        jax.block_until_ready(results)
        result_s = time.perf_counter() - result_start

        mc_start = time.perf_counter()
        evidence_key = (
            jax.random.PRNGKey(args.mc_key)
            if args.mc_key is not None
            else jax.random.fold_in(key, 1)
        )
        evidence = results.sample_evidence_mc(
            num_samples=args.mc_draws,
            conditioning=conditioning,
            key=evidence_key,
            C_min=args.c_min,
        )
        jax.block_until_ready(evidence)
        mc_s = time.perf_counter() - mc_start
        valid_blocks = np.isfinite(np.asarray(evidence.log_L_blocks))
        gate = np.asarray(evidence.phantom_gate_active, dtype=bool)
        kish = np.asarray(evidence.kish_participating_cluster_counts)
        gated_valid_blocks = valid_blocks & gate
        classic_alpha = (
            np.asarray(evidence.classic_alpha_gt)
            + np.asarray(evidence.classic_alpha_eq)
            + np.asarray(evidence.classic_alpha_lt)
        )
        classic_p_gt = np.divide(
            np.asarray(evidence.classic_alpha_gt),
            classic_alpha,
            out=np.ones_like(classic_alpha),
            where=classic_alpha > 0,
        )
        phantom_A = np.asarray(evidence.phantom_A)
        observed_p_gt = np.divide(
            np.asarray(evidence.phantom_B),
            phantom_A,
            out=np.zeros_like(phantom_A),
            where=phantom_A > 0,
        )
        evidence_weight = np.exp(np.asarray(evidence.log_dZ_mean))
        evidence_weight = np.where(valid_blocks, evidence_weight, 0.0)
        evidence_weight /= np.sum(evidence_weight)
        diagnostic_mask = gated_valid_blocks & (phantom_A > 0)
        diagnostic_weight = np.where(
            diagnostic_mask,
            evidence_weight,
            0.0,
        )
        diagnostic_weight_sum = np.sum(diagnostic_weight)
        if diagnostic_weight_sum > 0:
            diagnostic_weight /= diagnostic_weight_sum
        record = dict(base)
        record.update({
            "seed": seed,
            "run_s": run_s,
            "result_s": result_s,
            "mc_s": mc_s,
            "log_Z_mean": float(results.log_Z_mean),
            "log_Z_uncert": float(results.log_Z_uncert),
            "log_Z_error": float(results.log_Z_mean - truth),
            "mc_log_Z_mean": float(evidence.log_Z_mean),
            "mc_log_Z_std": float(evidence.log_Z_uncert),
            "mc_log_Z_error": float(evidence.log_Z_mean - truth),
            "mc_z_score": float(
                (evidence.log_Z_mean - truth)
                / jnp.maximum(evidence.log_Z_uncert, jnp.finfo(jnp.float64).tiny)
            ),
            "classic_samples": int(results.total_num_samples),
            "phantom_samples": int(results.total_phantom_samples),
            "likelihood_evaluations": int(results.total_num_likelihood_evaluations),
            "ess": float(results.ess),
            "goal_loop_iterations": int(state.goal_loop_iter),
            "depth_loop_iterations": int(state.depth_loop_iter),
            "phantom_gate_active_blocks": int(np.sum(gated_valid_blocks)),
            "phantom_gate_active_fraction": float(
                np.mean(gate[valid_blocks]) if np.any(valid_blocks) else 0.0
            ),
            "phantom_kish_median_active": float(
                np.median(kish[gated_valid_blocks])
                if np.any(gated_valid_blocks)
                else 0.0
            ),
            "phantom_evidence_weighted_observed_minus_classic_p_gt": float(
                np.sum(diagnostic_weight * (observed_p_gt - classic_p_gt))
            ),
            "phantom_evidence_weighted_sampled_minus_classic_p_gt": float(
                np.sum(
                    diagnostic_weight
                    * np.where(
                        diagnostic_mask,
                        np.asarray(evidence.p_gt_mean) - classic_p_gt,
                        0.0,
                    )
                )
            ),
            "process_peak_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        })
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if output_path is not None:
            with output_path.open("a") as output_file:
                output_file.write(line + "\n")


if __name__ == "__main__":
    main()
