"""Measure final evidence sampling independently from the nested-sampling run."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax

import jaxns
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.results import _block_state_from_results
from tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES_BY_NAME


def _compiled_program_record(results, *, key, draws, batch_size):
    """Inspect the exact final-MC executable used by the selected source."""
    from jaxns import phantom_eval

    block_state = _block_state_from_results(results)
    if hasattr(phantom_eval, "_sample_mc_shrinkage_summary_jit"):
        lowered = phantom_eval._sample_mc_shrinkage_summary_jit.lower(
            key=key,
            log_L_constraints=results.log_L_constraints,
            log_L_classic=results.log_L,
            K_classic=results.num_live_points_per_sample,
            valid_phantom=results.valid_phantom,
            log_L_phantom=results.log_L_phantom,
            num_samples=results.total_num_samples,
            num_Z_samples=draws,
            block_state=block_state,
            batch_size=min(batch_size, draws),
            C_min=20,
        )
        program = "bounded-summary"
    else:
        # Develop before issue 249 has one monolithic full-diagnostic program;
        # its public batch argument is present but not used by the kernel.
        from jaxns.results import _sample_mc_shrinkage_with_block_state_jit

        lowered = _sample_mc_shrinkage_with_block_state_jit.lower(
            key=key,
            log_L_constraints=results.log_L_constraints,
            log_L_classic=results.log_L,
            K_classic=results.num_live_points_per_sample,
            valid_phantom=results.valid_phantom,
            log_L_phantom=results.log_L_phantom,
            total_num_samples=results.total_num_samples,
            num_Z_samples=draws,
            block_state=block_state,
            batch_size=batch_size,
            C_min=20,
        )
        program = "monolithic-diagnostics"
    compiled = lowered.compile()
    memory = compiled.memory_analysis()
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    return {
        "compiled_program": program,
        "compiled_argument_mib": memory.argument_size_in_bytes / 1024 ** 2,
        "compiled_output_mib": memory.output_size_in_bytes / 1024 ** 2,
        "compiled_temporary_mib": memory.temp_size_in_bytes / 1024 ** 2,
        "compiled_alias_mib": memory.alias_size_in_bytes / 1024 ** 2,
        "hlo_text_bytes": len(hlo.encode("utf-8")),
        "hlo_while_calls": len(re.findall(r"\bwhile\(", hlo)),
        "hlo_sort_calls": len(re.findall(r"\bsort\(", hlo)),
        "hlo_scatter_calls": len(re.findall(r"\bscatter\(", hlo)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--conditioning",
        choices=("classic", "phantom"),
        default="phantom",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--inspect-program", action="store_true")
    args = parser.parse_args()

    case = STANDARD_PROBLEM_CASES_BY_NAME[args.case]
    model, truth = case.build_case()
    ndims = int(model.U_ndims())
    root_degree = 30 * ndims
    replacement_width = min(root_degree, 10 * ndims)
    num_slices = 5 * ndims
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        no_step_out=True,
        gradient_guided=False,
        collect_phantom_samples=True,
        phantom_burn_in=num_slices - 1 - ndims,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        shell_size=replacement_width,
        max_samples=100 * root_degree,
        collect_phantom_samples=True,
        sampler=sampler,
    )
    state = nested_sampler.run(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(state)
    results = state.to_result().trim()
    jax.block_until_ready(results)

    process = psutil.Process()
    rss_before_mc = process.memory_info().rss
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    durations = []
    log_Z_means = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        samples = results.sample_evidence_mc(
            num_samples=args.draws,
            conditioning=args.conditioning,
            key=key,
            batch_size=args.batch_size,
        )
        jax.block_until_ready(samples)
        durations.append(time.perf_counter() - start)
        log_Z_means.append(float(samples.log_Z_mean))

    record = {
        "source_id": args.source_id,
        "case": args.case,
        "conditioning": args.conditioning,
        "seed": args.seed,
        "draws": args.draws,
        "batch_size": args.batch_size,
        "compile_and_first_s": durations[0],
        "steady_s": durations[1:],
        "rss_before_mc_mib": rss_before_mc / 1024 ** 2,
        "process_peak_rss_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "log_Z_means": log_Z_means,
        "truth_log_Z": float(truth),
        "classic_samples": int(results.total_num_samples),
        "phantom_samples": int(results.total_phantom_samples),
        "num_blocks": int(samples.log_L_blocks.shape[0]),
        "full_diagnostics": samples.p_gt_samples is not None,
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "jaxns_version": importlib.metadata.version("jaxns"),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
    }
    if args.inspect_program:
        record.update(
            _compiled_program_record(
                results,
                key=key,
                draws=args.draws,
                batch_size=args.batch_size,
            )
        )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
