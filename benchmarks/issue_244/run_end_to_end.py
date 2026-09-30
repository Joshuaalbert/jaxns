"""Measure continuation accuracy and core time on every standard problem."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax

from cicd.tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES
from jaxns.core import NestedSampler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--cases",
        help="Optional comma-separated subset of standard problem names.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    output = {
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "device": str(jax.devices()[0]),
            "x64": bool(jax.config.x64_enabled),
            "python": platform.python_version(),
        },
        "seeds": args.seeds,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_cases = None
    if args.cases is not None:
        selected_cases = set(args.cases.split(","))
    for case in STANDARD_PROBLEM_CASES:
        if selected_cases is not None and case.name not in selected_cases:
            continue
        for phantoms in (False, True):
            model, truth = case.build_case()
            sampler = NestedSampler(
                model=model,
                collect_phantom_samples=phantoms,
            )
            for seed in range(args.seeds):
                key = jax.random.PRNGKey(seed)
                started = time.perf_counter()
                state = sampler.run(key)
                jax.block_until_ready(state)
                core_s = time.perf_counter() - started

                started = time.perf_counter()
                results = state.to_result().trim()
                jax.block_until_ready(results)
                result_s = time.perf_counter() - started
                output["records"].append({
                    "case": case.name,
                    "phantoms": phantoms,
                    "seed": seed,
                    "truth_log_Z": float(truth),
                    "log_Z_mean": float(results.log_Z_mean),
                    "log_Z_error": float(results.log_Z_mean - truth),
                    "log_Z_uncert": float(results.log_Z_uncert),
                    "ess": float(results.ess),
                    "classic_samples": int(results.total_num_samples),
                    "phantom_samples": int(results.total_phantom_samples),
                    "likelihood_evaluations": int(
                        results.total_num_likelihood_evaluations
                    ),
                    "depth_loop_iterations": int(state.depth_loop_iter),
                    "core_s": core_s,
                    "result_s": result_s,
                })
                # Preserve completed scientific records if a later model or
                # compilation is interrupted during the long release matrix.
                args.output.write_text(
                    json.dumps(output, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
