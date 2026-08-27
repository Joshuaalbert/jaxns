"""Continue one allocation trajectory through tighter uncertainty goals."""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import jax
import numpy as np

from benchmarks.issue_246.run_standard import _mode_mass
from benchmarks.issue_269.run_allocation_matrix import (
    _assign_components,
    _build_sampler,
    _environment,
)
from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.core import NestedSampler
from jaxns.termination_condition import TerminationCondition


def _parse_targets(value: str) -> list[float]:
    targets = [float(item.strip()) for item in value.split(",")]
    if not targets or any(target <= 0.0 for target in targets):
        raise argparse.ArgumentTypeError("Targets must be positive.")
    if any(left <= right for left, right in pairwise(targets)):
        raise argparse.ArgumentTypeError(
            "Targets must be strictly decreasing."
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="spike_slab")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        required=True,
        help="Strictly decreasing comma-separated log-Z uncertainties.",
    )
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed cannot be negative.")
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive.")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}"
        )

    repository_root = Path(__file__).resolve().parents[2]
    model, truth = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
    sampler = _build_sampler(model, collect_phantoms=False)
    # Keep the ordinary dlogZ contour rule as the compiled depth condition.
    # The sampler-level condition contains only the hard safety ceiling so a
    # completed uncertainty checkpoint remains resumable at the next target.
    depth_condition = TerminationCondition(
        dlogZ=np.log1p(1e-3),
        max_samples=args.max_samples,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=240,
        shell_size=80,
        delta_K=80,
        max_samples=args.max_samples,
        sampler=sampler,
        termination_condition=TerminationCondition(
            max_samples=args.max_samples,
        ),
    )
    payload = {
        "schema_version": 1,
        "environment": _environment(repository_root),
        "scientific_config": {
            "case": args.case,
            "truth_log_Z": float(truth),
            "seed": args.seed,
            "targets": args.targets,
            "max_samples": args.max_samples,
            "root_degree": 240,
            "delta_k": 80,
            "shell_size": 80,
            "dlogZ": float(np.log1p(1e-3)),
            "mode_loss_threshold": 1e-3,
            "stop_at_first_recovery": True,
            "compile_warm_excluded": False,
        },
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint()
    state = None
    cumulative_run_s = 0.0
    key = jax.random.PRNGKey(args.seed)
    for target in args.targets:
        def goal_condition(
                current_state,
                target_log_Z_uncert: float = target,
        ) -> bool:
            if int(current_state.goal_loop_iter) == 0:
                return False
            register = current_state.compute_termination_register()
            return float(register.log_Z_uncert) <= target_log_Z_uncert

        started = time.perf_counter()
        if state is None:
            state = nested_sampler.run_until_goal(
                goal_condition,
                depth_cond=depth_condition,
                key=key,
            )
        else:
            state = nested_sampler.resume_until_goal(
                state,
                goal_condition,
                depth_cond=depth_condition,
            )
        jax.block_until_ready(state)
        incremental_run_s = time.perf_counter() - started
        cumulative_run_s += incremental_run_s

        results = state.to_result().trim()
        jax.block_until_ready(results)
        assignments = _assign_components(args.case, results)
        mode_mass, mode_mass_truth = _mode_mass(args.case, results)
        num_samples = int(results.total_num_samples)
        log_likelihood = np.asarray(results.log_L)[:num_samples]
        weak_log_likelihood = log_likelihood[assignments == 0]
        best_weak_log_likelihood = (
            None
            if weak_log_likelihood.size == 0
            else float(np.max(weak_log_likelihood))
        )
        register_uncertainty = float(
            state.compute_termination_register().log_Z_uncert
        )
        record = {
            "target_log_Z_uncert": target,
            "goal_reached": register_uncertainty <= target,
            "goal_register_log_Z_uncert": register_uncertainty,
            "goal_loop_iterations": int(state.goal_loop_iter),
            "final_root_out_degree": int(state.root_out_degree),
            "samples": num_samples,
            "termination_reason": int(state.termination_reason),
            "mode_mass": mode_mass,
            "mode_mass_truth": mode_mass_truth,
            "mode_lost": mode_mass < 1e-3,
            "best_weak_log_likelihood": best_weak_log_likelihood,
            "log_Z_mean": float(results.log_Z_mean),
            "log_Z_error": float(results.log_Z_mean - truth),
            "incremental_run_s": incremental_run_s,
            "cumulative_run_s": cumulative_run_s,
        }
        payload["records"].append(record)
        checkpoint()
        print(json.dumps(record, sort_keys=True), flush=True)
        if not record["mode_lost"]:
            break
        if int(state.termination_reason) != 0:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
