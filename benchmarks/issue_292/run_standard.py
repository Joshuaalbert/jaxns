"""Run one paired scientific cell for phantom-assisted seeding."""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from benchmarks.issue_246.run_standard import _environment, _mode_mass
from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler


def _tree_nbytes(tree) -> int:
    """Return bytes occupied by every resident array leaf in a Pytree."""
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
    )


def _pool_metrics(state, phantom_seeding: bool) -> dict[str, int | float]:
    """Describe the bounded seed pool without adding runtime diagnostics."""
    if not phantom_seeding:
        return {
            "pool_capacity": 0,
            "pool_active": 0,
            "pool_staging": 0,
            "pool_active_fraction": 0.0,
        }
    pool = state.phantom_seed_pool
    if pool is None:
        raise ValueError("Phantom seeding finished without its seed pool.")
    active = int(np.sum(np.asarray(pool.active.valid)))
    staging = int(np.sum(np.asarray(pool.staging.valid)))
    capacity = int(pool.active.valid.shape[0])
    return {
        "pool_capacity": capacity,
        "pool_active": active,
        "pool_staging": staging,
        "pool_active_fraction": active / capacity,
    }


def _make_nested_sampler(case_name: str, phantom_seeding: bool):
    """Build the issue-246 standard configuration with matched retention."""
    model, truth = STANDARD_PROBLEM_CASES_BY_NAME[case_name].build_case()
    dimension = int(model.U_ndims())
    root_degree = 30 * dimension
    shell_size = min(root_degree, 10 * dimension)
    num_slices = 5 * dimension
    retained_phantoms = dimension
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        collect_phantom_samples=True,
        max_phantom_samples=retained_phantoms,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        shell_size=shell_size,
        max_samples=100 * root_degree,
        collect_phantom_samples=True,
        max_phantom_samples=retained_phantoms,
        phantom_seeding=phantom_seeding,
        sampler=sampler,
    )
    settings = {
        "case": case_name,
        "truth_log_Z": float(truth),
        "dimension": dimension,
        "root_degree": root_degree,
        "replacement_width": shell_size,
        "num_slices": num_slices,
        "configured_phantom_capacity": retained_phantoms,
        "dlogZ": float(nested_sampler.depth_condition.dlogZ),
    }
    return nested_sampler, truth, settings


def _run_one(
        nested_sampler: NestedSampler,
        truth,
        settings: dict,
        *,
        seed: int,
        phantom_seeding: bool,
        mc_draws: int,
) -> dict:
    """Measure one warmed run and its classic-shrinkage evidence."""
    key = jax.random.PRNGKey(seed)  # [2]
    started = time.perf_counter()
    state = nested_sampler.run(key)
    jax.block_until_ready(state)
    warm_wall_s = time.perf_counter() - started
    state_bytes = _tree_nbytes(state)

    results = state.to_result().trim()
    jax.block_until_ready(results)
    # Use the same MC key in both arms. The race trees may differ, but shared
    # gamma randomness reduces noise in the paired exploration comparison.
    evidence = results.sample_evidence_mc(
        num_samples=mc_draws,
        conditioning="classic",
        key=jax.random.fold_in(key, 292),
    )
    jax.block_until_ready(evidence)
    log_Z_mean = float(evidence.log_Z_mean)
    log_Z_uncert = float(evidence.log_Z_uncert)
    log_Z_error = log_Z_mean - float(truth)

    mode_mass, mode_mass_truth = _mode_mass(settings["case"], results)
    if mode_mass is None:
        minor_mode_mass = None
        minor_mode_mass_truth = None
        mode_lost = None
    else:
        # Track whichever analytic component is smaller. A practical loss
        # means less than ten percent of that component's true posterior mass.
        if mode_mass_truth <= 0.5:
            minor_mode_mass = mode_mass
            minor_mode_mass_truth = mode_mass_truth
        else:
            minor_mode_mass = 1.0 - mode_mass
            minor_mode_mass_truth = 1.0 - mode_mass_truth
        mode_lost = minor_mode_mass < 0.1 * minor_mode_mass_truth

    record = dict(settings)
    record.update({
        "phantom_seeding": phantom_seeding,
        "seed": seed,
        "mc_draws": mc_draws,
        "log_Z_mean": log_Z_mean,
        "log_Z_uncert": log_Z_uncert,
        "log_Z_error": log_Z_error,
        "log_Z_z": log_Z_error / log_Z_uncert,
        "expected_log_Z_mean": float(results.log_Z_mean),
        "expected_log_Z_uncert": float(results.log_Z_uncert),
        "mode_mass": mode_mass,
        "mode_mass_truth": mode_mass_truth,
        "mode_mass_error": (
            None if mode_mass is None else mode_mass - mode_mass_truth
        ),
        "minor_mode_mass": minor_mode_mass,
        "minor_mode_mass_truth": minor_mode_mass_truth,
        "mode_lost": mode_lost,
        "likelihood_evaluations": int(
            results.total_num_likelihood_evaluations
        ),
        "classic_samples": int(results.total_num_samples),
        "retained_phantom_samples": int(results.total_phantom_samples),
        "retained_phantom_capacity": int(results.log_L_phantom.shape[1]),
        "goal_loops": int(state.goal_loop_iter),
        "warm_wall_s": warm_wall_s,
        "state_bytes": state_bytes,
        "state_sample_capacity": int(state.samples.log_likelihoods.shape[0]),
        **_pool_metrics(state, phantom_seeding),
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--phantom-seeding",
        choices=("off", "on"),
        required=True,
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in range(30)),
    )
    parser.add_argument("--warmup-seed", type=int, default=292_000)
    parser.add_argument("--mc-draws", type=int, default=2048)
    parser.add_argument("--source-id", default="working-tree")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.case not in STANDARD_PROBLEM_CASES_BY_NAME:
        raise ValueError(f"Unknown standard problem: {args.case}")
    if args.mc_draws <= 1:
        raise ValueError("mc-draws must exceed one to estimate uncertainty.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)

    phantom_seeding = args.phantom_seeding == "on"
    nested_sampler, truth, settings = _make_nested_sampler(
        args.case,
        phantom_seeding,
    )
    base = {
        **settings,
        "source_id": args.source_id,
        "environment": _environment(),
    }

    # Compile and execute the exact specialised path before timing. The
    # warm-up seed is not included in the scientific ensemble.
    warmup = nested_sampler.run(jax.random.PRNGKey(args.warmup_seed))
    jax.block_until_ready(warmup)

    for seed in [int(value) for value in args.seeds.split(",")]:
        record = _run_one(
            nested_sampler,
            truth,
            base,
            seed=seed,
            phantom_seeding=phantom_seeding,
            mc_draws=args.mc_draws,
        )
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
