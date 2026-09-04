"""Exploratory paired capacity/mixture benchmark for issue 292.

This runner changes only the initial immutable pool shape so candidate
capacities can be compared without adding a public option before evidence
selects a policy.
"""

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax

from benchmarks.issue_246.run_standard import _mode_mass
from benchmarks.issue_292.run_standard import (
    _make_nested_sampler,
    _pool_metrics,
    _tree_nbytes,
)
from jaxns.algorithm import depth
from jaxns.sampling.seeding import PhantomSeedPool


def _run_state(nested_sampler, key, capacity):
    """Run the ordinary goal loop from a state with one benchmark pool size."""
    state = nested_sampler.initialise(key)
    if capacity is not None:
        state = dataclasses.replace(
            state,
            phantom_seed_pool=PhantomSeedPool.empty(
                capacity,
                state.U_supremum,
            ),
        )

    def default_goal(current):
        if int(current.goal_loop_iter) == 0:
            return False
        return bool(depth._depth_condition_reached(
            current,
            nested_sampler.depth_condition,
        ))

    return nested_sampler.resume_until_goal(state, default_goal)


def _record(nested_sampler, truth, settings, seed, capacity):
    key = jax.random.PRNGKey(seed)
    started = time.perf_counter()
    state = _run_state(nested_sampler, key, capacity)
    jax.block_until_ready(state)
    warm_wall_s = time.perf_counter() - started
    if (
        int(state.termination_reason) != 0
        or bool(state.needs_growth)
        or not bool(state.depth_reached)
    ):
        raise RuntimeError("Scientific run did not finish normally.")

    results = state.to_result().trim()
    jax.block_until_ready(results)
    evidence = results.sample_evidence_mc(
        num_samples=256,
        conditioning="classic",
        key=jax.random.fold_in(key, 292),
    )
    jax.block_until_ready(evidence)
    mode_mass, mode_mass_truth = _mode_mass(settings["case"], results)
    return {
        **settings,
        "seed": seed,
        "log_Z_error": float(evidence.log_Z_mean) - float(truth),
        "log_Z_uncert": float(evidence.log_Z_uncert),
        "mode_mass": mode_mass,
        "mode_mass_truth": mode_mass_truth,
        "mode_mass_error": (
            None if mode_mass is None else mode_mass - mode_mass_truth
        ),
        "likelihood_evaluations": int(
            results.total_num_likelihood_evaluations
        ),
        "classic_samples": int(results.total_num_samples),
        "warm_wall_s": warm_wall_s,
        "state_bytes": _tree_nbytes(state),
        # Standard-problem models are local test closures and intentionally
        # cannot be pickled. Array leaves contain the capacity-dependent
        # checkpoint payload; static model metadata is constant across cells.
        "checkpoint_array_bytes": len(pickle.dumps(
            jax.tree.leaves(state),
            protocol=pickle.HIGHEST_PROTOCOL,
        )),
        **_pool_metrics(state, settings["phantom_seeding"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--capacity-multiplier", type=float, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capacity_multiplier < 0:
        raise ValueError("capacity multiplier must be non-negative")
    if not 0 <= args.probability <= 1:
        raise ValueError("probability must be in [0, 1]")

    enabled = args.capacity_multiplier > 0
    depth.PHANTOM_SEED_PROBABILITY = args.probability
    nested_sampler, truth, base = _make_nested_sampler(args.case, enabled)
    capacity = (
        None
        if not enabled
        else max(
            1,
            round(
                args.capacity_multiplier
                * nested_sampler.root_allocation_degree
            ),
        )
    )
    settings = {
        **base,
        "phantom_seeding": enabled,
        "pool_capacity_multiplier": args.capacity_multiplier,
        "phantom_seed_probability": args.probability,
    }

    # Compile all shapes and the full depth path once before timing seeds.
    warmup = _run_state(
        nested_sampler,
        jax.random.PRNGKey(292_000),
        capacity,
    )
    jax.block_until_ready(warmup)
    records = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        record = _record(
            nested_sampler,
            truth,
            settings,
            seed,
            capacity,
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    args.output.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
