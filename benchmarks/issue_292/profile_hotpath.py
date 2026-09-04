"""Measure compilation and device plans for one phantom-seeding depth call."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
from jax import numpy as jnp

from benchmarks.issue_246.run_standard import _depth_program, _environment
from benchmarks.issue_292.run_standard import (
    _make_nested_sampler,
    _tree_nbytes,
)
from jaxns.algorithm import depth
from jaxns.core import NestedSampler
from jaxns.mixed_precision import mp_policy
from jaxns.state import State


def _planning_program(
        nested_sampler: NestedSampler,
        state: State,
) -> tuple[dict[str, float | int | None], State]:
    """Compile one real planning boundary and return its scheduled state."""
    state = dataclasses.replace(
        state,
        allocation_loop_iter=jnp.asarray(1, mp_policy.count_dtype),
    )
    started = time.perf_counter()
    lowered = depth._start_schedule_round.lower(
        state,
        nested_sampler.depth_condition,
        shell_size=int(nested_sampler.shell_size),
        allocation_target=nested_sampler.allocation_target,
        root_degree=int(nested_sampler.root_allocation_degree),
        delta_K=int(nested_sampler.delta_K),
    )
    lower_s = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - started
    scheduled = compiled(state, nested_sampler.depth_condition)
    jax.block_until_ready(scheduled)
    memory = compiled.memory_analysis()
    return {
        "planning_lower_s": lower_s,
        "planning_compile_s": compile_s,
        "planning_hlo_bytes": len(lowered.as_text().encode()),
        "planning_temporary_bytes": (
            None if memory is None else int(memory.temp_size_in_bytes)
        ),
    }, scheduled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="basic_mvn")
    parser.add_argument(
        "--phantom-seeding",
        choices=("off", "on"),
        required=True,
    )
    parser.add_argument(
        "--phantom-seed-probability",
        type=float,
        default=0.1,
    )
    parser.add_argument("--root-degree", type=int)
    parser.add_argument("--replacement-width", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.phantom_seed_probability <= 1.0:
        raise ValueError("phantom-seed-probability must be in [0, 1].")

    depth.PHANTOM_SEED_PROBABILITY = args.phantom_seed_probability
    phantom_seeding = args.phantom_seeding == "on"
    nested_sampler, _, settings = _make_nested_sampler(
        args.case,
        phantom_seeding,
        root_degree=args.root_degree,
        shell_size=args.replacement_width,
    )
    state = nested_sampler.initialise(jax.random.PRNGKey(292))
    planning, state = _planning_program(nested_sampler, state)
    record = {
        **settings,
        "phantom_seeding": phantom_seeding,
        "environment": _environment(),
        "scheduled_state_bytes": _tree_nbytes(state),
        **planning,
        **_depth_program(nested_sampler, state),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
