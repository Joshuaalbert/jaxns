"""Measure repeated scheduler coordination without likelihood cost."""

import dataclasses
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from cicd.tests.distributed_support import make_toy_model
from cicd.tests.test_core import DeterministicSampler
from jaxns.algorithm.depth import _run_depth
from jaxns.algorithm.race_tree import LikelihoodOrder
from jaxns.mixed_precision import mp_policy
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State

if sys.argv[1] == "develop":
    from jaxns.termination_condition import TerminationCondition
else:
    from jaxns.algorithm.depth import _continue_schedule_round
    from jaxns.depth_condition import DepthCondition


def make_state(size: int, width: int, batches: int) -> State:
    """Construct a deterministic valid race with a long lineage population."""
    capacity = size + batches * width
    likelihood = jnp.arange(capacity, dtype=mp_policy.measure_dtype)  # [A]
    constraints = jnp.full((capacity,), jnp.inf)  # [A]
    constraints = constraints.at[:width].set(-jnp.inf)
    constraints = constraints.at[width:size].set(
        likelihood[:size - width]
    )
    degree = jnp.zeros((capacity,), mp_policy.count_dtype)  # [A]
    degree = degree.at[:size - width].set(1)
    samples = Samples(
        log_L_constraints=constraints,
        log_likelihoods=likelihood,
        U_samples=jnp.linspace(0.0, 1.0, capacity),
        out_degree=degree,
        num_likelihood_evaluations=jnp.where(
            jnp.arange(capacity) < size,
            1,
            0,
        ),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.zeros((capacity, 0), dtype=bool),
            log_L=jnp.zeros((capacity, 0)),
        ),
    )
    key = jax.random.PRNGKey(290)  # [2]
    sample_idx = jnp.arange(capacity, dtype=mp_policy.index_dtype)  # [A]
    order = LikelihoodOrder(
        sample_indices=jnp.where(sample_idx < size, sample_idx, -1),
    )
    state_fields = {}
    if sys.argv[1] != "develop":
        state_fields["allocation_loop_iter"] = jnp.asarray(
            1,
            mp_policy.count_dtype,
        )
    return State(
        root_out_degree=jnp.asarray(width, mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.asarray(size, mp_policy.count_dtype),
        log_L_supremum=likelihood[size - 1],
        U_supremum=samples.U_samples[size - 1],
        termination_reason=jnp.asarray(0, mp_policy.count_dtype),
        model=make_toy_model(),
        goal_loop_iter=jnp.asarray(1, mp_policy.count_dtype),
        random_key=key,
        goal_key=key,
        depth_reached=jnp.asarray(True),
        likelihood_order=order,
        **state_fields,
    )


def main() -> None:
    """Print one synchronised timing and memory record as JSON."""
    implementation = sys.argv[1]
    size = int(sys.argv[2])
    batches = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    repetitions = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    if repetitions < 3:
        raise ValueError("At least two warmups and one measurement are required.")
    width = 80
    state = make_state(size, width, batches)
    sampler = DeterministicSampler()
    if implementation == "develop":
        condition = TerminationCondition()
        # Develop defines its first uniform target as root_degree + delta_K.
        # Match the candidate's first direct gap of one root-sized increment
        # so both executables accept the same ten full-width batches.
        delta_K = width
    else:
        condition = DepthCondition()
        delta_K = 1

    started = time.perf_counter()
    lowered = _run_depth.lower(
        state,
        sampler,
        condition,
        shell_size=width,
        allocation_target="uniform",
        root_degree=width,
        delta_K=delta_K,
        max_samples=size + batches * width,
    )
    lower_seconds = time.perf_counter() - started
    print(f"lowered in {lower_seconds:.3f} s", file=sys.stderr, flush=True)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    print(f"compiled in {compile_seconds:.3f} s", file=sys.stderr, flush=True)

    def run_candidate(initial):
        current = initial
        schedule_calls = 0
        while int(current.num_samples) < size + batches * width:
            current = _run_depth(
                current,
                sampler,
                condition,
                shell_size=width,
                allocation_target="uniform",
                root_degree=width,
                delta_K=delta_K,
                max_samples=size + batches * width,
            )
            schedule_calls += 1
            if int(current.termination_reason) != 0:
                break
            if bool(current.depth_reached):
                current, schedule, _ = _continue_schedule_round(
                    current,
                    current.scheduler_data,
                    condition,
                    shell_size=width,
                )
                current = dataclasses.replace(
                    current,
                    depth_reached=jnp.asarray(False),
                    scheduler_data=schedule,
                )
        return current, schedule_calls

    times = []
    output = None
    schedule_calls = 1
    for repetition in range(repetitions):
        started = time.perf_counter()
        if implementation == "develop":
            output = compiled(state, sampler, condition)
        else:
            output, schedule_calls = run_candidate(state)
        jax.block_until_ready(output)
        times.append(time.perf_counter() - started)
        print(
            f"execution {repetition + 1}/{repetitions}: "
            f"{times[-1]:.3f} s",
            file=sys.stderr,
            flush=True,
        )
    memory = compiled.memory_analysis()
    record = {
        "implementation": implementation,
        "size": size,
        "width": width,
        "batches": batches,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "warm_median_seconds": statistics.median(times[2:]),
        "warm_min_seconds": min(times[2:]),
        "warm_max_seconds": max(times[2:]),
        "hlo_bytes": len(lowered.as_text().encode()),
        "argument_bytes": memory.argument_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "temp_bytes": memory.temp_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
        "num_samples": int(output.num_samples),
        "schedule_calls": schedule_calls,
        "repetitions": repetitions,
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
    }
    if implementation != "develop":
        schedule = output.scheduler_data
        safe_idx = jnp.maximum(schedule.seed_reservoir_idx, 0)
        birth = output.samples.log_L_constraints[safe_idx]
        likelihood = output.samples.log_likelihoods[safe_idx]
        eligible = (
            schedule.valid[:, None]
            & schedule.seed_reservoir_valid[None, :]
            & (birth[None, :] <= schedule.log_L_constraint[:, None])
            & (likelihood[None, :] > schedule.log_L_constraint[:, None])
        )
        eligible_count = jnp.sum(eligible, axis=1)
        record["active_heads"] = int(jnp.sum(schedule.valid))
        record["reservoir_eligible_min"] = int(jnp.min(jnp.where(
            schedule.valid,
            eligible_count,
            schedule.seed_reservoir_idx.shape[0],
        )))
        record["reservoir_eligible_mean"] = float(jnp.sum(
            jnp.where(schedule.valid, eligible_count, 0)
        ) / jnp.maximum(jnp.sum(schedule.valid), 1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
