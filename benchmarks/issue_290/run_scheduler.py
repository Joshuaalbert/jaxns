"""Measure repeated scheduler coordination without likelihood cost."""

import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxns

# Select benchmark fixtures from the same checkout as the implementation.
# A comparison may execute this candidate script with develop on PYTHONPATH;
# mixing candidate-only fixtures into that process would invalidate the run.
IMPLEMENTATION_ROOT = Path(jaxns.__file__).resolve().parents[2]
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from cicd.tests.distributed_support import make_toy_model
from cicd.tests.test_core import DeterministicSampler
from jaxns.algorithm.depth import _run_depth
from jaxns.algorithm.race_tree import LikelihoodOrder
from jaxns.mixed_precision import mp_policy
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State


def make_state(
        size: int,
        width: int,
        batches: int,
        *,
        include_allocation_iteration: bool,
) -> State:
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
    if include_allocation_iteration:
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
    initial_state = make_state(
        size,
        width,
        batches,
        include_allocation_iteration=implementation != "develop",
    )
    state = initial_state
    sampler = DeterministicSampler()
    planning_lowered = None
    planning_compiled = None
    planning_lower_seconds = 0.0
    planning_compile_seconds = 0.0
    if implementation == "develop":
        from jaxns.termination_condition import TerminationCondition

        condition = TerminationCondition()
        # Develop defines its first uniform target as root_degree + delta_K.
        # Match the candidate's first direct gap of one root-sized increment
        # so both executables accept the same ten full-width batches.
        delta_K = width
    else:
        from jaxns.algorithm.depth import _start_schedule_round
        from jaxns.depth_condition import DepthCondition

        condition = DepthCondition()
        # Benchmark k=1, where the additive target adds one root-sized
        # increment, so candidate and develop request the same work.
        delta_K = width
        # Production materialises the compact schedule at the Python planning
        # boundary so every large replacement call has one stable Pytree
        # signature. Lower and compile that boundary separately so the exact
        # seed index cannot disappear from the end-to-end comparison.
        started = time.perf_counter()
        planning_lowered = _start_schedule_round.lower(
            state,
            condition,
            shell_size=width,
            allocation_target="uniform",
            root_degree=width,
            delta_K=delta_K,
        )
        planning_lower_seconds = time.perf_counter() - started
        started = time.perf_counter()
        planning_compiled = planning_lowered.compile()
        planning_compile_seconds = time.perf_counter() - started
        state = planning_compiled(state, condition)
        jax.block_until_ready(state)

    started = time.perf_counter()
    if implementation == "develop":
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
    else:
        lowered = _run_depth.lower(
            state,
            sampler,
            condition,
            max_samples=size + batches * width,
        )
    lower_seconds = time.perf_counter() - started
    print(f"lowered in {lower_seconds:.3f} s", file=sys.stderr, flush=True)
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    print(f"compiled in {compile_seconds:.3f} s", file=sys.stderr, flush=True)

    times = []
    planning_times = []
    end_to_end_times = []
    output = None
    schedule_calls = 1
    for repetition in range(repetitions):
        started = time.perf_counter()
        output = compiled(state, sampler, condition)
        jax.block_until_ready(output)
        times.append(time.perf_counter() - started)
        print(
            f"execution {repetition + 1}/{repetitions}: "
            f"{times[-1]:.3f} s",
            file=sys.stderr,
            flush=True,
        )
        if planning_compiled is None:
            end_to_end_times.append(times[-1])
            continue
        started = time.perf_counter()
        planned_state = planning_compiled(
            initial_state,
            condition,
        )
        jax.block_until_ready(planned_state)
        planning_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        output = compiled(planned_state, sampler, condition)
        jax.block_until_ready(output)
        end_to_end_times.append(
            planning_times[-1] + time.perf_counter() - started
        )
    memory = compiled.memory_analysis()
    record = {
        "implementation": implementation,
        "size": size,
        "width": width,
        "batches": batches,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "planning_lower_seconds": planning_lower_seconds,
        "planning_compile_seconds": planning_compile_seconds,
        "warm_median_seconds": statistics.median(times[2:]),
        "warm_min_seconds": min(times[2:]),
        "warm_max_seconds": max(times[2:]),
        "warm_end_to_end_median_seconds": statistics.median(
            end_to_end_times[2:]
        ),
        "warm_end_to_end_min_seconds": min(end_to_end_times[2:]),
        "warm_end_to_end_max_seconds": max(end_to_end_times[2:]),
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
        "x64_enabled": bool(jax.config.x64_enabled),
        "measure_dtype": str(mp_policy.measure_dtype),
        "jaxns_module": jaxns.__file__,
    }
    if planning_compiled is not None:
        planning_memory = planning_compiled.memory_analysis()
        record.update({
            "warm_planning_median_seconds": statistics.median(
                planning_times[2:]
            ),
            "warm_planning_min_seconds": min(planning_times[2:]),
            "warm_planning_max_seconds": max(planning_times[2:]),
            "planning_hlo_bytes": len(
                planning_lowered.as_text().encode()
            ),
            "planning_argument_bytes": (
                planning_memory.argument_size_in_bytes
            ),
            "planning_output_bytes": planning_memory.output_size_in_bytes,
            "planning_temp_bytes": planning_memory.temp_size_in_bytes,
            "planning_alias_bytes": planning_memory.alias_size_in_bytes,
        })
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
