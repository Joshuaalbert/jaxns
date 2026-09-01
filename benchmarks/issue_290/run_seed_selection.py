"""Measure compact reservoir-rank lookup against its dense reference."""

import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp

from jaxns.algorithm.depth import _select_reservoir_slots
from jaxns.mixed_precision import mp_policy


@jax.jit
def dense_lookup(cumulative, ranks):
    """Reference the former explicit [S, C, R] comparison."""
    return jnp.argmax(
        cumulative[:, None, :] > ranks[:, :, None],
        axis=2,
    ).astype(mp_policy.index_dtype)  # [S, C]


@jax.jit
def compact_lookup(cumulative, ranks):
    """Apply one binary rank query independently to every scheduler lane."""
    return jax.vmap(_select_reservoir_slots)(cumulative, ranks)  # [S, C]


def measure(reservoir_size: int, repetitions: int) -> dict[str, float | int]:
    """Return synchronized lookup evidence for one reservoir width."""
    shell_size = 80
    proposal_width = 64
    rows = jnp.arange(shell_size, dtype=mp_policy.index_dtype)  # [S]
    slots = jnp.arange(reservoir_size, dtype=mp_policy.index_dtype)  # [R]
    eligible = ((rows[:, None] + slots[None, :]) % 3) == 0  # [S, R]
    cumulative = jnp.cumsum(
        eligible.astype(mp_policy.index_dtype),
        axis=1,
    )  # [S, R]
    counts = cumulative[:, -1]  # [S]
    proposals = jnp.arange(
        proposal_width,
        dtype=mp_policy.index_dtype,
    )  # [C]
    ranks = proposals[None, :] % counts[:, None]  # [S, C]

    dense_lowered = dense_lookup.lower(cumulative, ranks)
    compact_lowered = compact_lookup.lower(cumulative, ranks)
    dense = dense_lowered.compile()
    compact = compact_lowered.compile()
    dense_times = []
    compact_times = []
    dense_output = None
    compact_output = None
    for repetition in range(repetitions):
        if repetition % 2 == 0:
            started = time.perf_counter()
            dense_output = dense(cumulative, ranks)
            jax.block_until_ready(dense_output)
            dense_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            compact_output = compact(cumulative, ranks)
            jax.block_until_ready(compact_output)
            compact_times.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            compact_output = compact(cumulative, ranks)
            jax.block_until_ready(compact_output)
            compact_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            dense_output = dense(cumulative, ranks)
            jax.block_until_ready(dense_output)
            dense_times.append(time.perf_counter() - started)

    if not bool(jnp.array_equal(dense_output, compact_output)):
        raise AssertionError("Compact reservoir lookup changed selected slots.")
    dense_memory = dense.runtime_executable().get_compiled_memory_stats()
    compact_memory = compact.runtime_executable().get_compiled_memory_stats()
    dense_median = statistics.median(dense_times[2:])
    compact_median = statistics.median(compact_times[2:])
    return {
        "shell_size": shell_size,
        "proposal_width": proposal_width,
        "reservoir_size": reservoir_size,
        "repetitions": repetitions,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled),
        "measure_dtype": str(jnp.dtype(mp_policy.measure_dtype)),
        "dense_median_seconds": dense_median,
        "dense_min_seconds": min(dense_times[2:]),
        "dense_max_seconds": max(dense_times[2:]),
        "compact_median_seconds": compact_median,
        "compact_min_seconds": min(compact_times[2:]),
        "compact_max_seconds": max(compact_times[2:]),
        "speedup": dense_median / compact_median,
        "dense_hlo_bytes": len(dense_lowered.as_text().encode()),
        "compact_hlo_bytes": len(compact_lowered.as_text().encode()),
        "dense_temp_bytes": dense_memory.temp_size_in_bytes,
        "compact_temp_bytes": compact_memory.temp_size_in_bytes,
    }


def main() -> None:
    """Print standard and stress-scale synchronized records as JSON."""
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    if repetitions < 3:
        raise ValueError("At least two warmups and one measurement are required.")
    print(json.dumps([
        measure(reservoir_size, repetitions)
        for reservoir_size in (300, 3000)
    ]))


if __name__ == "__main__":
    main()
