"""Compare audit fixes with the parent using matched scientific workloads.

Run this script with PYTHONPATH pointing at each checkout's src directory.
Use --baseline for the pre-audit sampler API and separate --output directories.
The NPZ snapshots allow exact scientific comparisons independently of timing.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfpd

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.priors import Prior
from jaxns.samples import SeedPoint
from jaxns.sampling.batching import sample_request
from jaxns.sampling.protocol import ConstrainedSampleRequest


def gaussian_model():
    x = Prior(tfpd.Normal(jnp.zeros(10), jnp.ones(10)), name="x").realise()
    return -0.5 * jnp.sum(jnp.square((x - 0.2) / 0.4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    model = Model(gaussian_model)
    width, transitions, phantoms = 100, 100, 99
    if options.baseline:
        sampler = UniDimSliceSampler(
            num_slices=transitions, collect_phantom_samples=True,
            max_phantom_samples=phantoms,
        )
    else:
        sampler = UniDimSliceSampler(
            num_slices=transitions, num_phantom_samples=phantoms,
        )
    seeds = jax.vmap(model.sample_U)(jax.random.split(jax.random.PRNGKey(1), width))
    seeds = jax.tree.map(lambda u: 0.49 + 0.02 * u, seeds)
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(307), width),
        valid=jnp.ones(width, dtype=bool),
        from_root=jnp.zeros(width, dtype=bool),
        log_L_constraints=jnp.full(width, -10.0),
        seed_points=SeedPoint(seeds, jax.vmap(model.log_likelihood)(seeds)),
        sampler_data=None,
    )
    jax.block_until_ready(request)
    execute = jax.jit(lambda data: sample_request(sampler, data, model=model))
    started = time.perf_counter()
    lowered = execute.lower(request)
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    output = jax.block_until_ready(compiled(request))
    durations = []
    for _ in range(11):
        started = time.perf_counter()
        jax.block_until_ready(compiled(request))
        durations.append(time.perf_counter() - started)
    memory = compiled.memory_analysis()
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    (options.output / "sampling.stablehlo").write_text(stablehlo)
    np.savez(options.output / "sampling.npz", *jax.tree.leaves(output))

    # End-to-end scientific equivalence includes root draws, allocation and
    # continuation keys, expected summaries, and both evidence ensembles.
    runner = NestedSampler(
        model=model, sampler=sampler, collect_phantom_samples=True,
        root_allocation_degree=300, replacement_width=100,
        initial_capacity=2048, max_samples=65536,
        allocation_target="evidence_improving", delta_K=300,
    )
    started = time.perf_counter()
    state = jax.block_until_ready(runner.run(jax.random.PRNGKey(307)))
    run_seconds = time.perf_counter() - started
    results = state.to_result().trim()
    classic = results.sample_evidence(128, key=jax.random.PRNGKey(2))
    phantom = results.sample_evidence(
        128, key=jax.random.PRNGKey(2), phantom_conditioning=True,
    )
    np.savez(options.output / "state.npz", *jax.tree.leaves(state))
    np.savez(
        options.output / "inference.npz",
        *jax.tree.leaves((results, classic.log_Z_samples, phantom.log_Z_samples)),
    )
    report = {
        "python": platform.python_version(), "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "x64": jax.config.jax_enable_x64,
        "shape": {"dimension": 10, "width": width, "transitions": transitions,
                  "phantoms": phantoms},
        "lower_seconds": lower_seconds, "compile_seconds": compile_seconds,
        "steady_median_seconds": statistics.median(durations),
        "steady_min_seconds": min(durations), "steady_max_seconds": max(durations),
        "steady_samples_seconds": durations,
        "argument_bytes": memory.argument_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "temporary_bytes": memory.temp_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
        "stablehlo_bytes": len(stablehlo.encode()),
        "run_seconds_including_compilation": run_seconds,
        "classic_samples": int(state.num_samples),
        "likelihood_evaluations": int(state.total_num_likelihood_evaluations),
        "expected_log_Z": float(results.log_Z_mean),
        "expected_uncertainty": float(results.log_Z_uncert),
        "reference_log_Z": float(-5 * np.log(1 + 1 / 0.16) - 0.2 / 1.16),
    }
    (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
