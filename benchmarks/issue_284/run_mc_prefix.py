"""Benchmark MC evidence prefixes on a completed eight-dimensional run."""

import argparse
import dataclasses
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.model import Model
from jaxns.shrinkage.phantom import _sample_mc_shrinkage_summary_jit

tfpd = tfp.distributions


def _make_model(dimension: int) -> Model:
    def prior_model():
        x = Prior(
            tfpd.Uniform(
                low=jnp.zeros((dimension,)),
                high=jnp.ones((dimension,)),
            ),
            name="x",
        ).realise()
        return -0.5 * jnp.sum(jnp.square((x - 0.5) / 0.15))

    return Model(prior_model=prior_model)


def _compile_summary_kernel(results, prefix: int, draws: int, batch_size: int):
    block_state = results.block_data.to_block_state()
    kwargs = {
        "key": jax.random.PRNGKey(284),
        "log_L_constraints": results.log_L_constraints,
        "log_L_classic": results.log_L,
        "K_classic": results.num_live_points_per_sample,
        "valid_phantom": results.valid_phantom,
        "log_L_phantom": results.log_L_phantom[:, :prefix],
        "num_samples": results.total_num_samples,
        "num_Z_samples": draws,
        "block_state": block_state,
        "batch_size": batch_size,
        "C_min": 20.0,
    }
    lower_started = time.perf_counter()
    lowered = _sample_mc_shrinkage_summary_jit.lower(**kwargs)
    lower_s = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - compile_started
    memory = compiled.memory_analysis()
    compiler_peak_bytes = None
    memory_fields = None
    if memory is not None:
        # XLA reports these plan components separately. Aliased bytes occupy
        # both an argument and output slot, so subtract them from the peak sum.
        compiler_peak_bytes = (
            memory.argument_size_in_bytes
            + memory.output_size_in_bytes
            + memory.temp_size_in_bytes
            - memory.alias_size_in_bytes
        )
        memory_fields = {
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "temporary_bytes": memory.temp_size_in_bytes,
            "alias_bytes": memory.alias_size_in_bytes,
        }
    return {
        "lower_s": lower_s,
        "compile_s": compile_s,
        "hlo_text_bytes": len(lowered.as_text().encode("utf-8")),
        "compiler_peak_bytes": compiler_peak_bytes,
        "memory_fields": memory_fields,
    }


def _measure_public_prefix(
        results,
        prefix: int,
        draws: int,
        batch_size: int,
        repeats: int,
) -> dict:
    key = jax.random.PRNGKey(1284)

    def run_once():
        samples = results.sample_evidence_mc(
            num_samples=draws,
            conditioning="phantom",
            num_phantoms=prefix,
            key=key,
            batch_size=batch_size,
        )
        jax.block_until_ready(samples)
        return samples

    run_once()
    runtimes = []
    samples = None
    for _ in range(repeats):
        started = time.perf_counter()
        samples = run_once()
        runtimes.append(time.perf_counter() - started)

    # Public prefix selection must be exactly the same program as supplying a
    # physically sliced result. This also guards the benchmark's interpretation
    # of the compiler plan below.
    sliced = dataclasses.replace(
        results,
        log_L_phantom=results.log_L_phantom[:, :prefix],
    )
    expected = sliced.sample_evidence_mc(
        num_samples=draws,
        conditioning="phantom",
        key=key,
        batch_size=batch_size,
    )
    jax.block_until_ready(expected)
    np.testing.assert_array_equal(
        samples.log_Z_samples,
        expected.log_Z_samples,
    )
    return {
        "steady_median_s": statistics.median(runtimes),
        "steady_min_s": min(runtimes),
        "steady_max_s": max(runtimes),
        "log_Z_mean": float(samples.log_Z_mean),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dimension = 8
    retained_phantoms = dimension
    model = _make_model(dimension)
    sampler = NestedSampler(
        model=model,
        root_allocation_degree=64,
        shell_size=16,
        max_samples=args.max_samples,
        initial_capacity=args.max_samples,
        collect_phantom_samples=True,
        max_phantom_samples=retained_phantoms,
        depth_condition=DepthCondition(),
    )
    run_started = time.perf_counter()
    state = sampler.run(jax.random.PRNGKey(284))
    jax.block_until_ready(state)
    run_s = time.perf_counter() - run_started
    results = state.to_result().trim()
    jax.block_until_ready(results)

    records = []
    for prefix in (1, 4, retained_phantoms):
        record = {"num_phantoms": prefix}
        record.update(
            _compile_summary_kernel(
                results,
                prefix,
                args.draws,
                args.batch_size,
            )
        )
        record.update(
            _measure_public_prefix(
                results,
                prefix,
                args.draws,
                args.batch_size,
                args.repeats,
            )
        )
        records.append(record)

    output = {
        "environment": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "x64": bool(jax.config.jax_enable_x64),
        },
        "shape": {
            "dimension": dimension,
            "classic_samples": int(results.total_num_samples),
            "retained_phantoms": results.log_L_phantom.shape[1],
            "blocks": int(jnp.sum(results.block_data.valid)),
            "draws": args.draws,
            "batch_size": args.batch_size,
        },
        "nested_sampling_run_s": run_s,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
