"""Measure bounded-refit and streaming-update programs separately."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import jax
from jax import numpy as jnp

from jaxns.multi_ellipsoid_utils import (
    empty_sampler_data,
    initialise_streaming_sampler_data,
    stream_sampler_data,
    update_sampler_data,
)


def _measure(name: str, function, arguments: tuple) -> dict:
    """Separate lowering, compilation, first execution, and steady execution."""
    start = time.perf_counter()
    lowered = function.lower(*arguments)
    lower_s = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    timings = []
    for _ in range(21):
        start = time.perf_counter()
        output = compiled(*arguments)
        jax.block_until_ready(output)
        timings.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    return {
        "program": name,
        "lower_s": lower_s,
        "compile_s": compile_s,
        "first_execution_s": timings[0],
        "median_steady_execution_s": statistics.median(timings[1:]),
        "hlo_bytes": len(lowered.as_text().encode()),
        "argument_bytes": (
            None if memory is None else int(memory.argument_size_in_bytes)
        ),
        "output_bytes": (
            None if memory is None else int(memory.output_size_in_bytes)
        ),
        "temporary_bytes": (
            None if memory is None else int(memory.temp_size_in_bytes)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", default="2,8,10,20,40")
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)

    for dimension in [int(value) for value in args.dimensions.split(",")]:
        population_size = args.population_size
        batch_size = 10 * dimension
        key, initial_key, next_key = jax.random.split(
            jax.random.PRNGKey(dimension),
            3,
        )
        points = jax.random.normal(key, (population_size, dimension))
        log_L = -jnp.sum(jnp.square(points), axis=1)
        mask = jnp.ones((population_size,), jnp.bool_)
        log_weights = jnp.zeros((population_size,))
        min_effective_samples = 4 * args.components * (dimension + 1)
        new_points = jax.random.normal(
            next_key,
            (batch_size, dimension),
        ) + 0.05
        new_log_L = -jnp.sum(jnp.square(new_points), axis=1) + 1.0
        new_mask = jnp.ones((batch_size,), jnp.bool_)

        warm = empty_sampler_data(args.components, dimension)
        online = empty_sampler_data(
            args.components,
            dimension,
            streaming=True,
        )
        warm_function = jax.jit(
            lambda data, values, likelihoods, weights, valid, count,
                   fit_key=initial_key,
                   information_gate=min_effective_samples: (
                update_sampler_data(
                    fit_key,
                    data,
                    values,
                    likelihoods,
                    weights,
                    valid,
                    count,
                    n_iters=args.iterations,
                    min_effective_samples=information_gate,
                    regularisation=1e-6,
                )
            )
        )
        initial_function = jax.jit(
            lambda data, values, likelihoods, valid, count,
                   fit_key=initial_key,
                   information_gate=min_effective_samples: (
                initialise_streaming_sampler_data(
                    fit_key,
                    data,
                    values,
                    likelihoods,
                    valid,
                    count,
                    n_iters=args.iterations,
                    min_effective_samples=information_gate,
                    regularisation=1e-6,
                )
            )
        )
        stream_function = jax.jit(
            lambda data, values, likelihoods, valid, count: (
                stream_sampler_data(
                    data,
                    values,
                    likelihoods,
                    valid,
                    count,
                    regularisation=1e-6,
                )
            )
        )

        # Construct healthy update inputs without including that setup in any
        # measured program. The measurements below clear compilation caches.
        fitted_warm = warm_function(
            warm,
            points,
            log_L,
            log_weights,
            mask,
            jnp.asarray(population_size),
        )
        fitted_online = initial_function(
            online,
            points,
            log_L,
            mask,
            jnp.asarray(population_size),
        )
        jax.block_until_ready((fitted_warm, fitted_online))

        programs = (
            (
                "warm_initial",
                warm_function,
                (
                    warm,
                    points,
                    log_L,
                    log_weights,
                    mask,
                    jnp.asarray(population_size),
                ),
            ),
            (
                "streaming_initial",
                initial_function,
                (
                    online,
                    points,
                    log_L,
                    mask,
                    jnp.asarray(population_size),
                ),
            ),
            (
                "warm_refresh",
                warm_function,
                (
                    fitted_warm,
                    points,
                    log_L,
                    log_weights,
                    mask,
                    jnp.asarray(2 * population_size),
                ),
            ),
            (
                "streaming_increment",
                stream_function,
                (
                    fitted_online,
                    new_points,
                    new_log_L,
                    new_mask,
                    jnp.asarray(population_size + batch_size),
                ),
            ),
        )
        for name, function, arguments in programs:
            jax.clear_caches()
            record = _measure(name, function, arguments)
            record.update({
                "dimension": dimension,
                "components": args.components,
                "population_size": population_size,
                "batch_size": batch_size,
                "jax_version": jax.__version__,
                "jaxns_version": importlib.metadata.version("jaxns"),
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "python": platform.python_version(),
                "x64": bool(jax.config.jax_enable_x64),
                "dtype": str(points.dtype),
            })
            with args.output.open("a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
