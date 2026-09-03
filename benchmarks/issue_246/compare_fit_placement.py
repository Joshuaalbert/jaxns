"""Compare device-conditional and Python-boundary GMM update placement."""

import argparse
import json
import statistics
import time

import jax
from jax import numpy as jnp

from jaxns.sampling.ellipsoid import (
    empty_sampler_data,
    update_sampler_data,
)


def _compile(lowered) -> tuple[object, float]:
    start = time.perf_counter()
    compiled = lowered.compile()
    return compiled, time.perf_counter() - start


def _temporary_bytes(compiled) -> int | None:
    memory = compiled.memory_analysis()
    return None if memory is None else int(memory.temp_size_in_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--iso-prob", type=float, default=1e-2)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()

    key = jax.random.PRNGKey(246)
    key, point_key = jax.random.split(key)
    points = jax.random.normal(
        point_key,
        shape=(args.population_size, args.dimension),
    )
    log_L = -0.5 * jnp.sum(jnp.square(points), axis=1)
    log_weights = jnp.full(
        (args.population_size,),
        -jnp.log(jnp.asarray(args.population_size, points.dtype)),
    )
    mask = jnp.ones((args.population_size,), jnp.bool_)
    data = empty_sampler_data(args.components, args.dimension)

    def update(current):
        return update_sampler_data(
            key,
            current,
            points,
            log_L,
            log_weights,
            mask,
            jnp.asarray(args.population_size),
            n_iters=args.iterations,
            iso_prob=args.iso_prob,
            regularisation=1e-6,
        )

    @jax.jit
    def device_conditional(current, trigger):
        return jax.lax.cond(
            trigger,
            lambda unused: update(current),
            lambda unused: current,
            operand=None,
        )

    update_jit = jax.jit(update)
    read_trigger = jax.jit(lambda trigger: trigger)
    trigger = jnp.asarray(True)

    device_lowered = device_conditional.lower(data, trigger)
    device_compiled, device_compile_s = _compile(device_lowered)
    update_lowered = update_jit.lower(data)
    update_compiled, update_compile_s = _compile(update_lowered)
    trigger_lowered = read_trigger.lower(trigger)
    trigger_compiled, trigger_compile_s = _compile(trigger_lowered)

    trigger_temporary_bytes = _temporary_bytes(trigger_compiled)
    update_temporary_bytes = _temporary_bytes(update_compiled)
    python_temporary_bytes = None
    if (
        trigger_temporary_bytes is not None
        and update_temporary_bytes is not None
    ):
        python_temporary_bytes = max(
            trigger_temporary_bytes,
            update_temporary_bytes,
        )
    output = {
        "dimension": args.dimension,
        "components": args.components,
        "population_size": args.population_size,
        "iterations": args.iterations,
        "iso_prob": args.iso_prob,
        "repetitions": args.repetitions,
        "device_compile_s": device_compile_s,
        "python_compile_s": update_compile_s + trigger_compile_s,
        "device_hlo_bytes": len(device_lowered.as_text().encode()),
        "python_hlo_bytes": (
            len(update_lowered.as_text().encode())
            + len(trigger_lowered.as_text().encode())
        ),
        "device_temporary_bytes": _temporary_bytes(device_compiled),
        "python_temporary_bytes": python_temporary_bytes,
    }
    for trigger_value in (False, True):
        device_times = []
        python_times = []
        device_trigger = jnp.asarray(trigger_value)
        for _ in range(args.repetitions):
            start = time.perf_counter()
            result = device_compiled(data, device_trigger)
            jax.block_until_ready(result)
            device_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            should_update = bool(trigger_compiled(device_trigger))
            result = update_compiled(data) if should_update else data
            jax.block_until_ready(result)
            python_times.append(time.perf_counter() - start)
        name = "update" if trigger_value else "skip"
        output[f"device_{name}_median_s"] = statistics.median(device_times)
        output[f"python_{name}_median_s"] = statistics.median(python_times)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
