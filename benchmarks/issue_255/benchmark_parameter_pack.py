"""Focused read-only benchmark for jaxns issue #255.

The script compares the current explicit U-space pytree, the public
jaxctx.priors.ParameterPack optimisation API, and a flat-U transport boundary
implemented with jaxns.pytree.pytree_ravel.  It writes JSON Lines so every
timing sample is retained alongside robust summaries.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import pickle
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import tensorflow_probability.substrates.jax as tfp
from jaxctx import CtxParams, get_parameter, transform
from jaxctx.context import set_state
from jaxctx.priors import ParameterPack, Prior
from jaxctx.priors.prior import quick_unit_inverse

from jaxns.pytree import pytree_ravel

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions
JAXNS_SOURCE = Path(inspect.getsourcefile(pytree_ravel)).resolve()
if not JAXNS_SOURCE.is_relative_to(REPO_ROOT):
    raise RuntimeError(
        f"Expected JAXNS from {REPO_ROOT}, imported {JAXNS_SOURCE}."
    )


@dataclass(frozen=True)
class Case:
    name: str
    total_dims: int
    leaf_shapes: tuple[tuple[int, ...], ...]
    batch: int

    @property
    def leaves(self) -> int:
        return len(self.leaf_shapes)


def regular_case(name: str, total_dims: int, leaves: int, batch: int) -> Case:
    if total_dims % leaves:
        raise ValueError((total_dims, leaves))
    leaf_dims = total_dims // leaves
    leaf_shape = () if leaf_dims == 1 else (leaf_dims,)
    return Case(name, total_dims, (leaf_shape,) * leaves, batch)


CASES = (
    regular_case("d64_l1_b1", 64, 1, 1),
    regular_case("d64_l64_b1", 64, 64, 1),
    regular_case("d1024_l1_b1", 1024, 1, 1),
    regular_case("d1024_l64_b1", 1024, 64, 1),
    regular_case("d1024_l64_b64", 1024, 64, 64),
    regular_case("d1024_l256_b16", 1024, 256, 16),
    Case(
        "d96_l8_b16_hetero",
        96,
        ((), (2,), (3, 1), (5,), (2, 4), (13,), (3, 7), (43,)),
        16,
    ),
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def timing_summary_ns(values: list[int]) -> dict[str, float | list[int]]:
    return {
        "median_ms": statistics.median(values) / 1e6,
        "q25_ms": percentile(values, 25) / 1e6,
        "q75_ms": percentile(values, 75) / 1e6,
        "samples_ns": values,
    }


def block(value):
    return jax.block_until_ready(value)


def timed_call(call: Callable[[], object]) -> tuple[int, object]:
    started = time.perf_counter_ns()
    value = call()
    elapsed = time.perf_counter_ns() - started
    return elapsed, value


def make_priors(case: Case) -> tuple[Prior, ...]:
    width = max(4, len(str(case.leaves)))
    priors = []
    for index, shape in enumerate(case.leaf_shapes):
        size = int(np.prod(shape)) if shape else 1
        loc = jnp.linspace(-0.25, 0.25, size, dtype=jnp.float64).reshape(shape)
        scale = jnp.linspace(
            0.75, 1.25, size, dtype=jnp.float64
        ).reshape(shape)
        priors.append(
            Prior(
                tfpd.Normal(loc=loc, scale=scale),
                name=f"x{index:0{width}d}",
            )
        )
    return tuple(priors)


def likelihood(physical_values) -> jax.Array:
    terms = []
    for value in physical_values:
        terms.append(
            jnp.sum(
                -0.5 * jnp.square(value / 1.3)
                + 0.01 * jnp.sin(value)
            )
        )
    return sum(terms[1:], start=terms[0])


def make_inputs(case: Case, priors: tuple[Prior, ...]):
    key = jax.random.PRNGKey(
        20260826 + case.total_dims + case.leaves + case.batch
    )
    if case.batch == 1:
        flat_u = jax.random.uniform(
            key,
            (case.total_dims,),
            minval=0.05,
            maxval=0.95,
            dtype=jnp.float64,
        )
    else:
        flat_u = jax.random.uniform(
            key,
            (case.batch, case.total_dims),
            minval=0.05,
            maxval=0.95,
            dtype=jnp.float64,
        )

    offset = 0
    tree = {}
    for prior in priors:
        stop = offset + prior.base_ndims
        if case.batch == 1:
            tree[prior.name] = flat_u[offset:stop].reshape(prior.base_shape)
        else:
            tree[prior.name] = flat_u[:, offset:stop].reshape(
                (case.batch,) + prior.base_shape
            )
        offset = stop
    current_u = CtxParams(tree)
    packed_n = quick_unit_inverse(flat_u)
    return current_u, flat_u, packed_n


def make_functions(
    case: Case,
    priors: tuple[Prior, ...],
    example_u: CtxParams,
):
    def current_model():
        return likelihood(tuple(prior.realise() for prior in priors))

    current_transformed = transform(current_model, base_dtype=jnp.float64)

    pack = ParameterPack(priors, name="packed")

    def packed_model():
        return likelihood(pack.parameter())

    packed_transformed = transform(packed_model, base_dtype=jnp.float64)

    # ParameterPack has no Bayesian realise() path.  This is the matched
    # hypothetical implementation such a path would need: one explicit,
    # uniformly sampled U leaf, followed by the same static slices, reshapes,
    # prior transforms, named X states, and log-probability states.
    def hypothetical_packed_u_model():
        packed_u = get_parameter(
            "packed",
            "U",
            shape=(case.total_dims,),
            dtype=jnp.float64,
        )
        values = []
        offset = 0
        for prior in priors:
            stop = offset + prior.base_ndims
            unit_value = jax.lax.slice_in_dim(packed_u, offset, stop)
            unit_value = jax.lax.reshape(unit_value, prior.base_shape)
            physical_value = prior.forward(unit_value)
            set_state(prior.name, "X", physical_value)
            set_state(prior.name, "log_prob", prior.log_prob(physical_value))
            values.append(physical_value)
            offset = stop
        return likelihood(tuple(values))

    hypothetical_packed_u_transformed = transform(
        hypothetical_packed_u_model,
        base_dtype=jnp.float64,
    )

    def current_base(unit_tree):
        applied = current_transformed.apply(
            None,
            {
                "params": CtxParams(),
                "U": unit_tree,
                "X": CtxParams(),
                "log_prob": CtxParams(),
            },
        )
        return applied.fn_val

    if case.batch == 1:
        example_single_u = example_u
    else:
        example_single_u = jax.tree.map(lambda value: value[0], example_u)
    _, unravel = pytree_ravel(example_single_u)

    def transport_base(flat_unit):
        return current_base(unravel(flat_unit))

    def packed_base(unconstrained):
        applied = packed_transformed.apply(
            None,
            {
                "params": CtxParams({"packed": unconstrained}),
                "U": CtxParams(),
                "X": CtxParams(),
                "log_prob": CtxParams(),
            },
        )
        return applied.fn_val

    def hypothetical_packed_u_base(flat_unit):
        applied = hypothetical_packed_u_transformed.apply(
            None,
            {
                "params": CtxParams(),
                "U": CtxParams({"packed": flat_unit}),
                "X": CtxParams(),
                "log_prob": CtxParams(),
            },
        )
        return applied.fn_val

    if case.batch == 1:
        return (
            current_base,
            transport_base,
            hypothetical_packed_u_base,
            packed_base,
        )
    return (
        jax.vmap(current_base),
        jax.vmap(transport_base),
        jax.vmap(hypothetical_packed_u_base),
        jax.vmap(packed_base),
    )


def compiler_metadata(lowered, compiled, jaxpr) -> dict[str, object]:
    stablehlo = lowered.as_text()
    memory = compiled.memory_analysis()
    memory_fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    memory_dict = {
        field: int(getattr(memory, field))
        for field in memory_fields
        if hasattr(memory, field)
    }
    executable = compiled.runtime_executable()
    try:
        executable_size = int(executable.size_of_generated_code_in_bytes())
    except (AttributeError, RuntimeError) as error:
        executable_size = f"unavailable: {type(error).__name__}: {error}"
    text_lower = stablehlo.lower()
    return {
        "jaxpr_eqns": len(jaxpr.jaxpr.eqns),
        "jaxpr_chars": len(str(jaxpr)),
        "stablehlo_chars": len(stablehlo),
        "stablehlo_counts": {
            name: text_lower.count(name)
            for name in (
                "concatenate",
                "dynamic_slice",
                "slice",
                "reshape",
                "copy",
                "get_tuple_element",
            )
        },
        "memory": memory_dict,
        "executable_code_bytes": executable_size,
    }


def benchmark_variant(name: str, fn: Callable, value) -> dict[str, object]:
    trace_samples = []
    jaxpr = None
    for _ in range(5):
        jax.clear_caches()
        elapsed, jaxpr = timed_call(lambda: jax.make_jaxpr(fn)(value))
        trace_samples.append(elapsed)

    lower_samples = []
    compile_samples = []
    lowered = None
    compiled = None
    for _ in range(3):
        jax.clear_caches()
        jitted = jax.jit(fn)
        elapsed, lowered = timed_call(
            lambda jitted=jitted: jitted.lower(value)
        )
        lower_samples.append(elapsed)
        elapsed, compiled = timed_call(lowered.compile)
        compile_samples.append(elapsed)

    block(compiled(value))
    execution_samples = []
    for _ in range(51):
        started = time.perf_counter_ns()
        block(compiled(value))
        execution_samples.append(time.perf_counter_ns() - started)

    return {
        "variant": name,
        "trace": timing_summary_ns(trace_samples),
        "lower": timing_summary_ns(lower_samples),
        "compile": timing_summary_ns(compile_samples),
        "steady_execution": timing_summary_ns(execution_samples),
        "compiler": compiler_metadata(lowered, compiled, jaxpr),
    }


def timed_pickle(value, repetitions: int = 101):
    dump_samples = []
    load_samples = []
    payload = None
    for _ in range(repetitions):
        elapsed, payload = timed_call(lambda: pickle.dumps(value, protocol=5))
        dump_samples.append(elapsed)
        elapsed, restored = timed_call(
            lambda payload=payload: pickle.loads(payload)
        )
        load_samples.append(elapsed)
        if restored is None:
            raise AssertionError("pickle round trip failed")
    return {
        "payload_bytes": len(payload),
        "dump": timing_summary_ns(dump_samples),
        "load": timing_summary_ns(load_samples),
    }


def benchmark_serialization(case: Case, current_u, flat_u, packed_n):
    host_current = jax.tree.map(np.asarray, current_u)
    host_flat = np.asarray(flat_u)
    host_n = np.asarray(packed_n)

    current_leaves = list(jax.tree.leaves(host_current))
    if case.batch == 1:

        def host_ravel():
            return np.concatenate(
                [leaf.reshape(-1) for leaf in current_leaves]
            )

        def host_unravel(value):
            output = {}
            offset = 0
            width = max(4, len(str(case.leaves)))
            for index, shape in enumerate(case.leaf_shapes):
                size = int(np.prod(shape)) if shape else 1
                chunk = value[offset:offset + size]
                output[f"x{index:0{width}d}"] = chunk.reshape(shape)
                offset += size
            return CtxParams(output)
    else:

        def host_ravel():
            return np.concatenate(
                [leaf.reshape(case.batch, -1) for leaf in current_leaves],
                axis=1,
            )

        def host_unravel(value):
            output = {}
            offset = 0
            width = max(4, len(str(case.leaves)))
            for index, shape in enumerate(case.leaf_shapes):
                size = int(np.prod(shape)) if shape else 1
                chunk = value[:, offset:offset + size]
                output[f"x{index:0{width}d}"] = chunk.reshape(
                    (case.batch,) + shape
                )
                offset += size
            return CtxParams(output)

    ravel_samples = []
    unravel_samples = []
    for _ in range(101):
        elapsed, ravelled = timed_call(host_ravel)
        ravel_samples.append(elapsed)
        elapsed, restored = timed_call(
            lambda ravelled=ravelled: host_unravel(ravelled)
        )
        unravel_samples.append(elapsed)
        if len(jax.tree.leaves(restored)) != case.leaves:
            raise AssertionError("host unravel leaf count changed")

    return {
        "current_pytree": timed_pickle(host_current),
        "transport_flat_u": timed_pickle(host_flat),
        "parameter_pack_n": timed_pickle(host_n),
        "transport_host_ravel": timing_summary_ns(ravel_samples),
        "transport_host_unravel": timing_summary_ns(unravel_samples),
    }


def benchmark_case(case: Case) -> dict[str, object]:
    priors = make_priors(case)
    current_u, flat_u, packed_n = make_inputs(case, priors)
    (
        current_fn,
        transport_fn,
        hypothetical_packed_u_fn,
        packed_fn,
    ) = make_functions(case, priors, current_u)

    current_result = block(current_fn(current_u))
    transport_result = block(transport_fn(flat_u))
    hypothetical_packed_u_result = block(hypothetical_packed_u_fn(flat_u))
    packed_result = block(packed_fn(packed_n))
    current_np = np.asarray(current_result)
    transport_np = np.asarray(transport_result)
    hypothetical_packed_u_np = np.asarray(hypothetical_packed_u_result)
    packed_np = np.asarray(packed_result)
    correctness = {
        "current_transport_max_abs": float(
            np.max(np.abs(current_np - transport_np))
        ),
        "current_pack_max_abs": float(np.max(np.abs(current_np - packed_np))),
        "current_transport_allclose_rtol1e-12_atol1e-12": bool(
            np.allclose(current_np, transport_np, rtol=1e-12, atol=1e-12)
        ),
        "current_pack_allclose_rtol1e-12_atol1e-12": bool(
            np.allclose(current_np, packed_np, rtol=1e-12, atol=1e-12)
        ),
        "current_hypothetical_packed_u_max_abs": float(
            np.max(np.abs(current_np - hypothetical_packed_u_np))
        ),
        "current_hypothetical_packed_u_allclose_rtol1e-12_atol1e-12": bool(
            np.allclose(
                current_np,
                hypothetical_packed_u_np,
                rtol=1e-12,
                atol=1e-12,
            )
        ),
    }

    variants = []
    for name, fn, value in (
        ("current_u_pytree", current_fn, current_u),
        ("transport_flat_u", transport_fn, flat_u),
        ("hypothetical_packed_u", hypothetical_packed_u_fn, flat_u),
        ("parameter_pack_public_n", packed_fn, packed_n),
    ):
        variants.append(benchmark_variant(name, fn, value))

    return {
        "record": "case",
        "case": asdict(case),
        "correctness": correctness,
        "variants": variants,
        "serialization": benchmark_serialization(
            case, current_u, flat_u, packed_n
        ),
    }


def environment_record() -> dict[str, object]:
    return {
        "record": "environment",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "jaxctx": importlib.metadata.version("jaxctx"),
        "jaxns": importlib.metadata.version("jaxns"),
        "jaxns_source": str(JAXNS_SOURCE),
        "repo_root": str(REPO_ROOT),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
        "timestamp_unix": time.time(),
        "notes": {
            "trace": "jax.make_jaxpr, five cache-cleared repetitions",
            "lower_compile": (
                "three cache-cleared repetitions; lower and compile timed "
                "separately"
            ),
            "steady": (
                "51 compiled calls, each synchronized with "
                "jax.block_until_ready"
            ),
            "serialization": "101 pickle protocol 5 host-array round trips",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", action="append", choices=[case.name for case in CASES]
    )
    args = parser.parse_args()
    selected = (
        CASES
        if args.case is None
        else tuple(case for case in CASES if case.name in set(args.case))
    )
    print(json.dumps(environment_record(), sort_keys=True), flush=True)
    for case in selected:
        print(json.dumps(benchmark_case(case), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
