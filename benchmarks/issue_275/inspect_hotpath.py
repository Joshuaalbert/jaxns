"""Inspect the all-false constrained-sampler program across revisions."""

# Importing the selected source tree must happen before importing JAXNS.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--output", type=Path)
parser.add_argument("--periodic", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / "src"))

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.samples import SeedPoint


tfpd = tfp.distributions


def _prior_model():
    value = Prior(
        tfpd.Uniform(low=0.0, high=1.0),
        name="value",
    ).realise()
    return -jnp.square(value - 0.5)


def _periodic_prior_model():
    value = Prior(
        tfpd.Uniform(low=0.0, high=1.0),
        name="value",
    ).realise(periodic=True)
    return -jnp.square(value - 0.5)


def main() -> None:
    model = Model(
        prior_model=(
            _periodic_prior_model if args.periodic else _prior_model
        )
    )
    configured = NestedSampler(
        model=model,
        root_allocation_degree=32,
        sampler=UniDimSliceSampler(model=model, num_slices=32),
    ).sampler
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    # [S, ...] structured unit-cube seeds.
    seeds = jax.vmap(model.sample_U)(keys)
    log_likelihoods = jax.vmap(model.log_likelihood)(seeds)
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(2), 8),
        valid=jnp.ones((8,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((8,), -0.3),
        seed_points=SeedPoint(
            U0=seeds,
            log_L0=log_likelihoods,
        ),
        sampler_data=None,
    )
    compile_timings = []
    for _ in range(5):
        jax.clear_caches()
        lowered = jax.jit(
            lambda value: sample_request(configured, value)
        ).lower(request)
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_timings.append(time.perf_counter() - started)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    output = compiled(request)
    jax.block_until_ready(output.log_likelihoods)

    timings = []
    for _ in range(30):
        started = time.perf_counter()
        for _ in range(100):
            output = compiled(request)
        jax.block_until_ready(output.log_likelihoods)
        timings.append((time.perf_counter() - started) / 100.0)
    memory = compiled.memory_analysis()
    report = {
        "source_root": str(args.source_root),
        "periodic": args.periodic,
        "stablehlo_sha256": hashlib.sha256(
            stablehlo.encode("utf-8")
        ).hexdigest(),
        "stablehlo_lines": len(stablehlo.splitlines()),
        "remainder_operations": stablehlo.count("stablehlo.remainder"),
        "select_operations": stablehlo.count("stablehlo.select"),
        "compile_s": float(np.median(compile_timings)),
        "compile_samples_s": compile_timings,
        "median_execute_s": float(np.median(timings)),
        "temporary_bytes": memory.temp_size_in_bytes,
        "argument_bytes": memory.argument_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
