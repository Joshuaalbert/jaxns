"""Bounded full NestedSampler comparison for issue #255."""

import inspect
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jaxctx import get_base_dtype, get_parameter, wrap_random
from jaxctx.context import set_state
from jaxctx.priors import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.termination_condition import TerminationCondition

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions
DIMS = 4
SEEDS = 30
JAXNS_SOURCE = Path(inspect.getsourcefile(Model)).resolve()
if not JAXNS_SOURCE.is_relative_to(REPO_ROOT):
    raise RuntimeError(
        f"Expected JAXNS from {REPO_ROOT}, imported {JAXNS_SOURCE}."
    )


def make_priors():
    return tuple(
        Prior(tfpd.Normal(jnp.asarray(0.0), jnp.asarray(1.0)), name=f"x{i}")
        for i in range(DIMS)
    )


def score(values):
    return sum(
        (-0.5 * jnp.square(value / 1.3) for value in values),
        start=jnp.asarray(0.0),
    )


def make_current():
    priors = make_priors()

    def prior_model():
        return score(tuple(prior.realise() for prior in priors))

    return Model(prior_model)


def make_packed_u():
    priors = make_priors()

    def prior_model():
        packed = get_parameter(
            "packed",
            "U",
            shape=(DIMS,),
            dtype=get_base_dtype(),
            init=wrap_random(jax.random.uniform, "U"),
        )
        values = []
        for index, prior in enumerate(priors):
            unit_value = jax.lax.reshape(
                jax.lax.slice_in_dim(packed, index, index + 1),
                prior.base_shape,
            )
            value = prior.forward(unit_value)
            set_state(prior.name, "X", value)
            set_state(prior.name, "log_prob", prior.log_prob(value))
            values.append(value)
        return score(tuple(values))

    return Model(prior_model)


def make_sampler(model):
    return NestedSampler(
        model=model,
        target_num_live_points=4,
        shell_size=2,
        max_samples=12,
        initial_capacity=12,
        termination_condition=TerminationCondition(max_samples=12),
    )


def synchronise(state):
    jax.block_until_ready(state)
    return state


def run_once(sampler, seed):
    started = time.perf_counter_ns()
    state = synchronise(sampler.run(jax.random.PRNGKey(seed)))
    elapsed = time.perf_counter_ns() - started
    result = state.to_result().trim()
    jax.block_until_ready(result.log_Z_mean)
    return elapsed, float(result.log_Z_mean), int(state.num_samples)


def summary(values):
    arr = np.asarray(values)
    return {
        "median_ms": statistics.median(values) / 1e6,
        "q25_ms": float(np.percentile(arr, 25)) / 1e6,
        "q75_ms": float(np.percentile(arr, 75)) / 1e6,
        "samples_ns": values,
    }


output = {
    "environment": {
        "jax": jax.__version__,
        "jaxns_source": str(JAXNS_SOURCE),
        "dims": DIMS,
        "backend": jax.default_backend(),
        "x64_enabled": jax.config.x64_enabled,
        "notes": (
            "One compile-and-run warmup followed by 30 synchronized seeds; "
            "variant execution order alternates by seed."
        ),
    }
}
samplers = {
    "current_u_pytree": make_sampler(make_current()),
    "hypothetical_packed_u": make_sampler(make_packed_u()),
}
warmups = {name: run_once(sampler, 900) for name, sampler in samplers.items()}
records = {
    name: {"elapsed": [], "evidences": [], "counts": []}
    for name in samplers
}
names = tuple(samplers)
for seed in range(SEEDS):
    # Alternate execution order so thermal drift or shared-host contention
    # cannot systematically favour the representation measured second.
    order = names if seed % 2 == 0 else names[::-1]
    for name in order:
        runtime, evidence, count = run_once(samplers[name], seed)
        records[name]["elapsed"].append(runtime)
        records[name]["evidences"].append(evidence)
        records[name]["counts"].append(count)

for name in names:
    output[name] = {
        "warm_compile_and_run_ms": warmups[name][0] / 1e6,
        "steady_end_to_end": summary(records[name]["elapsed"]),
        "log_Z_mean_by_seed": records[name]["evidences"],
        "num_samples_by_seed": records[name]["counts"],
    }
print(json.dumps(output, sort_keys=True))
