"""Measure disabled-path runtime and opt-in full-state checkpoint cost."""
# flake8: noqa

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--measure-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-capacity", type=int, default=1_000_000)
    return parser.parse_args()


ARGS = _parse_args()
sys.path.insert(0, str(ARGS.source.resolve()))

import jax
import jax.numpy as jnp
import jaxlib

from jaxns.constrained_sampler import AbstractSampler
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.depth_condition import DepthCondition


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkModel(PureDataclassPytree):
    """Eight-dimensional model with a cheap likelihood."""

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 8

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, shape=(8,))

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(self, U, args=(), params=None, *, allow_nan=True):
        del args, params, allow_nan
        return -jnp.sum(jnp.square(U - 0.5))

    def log_prior(self, U, args=(), params=None):
        del args, params
        return jnp.where(jnp.all((U >= 0.0) & (U <= 1.0)), 0.0, -jnp.inf)


BenchmarkModel.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkSampler(PureDataclassPytree, AbstractSampler):
    """Cheap sampler isolating Python run/checkpoint orchestration."""

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
    ):
        del key, args, params
        finite = jnp.where(
            jnp.isfinite(log_L_constraint),
            log_L_constraint,
            0.0,
        )
        return (
            seed_point.U0,
            finite + 1.0,
            jnp.asarray(1, dtype=jnp.int32),
            PhantomSamples(
                U_samples=jnp.zeros((0, 8)),
                valid_mask=jnp.zeros((0,), dtype=bool),
                log_L=jnp.zeros((0,)),
            ),
        )


BenchmarkSampler.register_pytree()


model = BenchmarkModel()
runner = NestedSampler(
    model=model,
    root_allocation_degree=16,
    shell_size=8,
    max_samples=256,
    initial_capacity=256,
    sampler=BenchmarkSampler(),
    depth_condition=DepthCondition(),
)


def goal(state):
    return int(state.goal_loop_iter) >= 3


compile_start = time.perf_counter()
warm_state = runner.run_until_goal(goal, key=jax.random.PRNGKey(0))
jax.block_until_ready(warm_state)
compile_and_first_run_seconds = time.perf_counter() - compile_start

run_seconds = []
signatures = []
for repeat in range(ARGS.repeats):
    start = time.perf_counter()
    state = runner.run_until_goal(
        goal,
        key=jax.random.PRNGKey(repeat + 1),
    )
    jax.block_until_ready(state)
    run_seconds.append(time.perf_counter() - start)
    signatures.append((
        int(state.num_samples),
        int(state.root_out_degree),
        float(jnp.sum(state.samples.log_likelihoods[:state.num_samples])),
    ))

checkpoint = None
if ARGS.measure_checkpoint:
    from jaxns.checkpoint import CheckpointManager

    # Capacity, rather than current valid count, dominates full-state bytes.
    # This enlarged real State keeps the benchmark tied to the public Pytree
    # schema while representing a materially longer scientific run.
    checkpoint_state = warm_state.resize(ARGS.checkpoint_capacity)
    jax.block_until_ready(checkpoint_state)
    checkpoint_seconds = []
    with tempfile.TemporaryDirectory() as directory:
        with CheckpointManager(directory) as manager:
            for _ in range(5):
                start = time.perf_counter()
                manager.save(checkpoint_state)
                checkpoint_seconds.append(time.perf_counter() - start)
        manifest = json.loads(
            (Path(directory) / "CHECKPOINT").read_text(encoding="utf-8")
        )
        state_bytes = (Path(directory) / manifest["state_file"]).stat().st_size
    checkpoint = {
        "capacity": ARGS.checkpoint_capacity,
        "state_bytes": state_bytes,
        "seconds": checkpoint_seconds,
        "median_seconds": statistics.median(checkpoint_seconds),
        "hourly_wall_fraction": statistics.median(checkpoint_seconds) / 3600.0,
    }

result = {
    "source": str(ARGS.source.resolve()),
    "python": platform.python_version(),
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "device": str(jax.devices()[0]),
    "x64": bool(jax.config.x64_enabled),
    "compile_and_first_run_seconds": compile_and_first_run_seconds,
    "run_seconds": run_seconds,
    "median_run_seconds": statistics.median(run_seconds),
    "signatures": signatures,
    "checkpoint": checkpoint,
}
ARGS.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "output": str(ARGS.output),
    "compile_and_first_run_seconds": compile_and_first_run_seconds,
    "median_run_seconds": result["median_run_seconds"],
    "checkpoint": checkpoint,
}, indent=2))
