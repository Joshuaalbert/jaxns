"""Measure real complete-chain throughput through the scalar task protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.distributed_core import WorkerSession
from jaxns.model import Model
from jaxns.runtime_client import SupervisorClient
from jaxns.samples import SeedPoint

tfpd = tfp.distributions


def prior_model():
    u = Prior(tfpd.Uniform(0.0, 1.0), name="u").realise()
    return -jnp.square(u - 0.25)


def write_config(path: Path, workers: int) -> None:
    path.write_text(
        f"""
[runtime]
stack_id = "issue-267-throughput-{workers}"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 60
shutdown_timeout_s = 20
task_timeout_s = 120

[[workers]]
name = "cpu"
platform = "cpu"
device = 0
batch_size = 1
count = {workers}
""".strip() + "\n",
        encoding="utf-8",
    )


def cli(config: Path, command: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jaxns.cli", "--config", str(config), command],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())


def request(key, seed_u, seed_log_likelihood) -> ConstrainedSampleRequest:
    return ConstrainedSampleRequest(
        keys=key[None],
        valid=jnp.asarray([True]),
        log_L_constraints=jnp.asarray([-1.0]),
        seed_points=SeedPoint(
            U0=jax.tree.map(lambda value: value[None], seed_u),
            log_L0=jnp.asarray([seed_log_likelihood]),
        ),
        sampler_data=None,
    )


def batched_request(keys, seed_u, seed_log_likelihood):
    """Build the exact worker sampling request at a local vmap width."""
    width = keys.shape[0]
    return ConstrainedSampleRequest(
        keys=keys,
        valid=jnp.ones((width,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((width,), -1.0),
        seed_points=SeedPoint(
            U0=jax.tree.map(
                lambda value: jnp.repeat(value[None], width, axis=0),
                seed_u,
            ),
            log_L0=jnp.repeat(seed_log_likelihood[None], width, axis=0),
        ),
        sampler_data=None,
    )


def measure_local_width(
        sampler,
        width: int,
        tasks: int,
        repeats: int,
        seed_u,
        seed_log_likelihood,
) -> dict[str, object]:
    """Measure the local core's compiled, vmapped sampler execution layer."""
    if tasks % width != 0:
        raise ValueError(f"Task count {tasks} is not divisible by width {width}.")
    batches = tasks // width

    # Keep the batch loop on device, as it is in the jitted local depth loop.
    # Returning only work counters prevents materialising every sample while
    # retaining all rejection-loop computation needed to produce each result.
    def sample_all(keys):
        def body(total, batch):
            result = sample_request(
                sampler,
                batched_request(batch, seed_u, seed_log_likelihood),
            )
            return total + jnp.sum(result.num_likelihood_evaluations), None

        total, _ = jax.lax.scan(
            body,
            jnp.asarray(0, dtype=jnp.int64),
            keys,
        )
        return total

    program = jax.jit(sample_all)
    warm_keys = jax.random.split(
        jax.random.PRNGKey(1_000_000 + width),
        tasks,
    ).reshape((batches, width, 2))
    compile_started = time.perf_counter()
    jax.block_until_ready(program(warm_keys))
    compile_and_warm_s = time.perf_counter() - compile_started

    elapsed = []
    likelihood_evaluations = []
    for repeat in range(repeats):
        # Every width receives the same per-repeat scalar random-key stream.
        # Reshaping changes only execution grouping, not scientific work.
        keys = jax.random.split(
            jax.random.PRNGKey(2_000_000 + repeat),
            tasks,
        ).reshape((batches, width, 2))
        started = time.perf_counter()
        total = jax.block_until_ready(program(keys))
        elapsed.append(time.perf_counter() - started)
        likelihood_evaluations.append(int(total))
    return {
        "width": width,
        "elapsed_s": elapsed,
        "samples_per_s": [tasks / value for value in elapsed],
        "likelihood_evaluations": likelihood_evaluations,
        "compile_and_warm_s": compile_and_warm_s,
    }


def run_round(
        client,
        session_id: str,
        first: int,
        tasks: int,
        seed_u,
        seed_log_likelihood,
) -> float:
    started = time.perf_counter()
    for task_id in range(first, first + tasks):
        client.submit(
            session_id,
            task_id,
            request(
                jax.random.PRNGKey(task_id),
                seed_u,
                seed_log_likelihood,
            ),
        )
    for _ in range(tasks):
        task_id, _ = client.receive(session_id, timeout_s=120.0)
        client.acknowledge(session_id, task_id)
    return time.perf_counter() - started


def measure(workers: int, tasks: int, repeats: int) -> dict[str, object]:
    model = Model(prior_model=prior_model)
    sampler = UniDimSliceSampler(model=model, num_slices=10)
    session = WorkerSession(sampler=sampler, args=(), params=None)
    seed_u = model.sample_U(jax.random.PRNGKey(0))
    seed_log_likelihood = model.log_likelihood(seed_u)
    with tempfile.TemporaryDirectory(prefix="jaxns-throughput-") as directory:
        config = Path(directory) / "workers.toml"
        write_config(config, workers)
        cli(config, "up")
        try:
            with SupervisorClient.from_config(config) as client:
                session_id = f"throughput-{workers}"
                client.register(session_id, session)
                first = 1
                run_round(
                    client,
                    session_id,
                    first,
                    max(12, 6 * workers),
                    seed_u,
                    seed_log_likelihood,
                )
                first += max(12, 6 * workers)
                elapsed = []
                for _ in range(repeats):
                    elapsed.append(run_round(
                        client,
                        session_id,
                        first,
                        tasks,
                        seed_u,
                        seed_log_likelihood,
                    ))
                    first += tasks
                status = client.status()
                client.release(session_id)
                worker_status = status["workers"]
                return {
                    "elapsed_s": elapsed,
                    "samples_per_s": [tasks / value for value in elapsed],
                    "peak_rss_kib": [
                        worker["peak_rss_kib"]
                        for worker in worker_status
                    ],
                    "compile_s": [
                        worker["compile_s"]
                        for worker in worker_status
                    ],
                }
        finally:
            cli(config, "down")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = Model(prior_model=prior_model)
    sampler = UniDimSliceSampler(model=model, num_slices=10)
    seed_u = model.sample_U(jax.random.PRNGKey(0))
    seed_log_likelihood = model.log_likelihood(seed_u)
    widths = tuple(
        width
        for width in (1, 2, 3, 4, 6, 10, 12)
        if args.tasks % width == 0
    )
    output = {
        "environment": {
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "x64": bool(jax.config.x64_enabled),
        },
        "tasks_per_repeat": args.tasks,
        "distributed_records": [
            {
                "workers": workers,
                **measure(workers, args.tasks, args.repeats),
            }
            for workers in (1, 2, 3)
        ],
        "local_vmap_records": [
            measure_local_width(
                sampler,
                width,
                args.tasks,
                args.repeats,
                seed_u,
                seed_log_likelihood,
            )
            for width in widths
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
