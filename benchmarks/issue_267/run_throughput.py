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
from tensorflow_probability.substrates import jax as tfp

import jaxns
from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    UniDimSliceSampler,
)
from jaxns.distributed_core import WorkerSession
from jaxns.model import Model
from jaxns.runtime_client import SupervisorClient
from jaxns.samples import SeedPoint

tfpd = tfp.distributions


def prior_model():
    u = jaxns.Prior(tfpd.Uniform(0.0, 1.0), name="u").realise()
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
    output = {
        "environment": {
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "x64": bool(jax.config.x64_enabled),
        },
        "tasks_per_repeat": args.tasks,
        "records": [
            {
                "workers": workers,
                **measure(workers, args.tasks, args.repeats),
            }
            for workers in (1, 2, 3)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
