"""Compare local and scalar-worker distributed scheduling on an analytic model."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxctx.priors.prior import Prior
from scipy.special import erf
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.distributed_core import DistributedNestedSampler
from jaxns.model import Model
from jaxns.runtime.config import load_runtime_config
from jaxns.termination_condition import TerminationCondition

tfpd = tfp.distributions


def prior_model():
    u = Prior(tfpd.Uniform(0.0, 1.0), name="u").realise()
    return -jnp.square(u - 0.25)


def truth() -> float:
    evidence = 0.5 * np.sqrt(np.pi) * (erf(0.75) + erf(0.25))
    return float(np.log(evidence))


def sampler(model: Model, phantoms: bool) -> UniDimSliceSampler:
    return UniDimSliceSampler(
        model=model,
        num_slices=3,
        collect_phantom_samples=phantoms,
        max_phantom_samples=2 if phantoms else None,
    )


def local_runner(model: Model, phantoms: bool) -> NestedSampler:
    return NestedSampler(
        model=model,
        root_allocation_degree=6,
        shell_size=6,
        delta_K=6,
        max_samples=512,
        initial_capacity=128,
        sampler=sampler(model, phantoms),
        termination_condition=TerminationCondition(dlogZ=jnp.asarray(0.1)),
    )


def distributed_runner(
        model: Model,
        phantoms: bool,
        coordinator_port: int,
) -> DistributedNestedSampler:
    return DistributedNestedSampler(
        model=model,
        coordinator_port=coordinator_port,
        root_allocation_degree=6,
        delta_K=6,
        max_samples=512,
        initial_capacity=128,
        sampler=sampler(model, phantoms),
        termination_condition=TerminationCondition(dlogZ=jnp.asarray(0.1)),
    )


def measure(runner, seed: int) -> dict[str, float | int]:
    started = time.perf_counter()
    completed = runner.run(jax.random.PRNGKey(seed))
    state = completed.state if hasattr(completed, "state") else completed
    elapsed_s = time.perf_counter() - started
    result = state.to_result()
    valid = int(state.num_samples)
    return {
        "seed": seed,
        "elapsed_s": elapsed_s,
        "log_Z": float(result.log_Z_mean),
        "samples": valid,
        "likelihood_evaluations": int(jnp.sum(
            state.samples.num_likelihood_evaluations[:valid]
        )),
        "retained_phantoms": int(jnp.sum(
            state.samples.phantom_samples.valid_mask[:valid]
        )),
    }


def write_local_config(path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workers = "\n\n".join(
        f"""[[workers]]
platform = "cpu"
device = {device}
batch_size = 1"""
        for device in range(3)
    )
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=3"
    path.write_text(
        f"""
[runtime]
stack_id = "issue-267-benchmark"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 60
shutdown_timeout_s = 20
task_timeout_s = 120

[network]
port = {port}

{workers}
""".strip() + "\n",
        encoding="utf-8",
    )


def cli(config: Path, command: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxns.cli",
            "--config",
            str(config),
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"jaxns-cli {command} failed: {completed.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = Model(prior_model=prior_model)
    records = []
    with tempfile.TemporaryDirectory(prefix="jaxns-issue-267-") as directory:
        config = Path(directory) / "workers.toml"
        write_local_config(config)
        cli(config, "up")
        try:
            for phantoms in (False, True):
                runners = {
                    "local": local_runner(model, phantoms),
                    "distributed": distributed_runner(
                        model,
                        phantoms,
                        load_runtime_config(config).network.port,
                    ),
                }
                for name, runner in runners.items():
                    warm = measure(runner, 0)
                    for seed in range(1, args.seeds + 1):
                        record = measure(runner, seed)
                        record.update({
                            "runner": name,
                            "collect_phantoms": phantoms,
                            "warm_s": warm["elapsed_s"],
                        })
                        records.append(record)
        finally:
            cli(config, "down")
    output = {
        "environment": {
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "x64": bool(jax.config.x64_enabled),
        },
        "log_Z_true": truth(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
