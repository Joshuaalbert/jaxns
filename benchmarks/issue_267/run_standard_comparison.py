"""Compare local and distributed execution on a maintained standard problem."""

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

from benchmarks.issue_246.run_standard import _mode_mass
from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.constrained_sampler import (
    EllipsoidalDirection,
    UniDimSliceSampler,
)
from jaxns.core import NestedSampler
from jaxns.distributed_core import DistributedNestedSampler


def write_config(
        path: Path,
        workers: int,
        batch_size: int,
) -> int:
    """Write one reproducible local topology for the distributed candidate."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    worker_tables = "\n\n".join(
        f"""[[workers]]
platform = "cpu"
device = {device}
batch_size = {batch_size}"""
        for device in range(workers)
    )
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={workers}"
    )
    path.write_text(
        f"""
[runtime]
stack_id = "issue-267-standard"
runtime_dir = "runtime"
log_dir = "logs"
startup_timeout_s = 120
shutdown_timeout_s = 30
task_timeout_s = 600

[network]
port = {port}

{worker_tables}
""".strip() + "\n",
        encoding="utf-8",
    )
    return port


def cli(config: Path, command: str) -> None:
    """Apply the same documented lifecycle used by an end user."""
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
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())


def measure(runner, case_name: str, truth, seed: int) -> dict[str, object]:
    """Measure a complete run, including planner-side GMM refinement."""
    key = jax.random.PRNGKey(seed)
    started = time.perf_counter()
    completed = runner.run(key)
    state = completed.state if hasattr(completed, "state") else completed
    jax.block_until_ready(state)
    elapsed_s = time.perf_counter() - started

    started = time.perf_counter()
    results = state.to_result().trim()
    jax.block_until_ready(results)
    result_s = time.perf_counter() - started

    conditioning = (
        "phantom" if int(results.total_phantom_samples) > 0 else "classic"
    )
    started = time.perf_counter()
    evidence = results.sample_evidence_mc(
        num_samples=1000,
        conditioning=conditioning,
        key=jax.random.fold_in(key, 1),
    )
    jax.block_until_ready(evidence)
    mc_s = time.perf_counter() - started
    mode_mass, mode_mass_truth = _mode_mass(case_name, results)
    sampler_data = state.sampler_data
    return {
        "seed": seed,
        "elapsed_s": elapsed_s,
        "result_s": result_s,
        "mc_s": mc_s,
        "log_Z": float(results.log_Z_mean),
        "log_Z_error": float(results.log_Z_mean - truth),
        "log_Z_uncert": float(results.log_Z_uncert),
        "mc_conditioning": conditioning,
        "mc_log_Z": float(evidence.log_Z_mean),
        "mc_log_Z_error": float(evidence.log_Z_mean - truth),
        "mc_log_Z_uncert": float(evidence.log_Z_uncert),
        "samples": int(results.total_num_samples),
        "likelihood_evaluations": int(
            results.total_num_likelihood_evaluations
        ),
        "retained_phantoms": int(results.total_phantom_samples),
        "ess": float(results.ess),
        "mode_mass": mode_mass,
        "mode_mass_truth": mode_mass_truth,
        "mode_mass_error": (
            None if mode_mass is None else mode_mass - mode_mass_truth
        ),
        "gmm_updates": (
            0 if sampler_data is None else int(sampler_data.num_updates)
        ),
        "directions": (
            0 if sampler_data is None else int(sampler_data.num_directions)
        ),
        "isotropic_directions": (
            0 if sampler_data is None else int(sampler_data.num_isotropic)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="spike_slab")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--phantoms", action="store_true")
    parser.add_argument(
        "--runner",
        choices=("local", "distributed", "both"),
        default="both",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a checkpointed matrix without repeating completed seeds.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, truth = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
    dimension = int(model.U_ndims())
    root_degree = 30 * dimension
    replacement_width = min(root_degree, 10 * dimension)
    num_slices = 5 * dimension
    retained_phantoms = dimension if args.phantoms else 0
    direction = EllipsoidalDirection(prob_isotropic=1e-2)
    min_effective_samples = direction.min_effective_samples
    if min_effective_samples is None:
        min_effective_samples = (
            4 * direction.num_components * (dimension + 1)
        )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        collect_phantom_samples=args.phantoms,
        phantom_burn_in=num_slices - 1 - retained_phantoms,
        direction=direction,
    )
    common = {
        "model": model,
        "root_allocation_degree": root_degree,
        "delta_K": replacement_width,
        "max_samples": 100 * root_degree,
        "collect_phantom_samples": args.phantoms,
        "sampler": sampler,
    }
    records = []
    output = {
        "environment": {
            "device": str(jax.devices()[0]),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "x64": bool(jax.config.x64_enabled),
        },
        "case": args.case,
        "truth_log_Z": float(truth),
        "dimension": dimension,
        "root_degree": root_degree,
        "replacement_width": replacement_width,
        "num_slices": num_slices,
        "collect_phantoms": args.phantoms,
        "ellipsoidal_direction": {
            "num_components": direction.num_components,
            "min_effective_samples": direction.min_effective_samples,
            "resolved_min_effective_samples": min_effective_samples,
            "num_iterations": direction.num_iterations,
            "population_size": direction.population_size,
            "prob_isotropic": direction.prob_isotropic,
        },
        "workers": args.workers,
        "worker_batch_size": args.batch_size,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and args.output.exists():
        checkpointed = json.loads(args.output.read_text(encoding="utf-8"))
        # Refuse to combine records produced by a scientifically different
        # configuration. Runtime metadata can change across an interrupted
        # run, but these fields determine the nested-sampling experiment.
        for field in (
                "case",
                "truth_log_Z",
                "dimension",
                "root_degree",
                "replacement_width",
                "num_slices",
                "collect_phantoms",
                "ellipsoidal_direction",
                "workers",
                "worker_batch_size",
        ):
            if checkpointed.get(field) != output[field]:
                raise ValueError(
                    f"Cannot resume: checkpoint field {field!r} differs."
                )
        records.extend(checkpointed.get("records", []))

    def checkpoint() -> None:
        # A representative distributed matrix is intentionally long-running.
        # Keep every completed seed reviewable if a worker or host is stopped.
        args.output.write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )

    checkpoint()

    if args.runner in ("local", "both"):
        local = NestedSampler(
            shell_size=replacement_width,
            **common,
        )
        warm = measure(local, args.case, truth, -1)
        completed_seeds = {
            record["seed"]
            for record in records
            if record["runner"] == "local"
        }
        for seed in range(args.seeds):
            if seed in completed_seeds:
                continue
            record = measure(local, args.case, truth, seed)
            record.update({
                "runner": "local",
                "warm_s": warm["elapsed_s"],
            })
            records.append(record)
            checkpoint()

    if args.runner == "both":
        # Wide local-vmap executables and multiple worker executables are not
        # resident together in either real deployment. Release the local JIT
        # cache before assessing the worker topology so the comparison does
        # not inherit an executable that its deployment would never retain.
        jax.clear_caches()

    if args.runner in ("distributed", "both"):
        with tempfile.TemporaryDirectory(
                prefix="jaxns-issue-267-standard-",
        ) as directory:
            config = Path(directory) / "workers.toml"
            coordinator_port = write_config(
                config,
                args.workers,
                args.batch_size,
            )
            cli(config, "up")
            try:
                distributed = DistributedNestedSampler(
                    coordinator_port=coordinator_port,
                    initial_capacity=root_degree + 10 * replacement_width,
                    **common,
                )
                warm = measure(distributed, args.case, truth, -1)
                completed_seeds = {
                    record["seed"]
                    for record in records
                    if record["runner"] == "distributed"
                }
                for seed in range(args.seeds):
                    if seed in completed_seeds:
                        continue
                    record = measure(
                        distributed,
                        args.case,
                        truth,
                        seed,
                    )
                    record.update({
                        "runner": "distributed",
                        "warm_s": warm["elapsed_s"],
                    })
                    records.append(record)
                    checkpoint()
            finally:
                cli(config, "down")
    checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
