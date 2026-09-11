"""Time one complete goal from an immutable saved tree, including index updates.

Each repeat starts from the same in-memory input and continuation keys. Nothing
is written to the archive. Conversion is timed separately; cold execution
includes compilation and warm execution synchronizes every returned array.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

import jaxns
from benchmarks.phantom_seeding.state_fingerprints import scientific_fingerprints
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--variant", choices=("reference", "combined"), required=True)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--ss10-accuracy", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source_root = Path(jaxns.__file__).resolve().parents[2]
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
).strip()
assert len(os.sched_getaffinity(0)) == 1 and jax.config.jax_enable_x64
if args.variant == "reference":
    assert commit == "aa79a0dc395c0d64b85cbb88d0e3e732decd7361"
    assert not subprocess.check_output(
        ["git", "diff", "HEAD", "--", "src"], cwd=source_root
    )
else:
    assert source_root == Path(__file__).resolve().parents[2]
harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
source_hashes = {
    str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((source_root / "src/jaxns").rglob("*.py"))
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(args.output)
with args.state.open("rb") as stream:
    state = pickle.load(stream)
jax.block_until_ready(state)
conversion_seconds = 0.0
if args.variant == "combined":
    from jaxns.sampling.phantom_index import build_phantom_seed_index

    started = time.perf_counter()
    state = dataclasses.replace(
        state,
        phantom_seed_index=build_phantom_seed_index(state.samples, state.num_samples),
        samples=dataclasses.replace(
            state.samples,
            phantom_samples=dataclasses.replace(
                state.samples.phantom_samples,
                seed_log_L_sorted=None,
            ),
        ),
    )
    jax.block_until_ready(state)
    conversion_seconds = time.perf_counter() - started
core = NestedSampler(
    model=state.model,
    root_allocation_degree=300,
    shell_size=100,
    delta_K=300,
    sampler=UniDimSliceSampler(
        model=state.model,
        num_slices=100,
        collect_phantom_samples=True,
        max_phantom_samples=99,
        no_step_out=True,
    ),
    collect_phantom_samples=True,
    max_phantom_samples=99,
    allocation_target="evidence_improving",
    unlimited_samples=True,
    depth_condition=DepthCondition(dlogZ=jnp.log1p(jnp.float64(1e-3))),
)
target_goal = int(state.goal_loop_iter) + 1
seconds = []
run_peak_rss = []
lifetime_peak_rss = 0
expected = None
for repeat in range(args.repeats + 1):
    gc.collect()
    jax.block_until_ready(state)
    # Preserve loading, compilation and validation peaks across marker resets.
    lifetime_peak_rss = max(
        lifetime_peak_rss,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    # Reset only this process's marker so fingerprinting is excluded from
    # each measured run peak; retained allocator memory still counts.
    Path("/proc/self/clear_refs").write_text("5\n")
    started = time.perf_counter()
    completed = core.resume_until_goal(
        state,
        lambda value: int(value.goal_loop_iter) >= target_goal,
    )
    jax.block_until_ready(completed)
    seconds.append(time.perf_counter() - started)
    status = Path("/proc/self/status").read_text().splitlines()
    run_peak_rss.append(
        int(next(line.split()[1] for line in status if line.startswith("VmHWM:")))
        * 1024
    )
    assert int(completed.goal_loop_iter) == target_goal
    assert int(completed.termination_reason) == 0
    fingerprint = scientific_fingerprints(completed)
    if expected is None:
        expected = fingerprint
    else:
        assert expected == fingerprint
    print(
        json.dumps(
            {
                "repeat": repeat,
                "seconds": seconds[-1],
                "classics": int(completed.num_samples),
            }
        ),
        flush=True,
    )
    if repeat != args.repeats:
        del completed
result = {
    "variant": args.variant,
    "state_path": str(args.state),
    "source_root": str(source_root),
    "source_commit": commit,
    "source_sha256": source_hashes,
    "harness_sha256": harness_hash,
    "conversion_including_compile_seconds": conversion_seconds,
    "cold_seconds": seconds[0],
    "warm_seconds": seconds[1:],
    "median_seconds": float(np.median(seconds[1:])),
    "quartiles_seconds": np.quantile(seconds[1:], [0.25, 0.75]).tolist(),
    "run_peak_rss_bytes": run_peak_rss,
    "warm_peak_rss_bytes": max(run_peak_rss[1:]),
    "memory_measurement_version": 2,
    "total_peak_rss_bytes": max(
        lifetime_peak_rss,
        *run_peak_rss,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    ),
    "input_classics": int(state.num_samples),
    "input_capacity": state.samples.log_likelihoods.size,
    "classics": int(completed.num_samples),
    "capacity": completed.samples.log_likelihoods.size,
    "calls": int(completed.total_num_likelihood_evaluations),
    "new_calls": int(
        completed.total_num_likelihood_evaluations
        - state.total_num_likelihood_evaluations,
    ),
    "goals": int(completed.goal_loop_iter),
    "fingerprints": expected,
    "random_key": np.asarray(completed.random_key).tolist(),
    "goal_key": np.asarray(completed.goal_key).tolist(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "hostname": platform.node(),
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "x64": jax.config.jax_enable_x64,
}
if args.ss10_accuracy:
    from benchmarks.phantom_seeding.accuracy import ss10_accuracy

    del state
    gc.collect()
    result["accuracy"] = ss10_accuracy(completed)
    result["post_analysis_peak_rss_bytes"] = max(
        result["total_peak_rss_bytes"],
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
args.output.write_text(json.dumps(result, indent=2) + "\n")
