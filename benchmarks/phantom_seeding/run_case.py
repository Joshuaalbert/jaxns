"""Run a bounded paper-protocol case and fingerprint its scientific state.

Use the frozen worktree's PYTHONPATH for reference, and the implementation
worktree for the other variants. Each process is pinned to one CPU. The
original precision ladder and its saved states are never resumed or modified.
"""

import argparse
import dataclasses
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
from benchmarks.paper_reproduction.evidence10_cases import build_case
from benchmarks.phantom_seeding.classic_metrics import classic_metrics
from jaxns.algorithm import depth, initialisation
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("g10", "cg10", "ss10"), required=True)
parser.add_argument("--variant", choices=("reference", "lazy", "blocks", "combined"), required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--goals", type=int, default=3)
parser.add_argument("--uncertainty", type=float)
parser.add_argument("--checkpoint-roundtrip", action="store_true")
parser.add_argument("--save-state", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source_root = Path(jaxns.__file__).resolve().parents[2]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
assert len(os.sched_getaffinity(0)) == 1 and jax.config.jax_enable_x64
if args.variant == "reference":
    assert commit == "aa79a0dc395c0d64b85cbb88d0e3e732decd7361"
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=source_root)
else:
    assert source_root == Path(__file__).resolve().parents[2]
if args.variant == "blocks":
    from benchmarks.phantom_seeding.block_eager import block_eager_stationary_seeds
    depth._sample_stationary_seeds = block_eager_stationary_seeds
if args.variant == "lazy":
    block_builder = initialisation._build_init_state

    def row_index_initial_state(*values, **options):
        state = block_builder(*values, **options)
        phantoms = state.samples.phantom_samples
        return dataclasses.replace(
            state, phantom_seed_index=None,
            samples=dataclasses.replace(state.samples, phantom_samples=dataclasses.replace(
                phantoms, seed_log_L_sorted=jnp.sort(jnp.where(
                    phantoms.valid_mask, phantoms.log_L, -jnp.inf,
                ), axis=-1),
            )),
        )

    initialisation._build_init_state = row_index_initial_state

args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "RESULT.json").exists():
    raise FileExistsError("Use a fresh output directory for each measurement.")
model, reference = build_case(args.case)
condition = DepthCondition(dlogZ=jnp.log1p(jnp.float64(1e-3)))
core = NestedSampler(
    model=model, root_allocation_degree=300, shell_size=100, delta_K=300,
    sampler=UniDimSliceSampler(
        model=model, num_slices=100, collect_phantom_samples=True,
        max_phantom_samples=99, no_step_out=True,
    ),
    collect_phantom_samples=True, max_phantom_samples=99,
    allocation_target="evidence_improving", unlimited_samples=True,
    depth_condition=condition,
)
progress = []
started = time.perf_counter()


def goal(state):
    metrics = classic_metrics(state)
    iteration = int(state.goal_loop_iter)
    if not progress or iteration != progress[-1]["goal"]:
        row = {
            "goal": iteration, "classics": int(state.num_samples),
            "capacity": state.samples.log_likelihoods.size,
            "calls": int(state.total_num_likelihood_evaluations),
            "log_Z": float(metrics.log_Z), "uncertainty": float(metrics.log_Z_uncert),
            "seconds": time.perf_counter() - started,
        }
        progress.append(row)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    if args.uncertainty is not None:
        return float(metrics.log_Z_uncert) < args.uncertainty
    return iteration >= args.goals


# Current develop uses additive allocation: d0 + delta_K * k, starting at k=0.
# The frozen reference uses d0 * delta_K * (k+1); both begin with d0 roots.
bootstrap = dataclasses.replace(
    core, allocation_target="uniform", delta_K=1 if args.variant == "reference" else 300,
)
state = bootstrap.run_until_goal(
    lambda value: int(value.goal_loop_iter) >= 1, key=jax.random.PRNGKey(args.seed),
)
if args.checkpoint_roundtrip:
    state = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
if not goal(state):
    state = core.resume_until_goal(state, goal)
jax.block_until_ready(state)
run_seconds = time.perf_counter() - started
assert int(state.termination_reason) == 0
assert goal(state)
run_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
if args.save_state:
    with (args.output / "state.pkl").open("xb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)

# Cache representations differ; every scientific row, including the original
# phantom coordinates/likelihoods/validity, must retain identical bits.
accepted = int(state.num_samples)
samples = dataclasses.replace(
    state.samples, phantom_samples=dataclasses.replace(
        state.samples.phantom_samples, seed_log_L_sorted=None,
    ),
)
fingerprints = {}
for path, leaf in jax.tree_util.tree_flatten_with_path(samples)[0]:
    value = np.ascontiguousarray(np.asarray(leaf)[:accepted])
    fingerprints[jax.tree_util.keystr(path)] = {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value).hexdigest(),
    }
metrics = classic_metrics(state)
keys = {
    "random_key": np.asarray(state.random_key).tolist(),
    "goal_key": np.asarray(state.goal_key).tolist(),
}
result = {
    "case": args.case, "variant": args.variant, "seed": args.seed,
    "goals": int(state.goal_loop_iter), "allocation_goals": int(state.allocation_loop_iter),
    "depth_iterations": int(state.depth_loop_iter), "root_out_degree": int(state.root_out_degree),
    "classics": accepted, "calls": int(state.total_num_likelihood_evaluations),
    "capacity": state.samples.log_likelihoods.size, "keys": keys, "fingerprints": fingerprints,
    "log_Z": float(metrics.log_Z), "uncertainty": float(metrics.log_Z_uncert),
    "kish_ess": float(metrics.kish_ess), "reference": reference,
    "run_including_compilation_seconds": run_seconds, "run_peak_rss_bytes": run_peak,
    "checkpoint_roundtrip": args.checkpoint_roundtrip, "progress": progress,
    "source_root": str(source_root), "source_commit": commit,
    "source_sha256": {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source_root / "src/jaxns").rglob("*.py"))
    },
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "affinity": sorted(os.sched_getaffinity(0)), "hostname": platform.node(),
    "jax": jax.__version__, "backend": jax.default_backend(), "x64": jax.config.jax_enable_x64,
}
(args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "case", "variant", "seed", "classics", "calls", "log_Z", "uncertainty",
    "run_including_compilation_seconds", "run_peak_rss_bytes",
)}), flush=True)
