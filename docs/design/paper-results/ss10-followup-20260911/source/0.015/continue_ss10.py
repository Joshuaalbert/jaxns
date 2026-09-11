"""Continue an archived SS10 tree to a tighter classic evidence uncertainty.

The sampler, allocation rule and keys are unchanged. A legacy row cache is
explicitly converted to the measured efficient index once, before continuing.
Only the supplied output directory receives progress, checkpoints and results.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import resource
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

import jaxns
from benchmarks.paper_reproduction.run_posterior import classic_metrics
from benchmarks.phantom_seeding.accuracy import ss10_accuracy
from benchmarks.phantom_seeding.state_fingerprints import scientific_fingerprints
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.sampling.phantom_index import build_phantom_seed_index
from jaxns.state import State


def save_checkpoint(state: State, output: Path) -> float:
    """Publish a complete checkpoint atomically, preserving allocated capacity."""
    started = time.perf_counter()
    temporary = output / "checkpoint.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "checkpoint.pkl")
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=.015)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.seed < 30 or not 0 < args.target < .02:
        parser.error("Use seeds 0--29 and a positive uncertainty below 0.02")
    if args.checkpoint_every < 1:
        parser.error("Checkpoint interval must be positive")
    root = Path(__file__).resolve().parents[2]
    assert Path(jaxns.__file__).resolve().parents[2] == root
    assert len(os.sched_getaffinity(0)) == 1
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=root)
    previous = json.loads((args.input / "CORE.json").read_text())
    provenance = json.loads((args.input / "MANIFEST.json").read_text())
    assert previous["complete"] and previous["case"] == "ss10"
    assert previous["seed"] == args.seed and provenance["goal_classic_log_Z_uncert"] == .02
    assert provenance["source_commit"] == "aa79a0dc395c0d64b85cbb88d0e3e732decd7361"
    manifest = dict(
        case="ss10", seed=args.seed, target=args.target, input=str(args.input.resolve()),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((root / "src/jaxns").rglob("*.py"))},
        harness_sha256={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("continue_ss10.py", "accuracy.py", "state_fingerprints.py")
        },
        checkpoint_every=args.checkpoint_every, affinity=sorted(os.sched_getaffinity(0)),
        jax=jax.__version__, x64=jax.config.jax_enable_x64,
        sampler="efficient exact A+C, isotropic, d0=delta_K=300, shell=100, slices=100, P=99",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prior_seconds = 0.
    if args.resume:
        saved_manifest = json.loads((args.output / "MANIFEST.json").read_text())
        assert manifest == saved_manifest, "Resume source or protocol changed"
        if (args.output / "CORE.json").exists():
            return
        source_state = args.output / "checkpoint.pkl"
        logged = [json.loads(line) for line in
                  (args.output / "progress.jsonl").read_text().splitlines()]
        prior_seconds = max(row["seconds"] for row in logged)
    else:
        with (args.output / "MANIFEST.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
        source_state = args.input / "state.pkl"
    started = time.perf_counter()
    with source_state.open("rb") as stream:
        state = pickle.load(stream)
    jax.block_until_ready(state)
    load_seconds = time.perf_counter() - started
    conversion_seconds = 0.
    if not args.resume:
        conversion_started = time.perf_counter()
        state = dataclasses.replace(
            state, phantom_seed_index=build_phantom_seed_index(state.samples, state.num_samples),
            samples=dataclasses.replace(state.samples, phantom_samples=dataclasses.replace(
                state.samples.phantom_samples, seed_log_L_sorted=None,
            )),
        )
        jax.block_until_ready(state)
        conversion_seconds = time.perf_counter() - conversion_started
    gc.collect()
    core = NestedSampler(
        model=state.model, root_allocation_degree=300, shell_size=100, delta_K=300,
        sampler=UniDimSliceSampler(
            model=state.model, num_slices=100, collect_phantom_samples=True,
            max_phantom_samples=99, no_step_out=True,
        ),
        collect_phantom_samples=True, max_phantom_samples=99,
        allocation_target="evidence_improving", unlimited_samples=True,
        depth_condition=DepthCondition(dlogZ=jnp.log1p(jnp.float64(1e-3))),
    )
    # These archived partial trajectories were produced by the frozen old A+C.
    # Check every overlapping goal without loading another large sample tree.
    old_progress = args.input.parents[2] / "0.01/ss10" / args.input.name / "progress.jsonl"
    reference_rows = {}
    if old_progress.exists():
        reference_rows = {row["iteration"]: row for row in
                          (json.loads(line) for line in old_progress.read_text().splitlines())}
    last_goal = -1
    checkpoint_seconds = 0.
    verified_goals = []

    def goal(value: State) -> bool:
        nonlocal last_goal, checkpoint_seconds
        metrics = classic_metrics(value)
        iteration = int(value.goal_loop_iter)
        success = float(metrics.log_Z_uncert) < args.target
        if iteration == last_goal:
            return success
        row = dict(
            pid=os.getpid(), iteration=iteration,
            allocation_iteration=int(value.allocation_loop_iter),
            root_out_degree=int(value.root_out_degree), classic_samples=int(value.num_samples),
            state_capacity=int(value.samples.log_likelihoods.size),
            likelihood_evaluations=int(value.total_num_likelihood_evaluations),
            classic_log_Z=float(metrics.log_Z), classic_log_Z_uncert=float(metrics.log_Z_uncert),
            classic_kish_ess=float(metrics.kish_ess),
            seconds=prior_seconds + time.perf_counter() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        if iteration in reference_rows:
            for field in ("allocation_iteration", "root_out_degree", "classic_samples",
                          "likelihood_evaluations", "classic_log_Z", "classic_log_Z_uncert",
                          "classic_kish_ess"):
                assert row[field] == reference_rows[iteration][field], (iteration, field)
            verified_goals.append(iteration)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        # Checkpoint only complete scientific goals, never an internal growth boundary.
        if iteration % args.checkpoint_every == 0 or success:
            checkpoint_seconds += save_checkpoint(value, args.output)
        last_goal = iteration
        return success

    if not goal(state):
        state = core.resume_until_goal(state, goal)
    jax.block_until_ready(state)
    continuation_seconds = prior_seconds + time.perf_counter() - started
    run_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert int(state.termination_reason) == 0 and goal(state)
    # The final callback has already saved this exact state; retain one inode.
    (args.output / "state.pkl").unlink(missing_ok=True)
    (args.output / "state.pkl").hardlink_to(args.output / "checkpoint.pkl")
    accuracy = ss10_accuracy(state)
    record = dict(
        case="ss10", seed=args.seed, complete=True, target=args.target,
        iteration=int(state.goal_loop_iter), classic_samples=int(state.num_samples),
        state_capacity=int(state.samples.log_likelihoods.size),
        added_goals=int(state.goal_loop_iter) - previous["iteration"],
        likelihood_evaluations=int(state.total_num_likelihood_evaluations),
        added_likelihood_evaluations=(int(state.total_num_likelihood_evaluations)
                                      - previous["likelihood_evaluations"]),
        classic_log_Z=accuracy["log_Z"], classic_log_Z_uncert=accuracy["log_Z_uncert"],
        accuracy=accuracy, continuation_seconds=continuation_seconds,
        load_seconds=load_seconds, conversion_seconds=conversion_seconds,
        checkpoint_seconds_this_attempt=checkpoint_seconds,
        run_peak_rss_bytes=run_peak, verified_archived_goals=verified_goals,
        fingerprints=scientific_fingerprints(state),
        random_key=np.asarray(state.random_key).tolist(),
        goal_key=np.asarray(state.goal_key).tolist(),
        total_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    temporary = args.output / "CORE.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output / "CORE.json")
    print(json.dumps(dict(event="complete", seed=args.seed, accuracy=accuracy)), flush=True)


if __name__ == "__main__":
    main()
