"""Run one evidence-paper cell: A+C at d0=300, followed by paired prefix analysis."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
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
from benchmarks.paper_reproduction.posterior_cases import DIMENSION
from benchmarks.paper_reproduction.posterior_cases import LEGACY_COMMIT
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.state import State
from benchmarks.paper_reproduction.run_posterior import classic_metrics
from benchmarks.paper_reproduction.evidence10_cases import CASES
from benchmarks.paper_reproduction.evidence10_cases import CASE_REVISION
from benchmarks.paper_reproduction.evidence10_cases import build_case
from benchmarks.paper_reproduction.posterior_cases import ss10_parameters
from benchmarks.paper_reproduction.prefix_sweep import sample_phantom_prefix_sweep_reference
from scipy.special import logsumexp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("core", "analysis"), required=True)
    parser.add_argument("--mc-draws", type=int, default=2048)
    parser.add_argument("--goal-log-z-uncert", type=float, default=.05)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed < 30 or args.mc_draws < 2:
        raise ValueError("Use seeds 0--29 and at least two shrinkage draws")
    if args.goal_log_z_uncert not in (.05, .02, .01, .005):
        raise ValueError("Use a predeclared uncertainty target")
    if args.case != "ss10" and args.goal_log_z_uncert != .05:
        raise ValueError("The tighter uncertainty ladder is only for SS10")
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Pin each worker to exactly one CPU")
    if Path(jaxns.__file__).resolve().parents[2] != ROOT:
        raise ValueError("Package imported from a different worktree")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise ValueError("These runs require CPU float64")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "src",
                                "benchmarks/paper_reproduction"], cwd=ROOT):
        raise ValueError("Freeze experiment sources before launching")
    model, reference = build_case(args.case)
    if args.phase == "analysis":
        analyse(args, reference)
        return
    weights = None
    target = "evidence_improving"
    depth = DepthCondition(dlogZ=jnp.log1p(jnp.float64(1e-3)))
    core = NestedSampler(
        model=model, root_allocation_degree=30 * DIMENSION,
        shell_size=10 * DIMENSION, delta_K=30 * DIMENSION,
        sampler=UniDimSliceSampler(
            model=model, num_slices=10 * DIMENSION,
            collect_phantom_samples=True,
            max_phantom_samples=10 * DIMENSION - 1, no_step_out=True,
        ),
        collect_phantom_samples=True, max_phantom_samples=10 * DIMENSION - 1,
        allocation_target=target, allocation_weights=weights,
        unlimited_samples=True, depth_condition=depth,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case": args.case, "seed": args.seed, "dimension": DIMENSION,
        "case_revision": CASE_REVISION, "intervention": "A+C",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "legacy_commit": LEGACY_COMMIT,
        "source_root": str(ROOT), "reference": reference,
        "ss10_definition": "repository" if args.case == "ss10" else None,
        "root_degree": 300, "delta_K": 300, "shell_size": 100,
        "num_slices": 100, "retained_phantoms": 99,
        "all_phantom_seeds": True, "direction": "isotropic",
        "allocation_target": target,
        "allocation_weights": None if weights is None else dataclasses.asdict(weights),
        "allocation_rule": "current_K + ceil(300 * unit_peak_evidence_utility)",
        "bootstrap": {"allocation_target": "uniform", "delta_K": 1, "goals": 1},
        "goal_classic_log_Z_uncert": args.goal_log_z_uncert,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "checkpoint_every_goals": 5, "checkpoint_preserves_capacity": True,
        "evidence_prefix_sizes": list(range(0, 91, 10)), "C_min": 20.,
        "depth_dlogZ": float(depth.dlogZ),
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("jax", "jaxlib", "jaxctx", "numpy", "scipy")},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),
                         Path(__file__).with_name("evidence10_cases.py"),
                         Path(__file__).with_name("posterior_cases.py"),
                         Path(__file__).with_name("run_posterior.py"),
                         Path(__file__).with_name("prefix_sweep.py"),
                         Path(__file__).with_name("references.py"))
        },
    }
    with (args.output / "MANIFEST.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    started = time.perf_counter()
    progress = []
    preceding_run_seconds = 0.
    if args.resume_from is not None:
        previous = json.loads((args.resume_from / "CORE.json").read_text())
        previous_manifest = json.loads((args.resume_from / "MANIFEST.json").read_text())
        if (previous["case"] != args.case or previous["seed"] != args.seed
                or previous_manifest["source_commit"] != manifest["source_commit"]
                or previous_manifest["case_revision"] != CASE_REVISION):
            raise ValueError("Resume state does not match this frozen experiment")
        progress = previous["goal_progress"].copy()
        preceding_run_seconds = previous["run_seconds"]

    def goal(state: State) -> bool:
        if int(state.goal_loop_iter) == 0:
            return False
        metrics = classic_metrics(state)
        row = {
            "iteration": int(state.goal_loop_iter),
            "allocation_iteration": int(state.allocation_loop_iter),
            "root_out_degree": int(state.root_out_degree),
            "classic_log_Z": float(metrics.log_Z),
            "classic_log_Z_uncert": float(metrics.log_Z_uncert),
            "classic_kish_ess": float(metrics.kish_ess),
            "likelihood_evaluations": int(np.asarray(
                state.samples.num_likelihood_evaluations,
            ).sum()),
            "classic_samples": int(state.num_samples),
            "seconds": preceding_run_seconds + time.perf_counter() - started,
        }
        if not progress or row["iteration"] != progress[-1]["iteration"]:
            progress.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            # Save a resumable state periodically without writing the full
            # growing phantom population after every goal in a 90-run cohort.
            if (row["iteration"] == 1 or row["iteration"] % 5 == 0
                    or float(metrics.log_Z_uncert) < args.goal_log_z_uncert):
                temp = args.output / "checkpoint.tmp"
                with temp.open("wb") as stream:
                    pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
                temp.replace(args.output / "checkpoint.pkl")
        return float(metrics.log_Z_uncert) < args.goal_log_z_uncert

    if args.resume_from is None:
        bootstrap = dataclasses.replace(
            core, allocation_target="uniform", allocation_weights=None, delta_K=1,
        )
        state = bootstrap.run_until_goal(
            lambda state: int(state.goal_loop_iter) == 1,
            depth_cond=depth, key=jax.random.PRNGKey(args.seed),
        )
    else:
        with (args.resume_from / "state.pkl").open("rb") as stream:
            state = pickle.load(stream)
    if not goal(state):
        state = core.resume_until_goal(state, goal, depth_cond=depth)
    jax.block_until_ready(state)
    stage_run_seconds = time.perf_counter() - started
    run_seconds = preceding_run_seconds + stage_run_seconds
    metrics = classic_metrics(state)
    success = float(metrics.log_Z_uncert) < args.goal_log_z_uncert
    if not success:
        raise RuntimeError("Sampler stopped before the requested goal; checkpoint saved")
    with (args.output / "state.pkl").open("xb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    # The final checkpoint and finished state are the same durable object;
    # a hard link avoids retaining two large phantom populations on disk.
    checkpoint = args.output / "checkpoint.pkl"
    checkpoint.unlink(missing_ok=True)
    checkpoint.hardlink_to(args.output / "state.pkl")
    results = state.trim().to_result().trim()
    np.testing.assert_allclose(float(results.ess), float(metrics.kish_ess), rtol=1e-10)
    x = np.asarray(jax.tree.leaves(results.X_samples)[0])  # [N, D]
    log_weights = np.asarray(results.log_dp)  # [N], classic expected weights
    np.savez_compressed(args.output / "classic_posterior.npz",
                        x=x, log_dp=log_weights, log_L=np.asarray(results.log_L))
    probabilities = np.exp(log_weights)
    probabilities /= probabilities.sum()
    mean = probabilities @ x  # [D]
    variance = probabilities @ ((x - mean) ** 2)  # [D]
    record = {
        **progress[-1], "case": args.case, "seed": args.seed, "complete": True,
        "run_seconds": run_seconds, "stage_run_seconds": stage_run_seconds,
        "goal_progress": progress, "goal_log_Z_uncert": args.goal_log_z_uncert,
        "reference": reference, "posterior_mean": mean.tolist(),
        "posterior_standard_deviation": np.sqrt(variance).tolist(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "state_capacity": int(state.samples.log_likelihoods.size),
    }
    with (args.output / "CORE.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record, allow_nan=False), flush=True)


def analyse(args: argparse.Namespace, reference: dict) -> None:
    """Reduce the saved tree, holding classic posterior weights fixed."""
    started = time.perf_counter()
    manifest = json.loads((args.output / "MANIFEST.json").read_text())
    if manifest["case_revision"] != CASE_REVISION:
        raise ValueError("Refusing a superseded problem definition")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    sampling_commit = manifest["source_commit"]
    # An analysis-only correction may have a different commit, but the saved
    # tree's complete sampler source and model definitions must still match.
    for revision in (sampling_commit, source_commit):
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{revision}:src"], cwd=ROOT, text=True,
        ).strip()
        if revision == sampling_commit:
            sampling_tree = tree
        elif tree != sampling_tree:
            raise ValueError("Analysis sampler source differs from the saved tree")
    for name in ("evidence10_cases.py", "posterior_cases.py", "references.py"):
        digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if digest != manifest["source_sha256"][name]:
            raise ValueError(f"Analysis model/reference source differs: {name}")
    with (args.output / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    results = state.trim().to_result().trim()
    blocks = results.block_data.to_block_state()
    block_sizes = np.asarray(blocks.block_size)
    posterior = np.load(args.output / "classic_posterior.npz")
    weights = np.exp(posterior["log_dp"] - logsumexp(posterior["log_dp"]))
    np.testing.assert_allclose(1. / np.sum(weights**2), float(results.ess), rtol=1e-10)
    spike_mass = None
    if args.case == "ss10":
        means, scales = ss10_parameters("repository")
        component_log_L = np.stack([
            norm.logpdf(posterior["x"], loc=mean, scale=scale).sum(axis=1)
            for mean, scale in zip(means, scales, strict=True)
        ], axis=1)  # [N,2]
        responsibility = np.exp(component_log_L[:, 0]
                                - logsumexp(component_log_L, axis=1))  # [N]
        spike_mass = float(weights @ responsibility)
    draws, gates = sample_phantom_prefix_sweep_reference(
        key=jax.random.fold_in(jax.random.PRNGKey(args.seed), 1),
        log_L_constraints=results.log_L_constraints,
        K_classic=results.num_live_points_per_sample,
        valid_phantom=results.valid_phantom,
        log_L_phantom=results.log_L_phantom[:, :90],
        num_samples=results.total_num_samples,
        block_state=blocks,
        dimension=10, num_draws=args.mc_draws, batch_size=1,
        num_workers=1, num_groups=9, C_min=20.,
    )
    np.savez_compressed(args.output / "evidence_draws.npz", log_Z=draws)
    record = {
        "case": args.case, "seed": args.seed, "case_revision": CASE_REVISION,
        "source_commit": sampling_commit, "reference": reference,
        "analysis_source_commit": source_commit,
        "analysis_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("prefix_sweep.py"))
        },
        "num_tied_blocks": int(np.sum(block_sizes > 1)),
        "largest_block": int(np.max(block_sizes)),
        "goal_log_Z_uncert": manifest["goal_classic_log_Z_uncert"],
        "classic_spike_mass": spike_mass,
        "classic_kish_ess": float(results.ess), "mc_draws": args.mc_draws,
        "prefix_sizes": list(range(0, 91, 10)),
        "log_Z_mean": draws.mean(axis=0).tolist(),
        "log_Z_uncert": draws.std(axis=0, ddof=1).tolist(),
        "gate_fraction": gates.mean(axis=1).tolist(),
        "analysis_seconds": time.perf_counter()-started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    with (args.output / "ANALYSIS.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
