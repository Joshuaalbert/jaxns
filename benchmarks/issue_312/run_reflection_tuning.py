"""One CPU matched-work run, with immutable snapshots and offline chain replay."""

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import subprocess
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--arm", choices=("fixed", "tuned"), required=True)
    parser.add_argument(
        "--allocation", choices=("uniform", "evidence_improving"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.tuning_common import (
        BASELINE,
        OUTPUT_ROOT,
        SOURCE_ROOT,
        atomic_json,
        atomic_npz,
        make_model,
    )
    from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
    from jaxns.algorithm.depth import _refresh_likelihood_order
    from jaxns.core import NestedSampler
    from jaxns.pytree import pytree_ravel

    if not options.smoke:
        gate = json.loads((OUTPUT_ROOT / "validation/VALIDATION.json").read_text())
        assert (
            gate["passed"]
            and gate["schedule_sha256"]
            == hashlib.sha256(
                options.schedule.read_bytes(),
            ).hexdigest()
        )
        assert 0 <= options.seed < 30
    model = make_model()
    assert Path(__import__("jaxns").__file__).resolve().is_relative_to(SOURCE_ROOT)
    unit, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122799)))
    assert unit.dtype == jnp.float64 and unit.shape == (18,)

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    schedule = json.loads(options.schedule.read_text())
    peak = float(likelihood(jnp.full(18, 0.5)))
    assert peak == schedule["peak"]
    kernel = dict(num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=0.003)
    sampler = (
        ReflectionSampler(**kernel)
        if options.arm == "fixed"
        else FrozenContourReflectionSampler(
            **kernel,
            gaps=tuple(schedule["gaps"]),
            steps=tuple(schedule["steps"]),
            peak=peak,
        )
    )
    runner = NestedSampler(
        model=model,
        root_allocation_degree=150,
        replacement_width=150,
        delta_K=150,
        sampler=sampler,
        allocation_target=options.allocation,
        unlimited_samples=True,
        collect_phantom_samples=True,
    )
    budgets = [30000, 60000] if options.smoke else [2000000, 4000000]
    folder = options.output
    folder.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((SOURCE_ROOT / "src/jaxns").rglob("*.py"))
    source_paths += [
        SOURCE_ROOT / "benchmarks/issue_312" / name
        for name in (
            "REFLECTION_TUNING_PROTOCOL.md",
            "tuning_schedule.py",
            "tuning_common.py",
            "reflection_sampler.py",
            "allocation_observer.py",
            "allocation_phantoms.py",
            "run_reflection_tuning.py",
            "run_ridge.py",
        )
    ]
    manifest = {
        "baseline_commit": BASELINE,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "schedule_sha256": hashlib.sha256(options.schedule.read_bytes()).hexdigest(),
        "versions": {
            name: version(name)
            for name in ("jax", "jaxlib", "jaxctx", "numpy", "scipy", "tfp-nightly")
        },
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "arm": options.arm,
        "allocation": options.allocation,
        "seed": options.seed,
        "budgets": budgets,
        "initial_roots": 150,
        "replacement_width": 150,
        "delta_K": 150,
        "unit_dtype": "float64",
        "phantoms": 8,
        "phantom_seed_selection": False,
        "sigma_stiff": 1e-5,
        "curvature": 0.3,
        "peak": peak,
        "dlogZ": float(np.log1p(1e-3)),
        "kernel": kernel,
        "unlimited_samples": True,
        "smoke": options.smoke,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
    }
    atomic_json(folder / "MANIFEST.json", manifest)
    started = time.monotonic()
    initial = runner.initialise(jax.random.PRNGKey(options.seed))
    assert initial.samples.phantom_samples.U_samples is None
    initial_points = np.asarray(jax.tree.leaves(initial.samples.U_samples)[0])[:150]
    initial_values = np.asarray(initial.samples.log_likelihoods)[:150]
    initial_hash = hashlib.sha256(
        initial_points.tobytes() + initial_values.tobytes()
    ).hexdigest()
    seeds, parents, batches = [[np.full(150, -1, dtype=np.int64)] for _ in range(3)]
    keys = [np.zeros((150, 2), np.uint32)]
    records = []
    snapshot_seconds = 0.0
    next_budget = 0

    def observe(before, current, work):
        nonlocal next_budget, snapshot_seconds
        start, count = int(before.num_samples), int(current.num_samples)
        width = count - start
        seeds.append(np.asarray(work.seed_idx[:width], dtype=np.int64))
        parents.append(np.asarray(work.parent_idx[:width], dtype=np.int64))
        batches.append(np.full(width, int(current.depth_loop_iter), dtype=np.int64))
        new_key, _ = jax.random.split(before.random_key)
        depth_key = jnp.where(before.depth_reached, new_key, before.random_key)
        _, sample_key, _ = jax.random.split(depth_key, 3)
        keys.append(np.asarray(jax.random.split(sample_key, work.seed_idx.shape[0]))[:width])
        evaluations = int(current.total_num_likelihood_evaluations)
        if int(current.depth_loop_iter) % 10 == 0 or evaluations >= budgets[next_budget]:
            with (folder / "progress.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "seconds": time.monotonic() - started,
                            "evaluations": evaluations,
                            "samples": count,
                            "batches": int(current.depth_loop_iter),
                            "completed_epochs": int(current.goal_loop_iter),
                            "capacity": current.samples.log_likelihoods.shape[0],
                            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        }
                    )
                    + "\n"
                )
        while next_budget < len(budgets) and evaluations >= budgets[next_budget]:
            snapshot_start = time.monotonic()
            budget = budgets[next_budget]
            result = jax.block_until_ready(_refresh_likelihood_order(current).to_result())
            assert current.samples.phantom_samples.U_samples is None
            points = np.asarray(result.X_samples["x"])[:count]
            weights = np.asarray(result.log_dp)[:count]
            probability = np.exp(weights - np.max(weights))
            probability /= probability.sum()
            mean = np.sum(points * probability[:, None], axis=0)
            std = np.sqrt(np.sum((points - mean) ** 2 * probability[:, None], axis=0))
            seed_idx, parent_idx, batch_idx = map(np.concatenate, (seeds, parents, batches))
            assert len(seed_idx) == count
            roots = int(current.root_out_degree)
            assert roots == np.sum(seed_idx < 0)
            archive = folder / f"calls-{budget}.npz"
            atomic_npz(
                archive,
                samples=points,
                unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
                log_weights=weights,
                log_likelihood=np.asarray(result.log_L)[:count],
                log_volume=np.asarray(result.log_X_mean)[:count],
                log_L_constraints=np.asarray(result.log_L_constraints)[:count],
                seed_indices=seed_idx,
                parent_indices=parent_idx,
                batch_indices=batch_idx,
                chain_keys=np.concatenate(keys),
                out_degrees=np.asarray(current.samples.out_degree)[:count],
                evaluations_per_sample=np.asarray(result.num_likelihood_evaluations_per_sample)[
                    :count
                ],
            )
            inputs = EvidenceInputs.from_result(result)
            np.testing.assert_array_equal(
                np.asarray(inputs.valid_phantom)[:count], seed_idx >= 0
            )
            assert not np.asarray(inputs.valid_phantom)[count:].any()
            phantom_path = folder / f"inputs-{budget}.npz"
            temporary = folder / f"inputs-{budget}.tmp.npz"
            inputs.save(temporary)
            temporary.replace(phantom_path)
            phantom_digest = hashlib.sha256(phantom_path.read_bytes()).hexdigest()
            phantom_path.with_suffix(".npz.sha256").write_text(
                phantom_digest + "  " + phantom_path.name + "\n",
            )
            previous = int(before.total_num_likelihood_evaluations)
            record = {
                "arm": options.arm,
                "allocation": options.allocation,
                "seed": options.seed,
                "budget": budget,
                "evaluations": evaluations,
                "evaluations_before_last_batch": previous,
                "last_batch_evaluations": evaluations - previous,
                "samples": count,
                "roots": roots,
                "initial_sha256": initial_hash,
                "gradient_evaluations": 90 * (count - roots),
                "value_only_evaluations": evaluations - 90 * (count - roots),
                "log_Z": float(result.log_Z_mean),
                "log_Z_uncert": float(result.log_Z_uncert),
                "kish_ess": float(1 / np.sum(probability**2)),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "peak_gap": float(peak - result.log_L_supremum),
                "completed_epochs": int(current.goal_loop_iter),
                "allocation_round": int(current.allocation_loop_iter),
                "batches": int(current.depth_loop_iter),
                "termination_reason": int(current.termination_reason),
                "scientific_goal_claimed": False,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "phantom_archive_sha256": phantom_digest,
                "seconds_including_compilation_and_snapshots": time.monotonic() - started,
                "prior_snapshot_seconds": snapshot_seconds,
                "snapshot_seconds": time.monotonic() - snapshot_start,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "random_key": np.asarray(current.random_key).tolist(),
            }
            snapshot_seconds += record["snapshot_seconds"]
            records.append(record)
            atomic_json(folder / f"CORE-{budget}.json", record)
            print(json.dumps(record), flush=True)
            next_budget += 1
        return next_budget == len(budgets)

    final = run_with_observer(runner, initial, observe, lambda current: False)
    assert len(records) == 2 and int(final.termination_reason) == 0
    temporary = folder / "state.tmp.pkl"
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(final), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(folder / "state.pkl")
    state_digest = hashlib.sha256((folder / "state.pkl").read_bytes()).hexdigest()
    (folder / "state.pkl.sha256").write_text(state_digest + "  state.pkl\n")
    atomic_json(
        folder / "COMPLETE.json",
        {
            "core_complete": True,
            "records": records,
            "state_sha256": state_digest,
            "seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


if __name__ == "__main__":
    main()
