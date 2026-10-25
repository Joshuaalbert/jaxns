"""Run unchanged ridge kernels with three allocation policies at equal work.

See ALLOCATION_PROTOCOL.md. Every process runs one kernel/allocation cell;
--seeds selects an independently reproducible shard of that cell.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("hessian", "reflection"), required=True)
    parser.add_argument("--allocation", choices=(
        "uniform", "evidence_improving", "posterior_improving",
    ), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[2000000, 4000000])
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if options.budgets != sorted(set(options.budgets)) or min(options.budgets) <= 150:
        parser.error("Budgets must be increasing distinct integers exceeding the 150 roots.")
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("Seed indices must be distinct.")
    os.sched_setaffinity(0, {options.cpu})

    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.algorithm.depth import _refresh_likelihood_order
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    model = Model(prior_model=partial(prior_model, 1e-5, 0.3))
    if options.kernel == "hessian":
        sampler = ToyHessianSliceSampler(num_slices=90, scale_to_contour=True)
        kernel_config = {"num_slices": 90, "scale_to_contour": True, "correct_mh": True}
        gradients_per_child = 0
    else:
        sampler = ReflectionSampler(
            num_cycles=9, slices_per_cycle=1, num_bounces=10,
            step_size=0.003, boundary_only=False,
        )
        kernel_config = {
            "num_cycles": 9, "slices_per_cycle": 1, "num_bounces": 10,
            "step_size": 0.003, "boundary_only": False,
        }
        gradients_per_child = 90
    runner = NestedSampler(
        model=model, root_allocation_degree=150, replacement_width=150, delta_K=150,
        sampler=sampler, allocation_target=options.allocation, unlimited_samples=True,
    )
    folder = options.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    source_folder = Path(__file__).resolve().parent
    source_paths = {name: source_folder / name for name in (
        "run_allocation_comparison.py", "allocation_observer.py", "hessian_sampler.py",
        "reflection_sampler.py", "run_ridge.py",
    )}
    for name in ("core.py", "algorithm/depth.py", "algorithm/allocation.py"):
        source_paths[f"src/jaxns/{name}"] = source_folder.parents[1] / "src/jaxns" / name
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in source_paths.items()},
        "jax": jax.__version__, "jaxctx": version("jaxctx"),
        "python": platform.python_version(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "kernel": options.kernel, "kernel_config": kernel_config,
        "allocation": options.allocation, "initial_roots": 150,
        "replacement_width": 150, "delta_K": 150, "unit_dtype": "float64", "phantoms": 0,
        "sigma_stiff": 1e-5, "curvature": 0.3, "dlogZ": float(np.log1p(1e-3)),
        "unlimited_samples": True, "budgets": options.budgets,
        "stop_rule": "first_complete_batch_crossing_model_call_budget",
        "evaluation_accounting": "includes_each_value_and_grad_once; gradients_also_separate",
    }
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    for seed in options.seeds:
        directory = folder / f"seed-{seed}"
        directory.mkdir()
        started = time.monotonic()
        initial = runner.initialise(jax.random.PRNGKey(seed))
        initial_points = np.asarray(jax.tree.leaves(initial.samples.U_samples)[0])[:150]
        initial_values = np.asarray(initial.samples.log_likelihoods)[:150]
        initial_hash = hashlib.sha256(
            initial_points.tobytes() + initial_values.tobytes(),
        ).hexdigest()
        seed_chunks = [np.full(150, -1, dtype=np.int64)]
        parent_chunks = [np.full(150, -1, dtype=np.int64)]
        batch_chunks = [np.full(150, -1, dtype=np.int64)]
        snapshots = []
        next_budget = 0

        def observe(
            before, current, work, seed=seed, directory=directory, started=started,
            initial_hash=initial_hash, seed_chunks=seed_chunks, parent_chunks=parent_chunks,
            batch_chunks=batch_chunks, snapshots=snapshots,
        ):
            nonlocal next_budget
            start, count = int(before.num_samples), int(current.num_samples)
            width = count - start
            seed_chunks.append(np.asarray(work.seed_idx[:width], dtype=np.int64))
            parent_chunks.append(np.asarray(work.parent_idx[:width], dtype=np.int64))
            batch_chunks.append(np.full(width, int(current.depth_loop_iter), dtype=np.int64))
            evaluations = int(current.total_num_likelihood_evaluations)
            previous_evaluations = int(before.total_num_likelihood_evaluations)
            while (next_budget < len(options.budgets)
                   and evaluations >= options.budgets[next_budget]):
                budget = options.budgets[next_budget]
                # Publish an exact ordering on an analysis copy. Keep the
                # continuing schedule and all random keys completely untouched.
                result = jax.block_until_ready(_refresh_likelihood_order(current).to_result())
                points = np.asarray(result.X_samples["x"])[:count]
                values = np.asarray(result.log_L)[:count]
                constraints = np.asarray(result.log_L_constraints)[:count]
                roots = int(current.root_out_degree)
                seeds = np.concatenate(seed_chunks)
                parents = np.concatenate(parent_chunks)
                batches = np.concatenate(batch_chunks)
                if (len(seeds) != count or roots != np.sum(seeds < 0)
                        or roots != np.sum(np.isneginf(constraints))):
                    raise RuntimeError("Observer seed/root counts do not match accepted samples.")
                log_weights = np.asarray(result.log_dp)[:count]
                weights = np.exp(log_weights - np.max(log_weights))
                weights /= weights.sum()
                mean = np.sum(points * weights[:, None], axis=0)
                std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
                archive = directory / f"calls-{budget}.npz"
                np.savez_compressed(
                    archive, samples=points,
                    unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
                    log_weights=log_weights, log_likelihood=values,
                    log_volume=np.asarray(result.log_X_mean)[:count], log_L_constraints=constraints,
                    seed_indices=seeds, parent_indices=parents, batch_indices=batches,
                    evaluations_per_sample=np.asarray(
                        result.num_likelihood_evaluations_per_sample,
                    )[:count],
                )
                gradients = (count - roots) * gradients_per_child
                record = {
                    **metadata, "seed": seed, "budget": budget,
                    "evaluations": evaluations,
                    "evaluations_before_last_batch": previous_evaluations,
                    "last_batch_evaluations": evaluations - previous_evaluations,
                    "gradient_evaluations": gradients,
                    "value_only_evaluations": evaluations - gradients,
                    "samples": count, "roots": roots, "initial_sha256": initial_hash,
                    "log_Z": float(result.log_Z_mean), "log_Z_uncert": float(result.log_Z_uncert),
                    "peak_gap": float(peak - result.log_L_supremum),
                    "kish_ess": float(1 / np.sum(weights ** 2)),
                    **{f"std_x{i}": float(std[i]) for i in (1, 16, 17, 5)},
                    "completed_epochs": int(current.goal_loop_iter),
                    "allocation_round": int(current.allocation_loop_iter),
                    "batches": int(current.depth_loop_iter),
                    "termination_reason": int(current.termination_reason),
                    "scientific_goal_claimed": False,
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "seconds": time.monotonic() - started, "timing": "including_compilation",
                }
                snapshots.append(record)
                with (folder / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                next_budget += 1
            return next_budget == len(options.budgets)

        final = run_with_observer(runner, initial, observe, lambda current: False)
        if len(snapshots) != len(options.budgets) or int(final.termination_reason) != 0:
            raise RuntimeError(f"Seed {seed} did not produce every requested budget snapshot.")


if __name__ == "__main__":
    main()
