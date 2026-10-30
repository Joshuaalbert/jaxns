"""Replay the frozen allocation cohort, retaining phantoms and checking classic parity.

Example: PYTHONPATH=.:src JAX_ENABLE_X64=1 python -m \
benchmarks.issue_312.run_allocation_phantoms --kernel hessian --allocation uniform \
--seeds 0 1 --cpu 0 --output-dir /tmp/jaxns-allocation-phantoms-hessian-uniform-a
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("hessian", "reflection"), required=True)
    parser.add_argument("--allocation", choices=(
        "uniform", "evidence_improving", "posterior_improving",
    ), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=Path("/tmp"))
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})

    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
    from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.algorithm.depth import _refresh_likelihood_order
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    model = Model(prior_model=partial(prior_model, 1e-5, 0.3))
    sampler = (ToyHessianSliceSampler(num_slices=90, scale_to_contour=True)
               if options.kernel == "hessian" else ReflectionSampler(
                   num_cycles=9, slices_per_cycle=1, num_bounces=10,
                   step_size=0.003, boundary_only=False,
               ))
    runner = NestedSampler(
        model=model, root_allocation_degree=150, replacement_width=150, delta_K=150,
        sampler=sampler, allocation_target=options.allocation, unlimited_samples=True,
        collect_phantom_samples=True,
    )
    folder = options.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    source_folder = Path(__file__).resolve().parent
    new_sources = {name: hashlib.sha256((source_folder / name).read_bytes()).hexdigest()
                   for name in ("run_allocation_phantoms.py", "allocation_phantoms.py")}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for seed in options.seeds:
        letter = "a" if seed < 15 else "b"
        stem = f"allocation-{options.kernel}-{options.allocation}-{letter}"
        baseline = {row["budget"]: row for row in (
            json.loads(line) for line in (source_folder / f"raw/{stem}.jsonl")
            .read_text().splitlines()
        ) if row["seed"] == seed}
        for name, digest in baseline[2000000]["source_sha256"].items():
            path = (source_folder.parents[1] / name
                    if name.startswith("src/") else source_folder / name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen baseline source changed: {name}")
        directory = folder / f"seed-{seed}"
        directory.mkdir()
        started = time.monotonic()
        initial = runner.initialise(jax.random.PRNGKey(seed))
        seed_chunks, parent_chunks, batch_chunks = [np.full(150, -1)], [np.full(150, -1)], [
            np.full(150, -1),
        ]
        budgets = [2000000, 4000000]
        next_budget = 0

        def observe(
            before, current, work, seed_chunks=seed_chunks, parent_chunks=parent_chunks,
            batch_chunks=batch_chunks, budgets=budgets, baseline=baseline, stem=stem,
            seed=seed, directory=directory, started=started,
        ):
            nonlocal next_budget
            start, count = int(before.num_samples), int(current.num_samples)
            width = count - start
            seed_chunks.append(np.asarray(work.seed_idx[:width]))
            parent_chunks.append(np.asarray(work.parent_idx[:width]))
            batch_chunks.append(np.full(width, int(current.depth_loop_iter)))
            budget = budgets[next_budget]
            if int(current.total_num_likelihood_evaluations) < budget:
                return False
            reference = baseline[budget]
            result = jax.block_until_ready(_refresh_likelihood_order(current).to_result())
            archive = options.archive_root / f"jaxns-{stem}/seed-{seed}/calls-{budget}.npz"
            if hashlib.sha256(archive.read_bytes()).hexdigest() != reference["archive_sha256"]:
                raise RuntimeError(f"Baseline archive changed: {archive}")
            with np.load(archive) as saved:
                comparisons = {
                    "samples": np.asarray(result.X_samples["x"])[:count],
                    "unit_samples": np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
                    "log_likelihood": np.asarray(result.log_L)[:count],
                    "log_weights": np.asarray(result.log_dp)[:count],
                    "log_volume": np.asarray(result.log_X_mean)[:count],
                    "log_L_constraints": np.asarray(result.log_L_constraints)[:count],
                    "evaluations_per_sample": np.asarray(
                        result.num_likelihood_evaluations_per_sample,
                    )[:count],
                    "seed_indices": np.concatenate(seed_chunks),
                    "parent_indices": np.concatenate(parent_chunks),
                    "batch_indices": np.concatenate(batch_chunks),
                }
                for name, actual in comparisons.items():
                    np.testing.assert_array_equal(actual, saved[name], err_msg=name)
            assert float(result.log_Z_mean) == reference["log_Z"]
            assert float(result.log_Z_uncert) == reference["log_Z_uncert"]
            assert int(current.total_num_likelihood_evaluations) == reference["evaluations"]
            assert int(current.root_out_degree) == reference["roots"]
            assert int(current.goal_loop_iter) == reference["completed_epochs"]
            inputs = EvidenceInputs.from_result(result)
            valid = np.asarray(inputs.valid_phantom)
            # New prior roots have no chain; every constrained child has one.
            np.testing.assert_array_equal(valid[:count], np.concatenate(seed_chunks) >= 0)
            assert not valid[count:].any()
            assert np.all(np.asarray(inputs.log_L_phantom)[valid]
                          > np.asarray(inputs.log_L_constraints)[valid, None])
            saved_path = directory / f"calls-{budget}.npz"
            inputs.save(saved_path)
            record = {
                **reference, "replay_commit": commit, "replay_source_sha256": new_sources,
                "phantoms": sampler.max_num_phantom(), "classic_bit_identical": True,
                "phantom_archive_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                "phantom_archive": str(saved_path),
                "replay_seconds": time.monotonic() - started,
            }
            with (folder / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"seed={seed} budget={budget} classic_equal=True "
                  f"phantoms={sampler.max_num_phantom()}", flush=True)
            next_budget += 1
            return next_budget == len(budgets)

        final = run_with_observer(runner, initial, observe, lambda current: False)
        if next_budget != 2 or int(final.termination_reason) != 0:
            raise RuntimeError(f"Seed {seed} did not reproduce both budget snapshots.")


if __name__ == "__main__":
    main()
