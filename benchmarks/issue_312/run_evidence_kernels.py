"""Compare unchanged Hessian/reflection kernels at a classic evidence-SD goal.

Example (from the repository root)::

    PYTHONPATH=.:src JAX_ENABLE_X64=1 conda run -n jaxns_py python \
        benchmarks/issue_312/run_evidence_kernels.py --kernel hessian \
        --cpu 0 --seeds 0 1 2 --output-dir /tmp/evidence-hessian

The outer goal is independent of the default remaining-evidence depth condition.
Each seed starts from the prior and uses evidence_improving throughout.
"""

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.4)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if not math.isfinite(options.target) or options.target <= 0:
        raise ValueError("The classic log-evidence SD goal must be positive and finite.")
    if len(set(options.seeds)) != len(options.seeds):
        raise ValueError("Seeds must be distinct within a job.")

    # Pin before JAX is imported so each process owns one CPU's thread pools.
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.core import NestedSampler
    from jaxns.model import Model
    from jaxns.state import State

    # JAX's x64 setting alone does not change JAXCTX's default base dtype.
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
        model=model, root_allocation_degree=150, sampler=sampler,
        allocation_target="evidence_improving", unlimited_samples=True,
    )
    folder = options.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "run_evidence_kernels.py", "hessian_sampler.py", "reflection_sampler.py",
                "run_ridge.py",
            )
        },
        "jax": jax.__version__, "jaxctx": version("jaxctx"),
        "python": platform.python_version(), "device": str(jax.devices()[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "kernel": options.kernel, "kernel_config": kernel_config,
        "initial_roots": 150, "replacement_width": 150, "delta_K": 150,
        "sigma_stiff": 1e-5, "curvature": 0.3, "unit_dtype": "float64", "phantoms": 0,
        "allocation_target": "evidence_improving", "unlimited_samples": True,
        "dlogZ": float(np.log1p(1e-3)), "target_log_Z_uncert": options.target,
        "evaluation_accounting": "includes_each_value_and_grad_once; gradients_also_separate",
    }
    for seed in options.seeds:
        archive = folder / f"seed-{seed}.npz"
        if archive.exists():
            raise FileExistsError(archive)
        started = time.monotonic()
        history = []

        def goal(current: State, seed=seed, started=started, history=history) -> bool:
            # An initial prior population is not a completed evidence epoch.
            if int(current.goal_loop_iter) == 0:
                return False
            uncertainty = float(current.expected_log_Z_uncert)
            if not math.isfinite(uncertainty):
                raise ValueError(f"Nonfinite evidence uncertainty for seed {seed}.")
            progress = {
                "kernel": options.kernel, "seed": seed, "epoch": int(current.goal_loop_iter),
                "samples": int(current.num_samples), "roots": int(current.root_out_degree),
                "evaluations": int(current.total_num_likelihood_evaluations),
                "log_Z": float(current.expected_log_Z_mean), "log_Z_uncert": uncertainty,
                "seconds": time.monotonic() - started,
            }
            history.append(progress)
            with (folder / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            return uncertainty <= options.target

        state = runner.run_until_goal(goal, key=jax.random.PRNGKey(seed))
        result = jax.block_until_ready(state.to_result())
        if not history or history[-1]["log_Z_uncert"] > options.target:
            raise RuntimeError(f"Seed {seed} returned without reaching its scientific goal.")
        if int(state.termination_reason) != 0 or not bool(state.depth_reached):
            raise RuntimeError(f"Seed {seed} returned outside a completed depth epoch.")
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        constraints = np.asarray(result.log_L_constraints)[:count]
        roots = int(state.root_out_degree)
        if roots != np.sum(np.isneginf(constraints)):
            raise RuntimeError("Root accounting does not agree with stored prior constraints.")
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = np.sum(points * weights[:, None], axis=0)
        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
        # Additional evidence epochs can allocate new prior roots; these have
        # no reflection trajectory. Count only actual non-root children.
        gradients = (count - roots) * gradients_per_child
        evaluations = int(result.total_num_likelihood_evaluations)
        np.savez_compressed(
            archive, samples=points,
            unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
            log_weights=log_weights, log_likelihood=np.asarray(result.log_L)[:count],
            log_volume=np.asarray(result.log_X_mean)[:count], log_L_constraints=constraints,
        )
        record = {
            **metadata, "seed": seed, "samples": count, "roots": roots,
            "evaluations": evaluations, "gradient_evaluations": gradients,
            "value_only_evaluations": evaluations - gradients,
            "log_Z": float(result.log_Z_mean), "log_Z_uncert": float(result.log_Z_uncert),
            "peak_gap": float(peak - result.log_L_supremum),
            **{f"std_x{i}": float(std[i]) for i in (1, 5, 16, 17)},
            "kish_ess": float(1 / np.sum(weights ** 2)),
            "goal_reached": True, "epochs": int(state.goal_loop_iter), "history": history,
            "seconds": time.monotonic() - started, "timing": "including_compilation",
            "termination_reason": int(state.termination_reason),
            "depth_reached": bool(state.depth_reached),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        with (folder / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
