"""Run the ridge with true-gradient reflections and ordinary isotropic slices."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=9)
    parser.add_argument("--slices", type=int, default=10)
    parser.add_argument("--bounces", type=int, default=50)
    parser.add_argument("--step", type=float, default=0.003)
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    model = Model(prior_model=partial(prior_model, 1e-5, options.curvature))
    sampler = ReflectionSampler(
        num_cycles=options.cycles, slices_per_cycle=options.slices,
        num_bounces=options.bounces, step_size=options.step, boundary_only=options.boundary_only,
    )
    runner = NestedSampler(model=model, root_allocation_degree=150, sampler=sampler)
    root = options.output_dir
    root.mkdir(parents=True, exist_ok=True)
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("reflection_sampler.py", "run_reflection.py", "run_ridge.py")
        },
        "jax": jax.__version__, "python": platform.python_version(),
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": "galilean" if options.boundary_only else "hug",
        "roots": 150, "cycles": options.cycles, "slices_per_cycle": options.slices,
        "bounces": options.bounces, "step_size": options.step,
        "curvature": options.curvature, "sigma_stiff": 1e-5, "unit_dtype": "float64",
        "allocation_target": "uniform", "phantoms": 0, "dlogZ": float(np.log1p(1e-3)),
        "gradient_source": "jax.value_and_grad_of_true_model_log_likelihood",
        "evaluation_accounting": "includes_each_value_and_grad_once; gradients_also_separate",
    }
    for seed in options.seeds:
        archive = root / f"seed-{seed}.npz"
        if archive.exists():
            raise FileExistsError(archive)
        started = time.monotonic()
        state = runner.run(key=jax.random.PRNGKey(seed))
        result = jax.block_until_ready(state.to_result())
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = np.sum(points * weights[:, None], axis=0)
        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
        gradients = (count - 150) * options.cycles * options.bounces
        evaluations = int(result.total_num_likelihood_evaluations)
        np.savez_compressed(
            archive, samples=points,
            unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
            log_weights=log_weights, log_likelihood=np.asarray(result.log_L)[:count],
            log_volume=np.asarray(result.log_X_mean)[:count],
            log_L_constraints=np.asarray(result.log_L_constraints)[:count],
        )
        record = {
            **metadata, "seed": seed, "samples": count, "evaluations": evaluations,
            "gradient_evaluations": gradients, "value_only_evaluations": evaluations - gradients,
            "log_Z": float(result.log_Z_mean), "log_Z_uncert": float(result.log_Z_uncert),
            "peak_gap": float(peak - result.log_L_supremum),
            **{f"std_x{i}": float(std[i]) for i in (1, 5, 16, 17)},
            "seconds": time.monotonic() - started, "timing": "including_compilation",
            "termination_reason": int(state.termination_reason),
            "depth_reached": bool(state.depth_reached), "epochs": int(state.goal_loop_iter),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        with (root / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
