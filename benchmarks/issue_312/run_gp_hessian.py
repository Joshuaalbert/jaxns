"""Ridge comparison using a GP refit once per expected-volume e-fold."""

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
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--training-points", type=int, default=128)
    parser.add_argument("--fit-iterations", type=int, default=100)
    parser.add_argument("--refresh-efolds", type=float, default=1.0)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.gp_geometry import fit_gp, negative_predictive_hessian
    from benchmarks.issue_312.gp_hessian_sampler import GPHessianSliceSampler
    from benchmarks.issue_312.gp_refresh import run_with_refresh
    from benchmarks.issue_312.hessian_sampler import negative_hessian_block
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.constrained_sampler import UniDimSliceSampler
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    sigma = 1e-5
    peak = -np.log(sigma * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    model = Model(prior_model=partial(prior_model, sigma, options.curvature))
    # Initialisation resolves the model and shape with an ordinary sampler.
    # All constrained chains then receive the explicitly fitted GP sampler.
    runner = NestedSampler(
        model=model, root_allocation_degree=150,
        sampler=UniDimSliceSampler(num_slices=options.slices),
    )
    root = options.output_dir
    root.mkdir(parents=True, exist_ok=True)
    source_names = ["gp_geometry.py", "gp_hessian_sampler.py", "gp_refresh.py", "run_gp_hessian.py"]
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in source_names
        },
        "jax": jax.__version__, "python": platform.python_version(),
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": "gp_scaled_hessian", "roots": 150, "slices": options.slices,
        "curvature": options.curvature, "sigma_stiff": sigma, "unit_dtype": "float64",
        "training_points": options.training_points, "fit_iterations": options.fit_iterations,
        "refresh_efolds": options.refresh_efolds, "allocation_target": "uniform",
        "warm_start": options.warm_start,
        "phantoms": 0, "dlogZ": float(np.log1p(1e-3)),
        "oracle_scale": {"peak": float(peak), "quadratic_dimension": 15},
    }

    @jax.jit
    def curvature_diagnostics(gp, points):
        learned = jax.vmap(negative_predictive_hessian, in_axes=(None, 0))(gp, points)
        block = jax.vmap(negative_hessian_block, in_axes=(0, None, None))(
            points, sigma, options.curvature,
        )
        true = jnp.broadcast_to(jnp.diag(jnp.array([0., 0.] + [400.] * 14 + [0., 0.])),
                                learned.shape).at[:, :2, :2].set(block)
        le, lv = jnp.linalg.eigh(learned)
        te, tv = jnp.linalg.eigh(true)
        alignment = jnp.abs(jnp.sum(lv[:, :, -1] * tv[:, :, -1], axis=1))
        return {
            "hessian_relative_error": jnp.median(
                jnp.linalg.norm(learned - true, axis=(1, 2)) / jnp.linalg.norm(true, axis=(1, 2)),
            ),
            "stiff_alignment": jnp.median(alignment),
            "stiff_eigenvalue_ratio": jnp.median(le[:, -1] / te[:, -1]),
        }

    for seed in options.seeds:
        directory = root / f"seed-{seed}"
        directory.mkdir()
        started = time.monotonic()
        state = runner.initialise(jax.random.PRNGKey(seed))
        records = []
        previous_gp = None

        def refresh(current, contour, log_volume, seed=seed, records=records, directory=directory):
            nonlocal previous_gp
            fit_started = time.monotonic()
            count = int(current.num_samples)
            points = np.asarray(jax.tree.leaves(current.samples.U_samples)[0])[:count]
            values = np.asarray(current.samples.log_likelihoods)[:count]
            eligible = values > contour
            points, values = points[eligible], values[eligible]
            gp, fit_record, indices = fit_gp(
                points, values, max_points=options.training_points,
                max_iterations=options.fit_iterations,
                previous=previous_gp if options.warm_start else None,
            )
            previous_gp = gp
            selected = np.zeros(len(points), dtype=bool)
            selected[indices] = True
            audit = points[~selected] if np.any(~selected) else points
            # This oracle audit is observational only and never changes the fit.
            diagnostics = curvature_diagnostics(gp, jnp.asarray(audit[:64]))
            record = {
                "seed": seed, "update": len(records), "samples": count,
                "contour": contour if np.isfinite(contour) else None,
                "log_X": log_volume, "fit_seconds": time.monotonic() - fit_started,
                **fit_record, **{name: float(value) for name, value in diagnostics.items()},
            }
            records.append(record)
            np.savez_compressed(
                directory / f"fit-{len(records) - 1:03d}.npz",
                **gp._asdict(), points=points, values=values, training_indices=indices,
            )
            with (directory / "fits.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({key: record[key] for key in (
                "seed", "update", "log_X", "population", "held_out_rmse",
                "stiff_alignment", "stiff_eigenvalue_ratio", "fit_seconds",
            )}), flush=True)
            return GPHessianSliceSampler(gp, float(peak), num_slices=options.slices)

        sampler = refresh(state, -np.inf, 0.0)
        state = run_with_refresh(runner, state, sampler, refresh, options.refresh_efolds)
        result = jax.block_until_ready(state.to_result())
        elapsed = time.monotonic() - started
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = np.sum(points * weights[:, None], axis=0)
        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
        np.savez_compressed(
            directory / "samples.npz", samples=points,
            unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
            log_weights=log_weights, log_likelihood=np.asarray(result.log_L)[:count],
            log_volume=np.asarray(result.log_X_mean)[:count],
            log_L_constraints=np.asarray(result.log_L_constraints)[:count],
        )
        record = {
            **metadata, "seed": seed, "samples": count,
            "evaluations": int(result.total_num_likelihood_evaluations),
            "log_Z": float(result.log_Z_mean), "log_Z_uncert": float(result.log_Z_uncert),
            "peak_gap": float(peak - result.log_L_supremum),
            **{f"std_x{i}": float(std[i]) for i in (1, 5, 16, 17)},
            "seconds": elapsed, "timing": "including_compilation_and_fits",
            "fit_seconds": sum(record["fit_seconds"] for record in records),
            "geometry_updates": len(records),
            "termination_reason": int(state.termination_reason),
            "depth_reached": bool(state.depth_reached), "epochs": int(state.goal_loop_iter),
        }
        with (root / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
