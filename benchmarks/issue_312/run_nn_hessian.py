"""Run the ridge using an NN Hessian refitted at expected-volume e-folds."""

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
    parser.add_argument("--representatives", type=int, default=128)
    parser.add_argument("--fit-iterations", type=int, default=1000)
    parser.add_argument("--refresh-efolds", type=float, default=1.0)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--relative-loss", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if options.representatives < 2 or options.fit_iterations < 1 or options.refresh_efolds <= 0:
        parser.error("Representative count, fit budget and refresh interval must be positive.")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.gp_geometry import representative_indices
    from benchmarks.issue_312.gp_refresh import run_with_refresh
    from benchmarks.issue_312.hessian_sampler import negative_hessian_block
    from benchmarks.issue_312.nn_geometry import (
        fit_nn,
        negative_predictive_hessian,
        predictive_mean,
    )
    from benchmarks.issue_312.nn_hessian_sampler import NNHessianSliceSampler
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.constrained_sampler import UniDimSliceSampler
    from jaxns.core import NestedSampler
    from jaxns.model import Model
    from jaxns.pytree import pytree_ravel

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    model = Model(prior_model=partial(prior_model, 1e-5, 0.3))
    runner = NestedSampler(
        model=model, root_allocation_degree=150,
        sampler=UniDimSliceSampler(num_slices=options.slices),
    )
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    root = options.output_dir
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("nn_geometry.py", "nn_hessian_sampler.py", "hessian_chain.py",
                         "gp_refresh.py", "gp_geometry.py", "run_nn_hessian.py", "run_ridge.py")
        },
        "jax": jax.__version__, "python": platform.python_version(),
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": "nn_scaled_hessian", "roots": 150, "slices": options.slices,
        "sigma_stiff": 1e-5, "curvature": 0.3, "unit_dtype": "float64",
        "representatives": options.representatives, "fit_iterations": options.fit_iterations,
        "refresh_efolds": options.refresh_efolds, "augment": options.augment,
        "relative_loss": options.relative_loss, "warm_start": True,
        "allocation_target": "uniform", "phantoms": 0, "dlogZ": float(np.log1p(1e-3)),
        "oracle_scale": {"peak": float(peak), "quadratic_dimension": 15},
        "evaluation_accounting": "evaluations = sampling_evaluations + training_evaluations",
    }

    @jax.jit
    def curvature_diagnostics(network, points):
        learned = jax.vmap(negative_predictive_hessian, in_axes=(None, 0))(network, points)
        block = jax.vmap(negative_hessian_block, in_axes=(0, None, None))(points, 1e-5, 0.3)
        true = jnp.broadcast_to(jnp.diag(jnp.array([0., 0.] + [400.] * 14 + [0., 0.])),
                                learned.shape).at[:, :2, :2].set(block)
        le, lv = jnp.linalg.eigh(learned)
        te, tv = jnp.linalg.eigh(true)
        return {
            "hessian_relative_error": jnp.median(
                jnp.linalg.norm(learned - true, axis=(1, 2)) / jnp.linalg.norm(true, axis=(1, 2)),
            ),
            "stiff_alignment": jnp.median(jnp.abs(jnp.sum(lv[:, :, -1] * tv[:, :, -1], axis=1))),
            "stiff_eigenvalue_ratio": jnp.median(le[:, -1] / te[:, -1]),
        }

    for seed in options.seeds:
        directory = root / f"seed-{seed}"
        directory.mkdir()
        started = time.monotonic()
        state = runner.initialise(jax.random.PRNGKey(seed))
        _, unravel = pytree_ravel(jax.tree.map(lambda value: value[0], state.samples.U_samples))

        @jax.jit
        @jax.vmap
        def evaluate_probe(point, unravel=unravel):
            return model.log_likelihood(unravel(point))

        # Training randomness is separate from the runner's stream. Probe
        # observations fit geometry only; they never enter the race tree.
        rng = np.random.default_rng(np.random.SeedSequence([31205, seed]))
        records = []
        previous = None

        def refresh(
            current, contour, log_volume, seed=seed, records=records, directory=directory,
            rng=rng, evaluate_probe=evaluate_probe,
        ):
            nonlocal previous
            fit_started = time.monotonic()
            count = int(current.num_samples)
            points = np.asarray(jax.tree.leaves(current.samples.U_samples)[0])[:count]
            values = np.asarray(current.samples.log_likelihoods)[:count]
            eligible = values > contour
            points, values = points[eligible], values[eligible]
            centred = points - points.mean(axis=0)
            covariance = centred.T @ centred / (len(points) - 1)
            variance = max(float(np.trace(covariance) / 18), 1e-15)
            cholesky = np.linalg.cholesky(covariance + 1e-8 * variance * np.eye(18))
            whitened = np.linalg.solve(cholesky, centred.T).T
            centres = representative_indices(whitened, options.representatives)
            training_points, training_values = points, values
            probes = np.empty((0, 18))
            if options.augment:
                directions = rng.normal(size=(len(centres), 4, 18))
                directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
                offset = directions * np.array([1e-5, 1e-4, 1e-3, 1e-2])[None, :, None]
                probes = np.concatenate([
                    (points[centres, None] + offset).reshape(-1, 18),
                    (points[centres, None] - offset).reshape(-1, 18),
                ])
                probes = 1 - np.abs(1 - np.mod(probes, 2))
                training_points = np.concatenate([points[centres], probes])
                training_values = np.concatenate([
                    values[centres], np.asarray(evaluate_probe(jnp.asarray(probes))),
                ])
            network, fit_record, indices = fit_nn(
                training_points, training_values, peak=float(peak),
                max_points=len(training_points) if options.augment else options.representatives,
                max_iterations=options.fit_iterations, previous=previous,
                relative_loss=options.relative_loss,
            )
            previous = network
            held_out = np.ones(len(points), dtype=bool)
            held_out[centres] = False
            audit = points[held_out] if np.any(held_out) else points
            audit_values = values[held_out] if np.any(held_out) else values
            predictions = np.asarray(jax.jit(jax.vmap(predictive_mean, in_axes=(None, 0)))(
                network, jnp.asarray(audit),
            ))
            # True Hessians are observational audit data only. They do not
            # select a fit, set a direction or change any stopping decision.
            diagnostics = curvature_diagnostics(network, jnp.asarray(audit[:64]))
            record = {
                "seed": seed, "update": len(records), "samples": count,
                "contour": contour if np.isfinite(contour) else None, "log_X": log_volume,
                **fit_record, "contour_population": len(points),
                "population_held_out_count": int(np.sum(held_out)),
                "population_audit_rmse": float(np.sqrt(np.mean((predictions - audit_values) ** 2))),
                "training_evaluations": len(probes),
                **{name: float(value) for name, value in diagnostics.items()},
                "fit_seconds": time.monotonic() - fit_started,
            }
            records.append(record)
            np.savez_compressed(
                directory / f"fit-{len(records) - 1:03d}.npz", **network._asdict(),
                points=points, values=values, centres=centres,
                training_points=training_points, training_values=training_values,
                training_indices=indices,
            )
            with (directory / "fits.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({key: record[key] for key in (
                "seed", "update", "log_X", "contour_population", "population_audit_rmse",
                "stiff_alignment", "stiff_eigenvalue_ratio", "fit_seconds",
            )}), flush=True)
            return NNHessianSliceSampler(network, float(peak), num_slices=options.slices)

        sampler = refresh(state, -np.inf, 0.0)
        state = run_with_refresh(runner, state, sampler, refresh, options.refresh_efolds)
        result = jax.block_until_ready(state.to_result())
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = np.sum(points * weights[:, None], axis=0)
        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
        archive = directory / "samples.npz"
        np.savez_compressed(
            archive, samples=points,
            unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
            log_weights=log_weights, log_likelihood=np.asarray(result.log_L)[:count],
            log_volume=np.asarray(result.log_X_mean)[:count],
            log_L_constraints=np.asarray(result.log_L_constraints)[:count],
        )
        sampling_evaluations = int(result.total_num_likelihood_evaluations)
        training_evaluations = sum(record["training_evaluations"] for record in records)
        record = {
            **metadata, "seed": seed, "samples": count,
            "sampling_evaluations": sampling_evaluations,
            "training_evaluations": training_evaluations,
            "evaluations": sampling_evaluations + training_evaluations,
            "chain_hessian_evaluations": (count - 150) * (options.slices + 1),
            "log_Z": float(result.log_Z_mean), "log_Z_uncert": float(result.log_Z_uncert),
            "peak_gap": float(peak - result.log_L_supremum),
            **{f"std_x{i}": float(std[i]) for i in (1, 5, 16, 17)},
            "seconds": time.monotonic() - started, "timing": "including_compilation_fits_audits",
            "fit_seconds": sum(record["fit_seconds"] for record in records),
            "geometry_updates": len(records), "termination_reason": int(state.termination_reason),
            "depth_reached": bool(state.depth_reached), "epochs": int(state.goal_loop_iter),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        with (root / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
