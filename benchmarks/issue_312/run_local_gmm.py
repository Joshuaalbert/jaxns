"""Run the ridge with a 100-iteration local Gaussian-mixture direction fit."""

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
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--fit-iterations", type=int, default=100)
    parser.add_argument("--refresh-efolds", type=float, default=1.0)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--regularisation", type=float, default=1e-6)
    parser.add_argument("--iso-prob", type=float, default=0.01)
    parser.add_argument("--selection", choices=("local", "global"), default="local")
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    if (not 1 <= options.components <= 150 or options.fit_iterations < 1
            or options.refresh_efolds <= 0 or options.regularisation <= 0
            or options.slices < 1 or not 0 <= options.iso_prob <= 1):
        parser.error("Invalid GMM fitting budget or refresh interval.")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jaxctx
    import numpy as np
    from jax import numpy as jnp
    from scipy.special import logsumexp

    jax.config.update("jax_enable_x64", True)
    import jaxns.model as model_module
    from benchmarks.issue_312.gp_refresh import run_with_refresh
    from benchmarks.issue_312.local_gmm_sampler import (
        LocalGMMSliceSampler,
        factor_mixture,
    )
    from benchmarks.issue_312.run_ridge import prior_model
    from jaxns.constrained_sampler import UniDimSliceSampler
    from jaxns.core import NestedSampler
    from jaxns.model import Model
    from jaxns.sampling.gmm import e_step, fit_gmm

    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    model = Model(prior_model=partial(prior_model, 1e-5, 0.3))
    runner = NestedSampler(
        model=model, root_allocation_degree=150,
        sampler=UniDimSliceSampler(num_slices=options.slices),
    )
    compiled_fit = jax.jit(fit_gmm, static_argnames=("n_components", "n_iters", "regularisation"))
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    root = options.output_dir
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).resolve().parent
    source_paths = {name: folder / name for name in (
        "local_gmm_sampler.py", "run_local_gmm.py", "gp_refresh.py", "run_ridge.py",
    )}
    source_paths["src/jaxns/sampling/gmm.py"] = folder.parents[1] / "src/jaxns/sampling/gmm.py"
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in source_paths.items()},
        "jax": jax.__version__, "python": platform.python_version(),
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": f"gmm_{options.selection}", "roots": 150, "slices": options.slices,
        "sigma_stiff": 1e-5, "curvature": 0.3, "unit_dtype": "float64",
        "components": options.components, "fit_iterations": options.fit_iterations,
        "refresh_efolds": options.refresh_efolds, "regularisation": options.regularisation,
        "iso_prob": options.iso_prob, "warm_start": options.warm_start, "correct_mh": True,
        "fit_population": "all_stored_classics_above_contour_unweighted",
        "allocation_target": "uniform", "phantoms": 0, "dlogZ": float(np.log1p(1e-3)),
        "training_evaluations": 0,
    }
    for seed in options.seeds:
        directory = root / f"seed-{seed}"
        directory.mkdir()
        started = time.monotonic()
        state = runner.initialise(jax.random.PRNGKey(seed))
        records = []
        previous = None

        def refresh(current, contour, log_volume, seed=seed, directory=directory, records=records):
            nonlocal previous
            fit_started = time.monotonic()
            count = int(current.num_samples)
            points = np.asarray(jax.tree.leaves(current.samples.U_samples)[0])[:count]
            values = np.asarray(current.samples.log_likelihoods)[:count]
            eligible = values > contour
            points, values = points[eligible], values[eligible]
            # Fit every eligible classic with equal weight. Power-of-two
            # padding only bounds compilation shapes; masked rows never enter
            # the initializer, EM moments or installed direction geometry.
            capacity = max(256, 1 << (len(points) - 1).bit_length())
            padded = np.zeros((capacity, 18), dtype=np.float64)
            padded[:len(points)] = points
            mask = np.arange(capacity) < len(points)
            mixture, _, _ = jax.block_until_ready(compiled_fit(
                jax.random.PRNGKey(31227), jnp.asarray(padded), options.components,
                mask=jnp.asarray(mask), initial=previous if options.warm_start else None,
                n_iters=options.fit_iterations, regularisation=options.regularisation,
            ))
            previous = mixture
            geometry = factor_mixture(mixture)
            responsibilities = np.exp(np.asarray(e_step(
                jnp.asarray(points), mixture.centres, mixture.covariances,
                mixture.log_masses, None,
            )))
            effective_members = responsibilities.sum(axis=1) ** 2 / np.sum(
                responsibilities ** 2, axis=1,
            ).clip(min=1e-300)
            record = {
                "seed": seed, "update": len(records), "samples": count,
                "contour": contour if np.isfinite(contour) else None, "log_X": log_volume,
                "population": len(points), "capacity": capacity,
                "valid_components": int(np.sum(mixture.valid)),
                "component_masses": np.exp(np.asarray(mixture.log_masses)).tolist(),
                "effective_members": effective_members.tolist(),
                "minimum_eigenvalues": np.linalg.eigvalsh(mixture.covariances)[:, 0].tolist(),
                "fit_seconds": time.monotonic() - fit_started,
            }
            records.append(record)
            np.savez_compressed(
                directory / f"fit-{len(records) - 1:03d}.npz",
                centres=mixture.centres, covariances=mixture.covariances,
                log_masses=mixture.log_masses, valid=mixture.valid, points=points, values=values,
            )
            with (directory / "fits.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({key: record[key] for key in (
                "seed", "update", "log_X", "population", "fit_seconds",
            )}), flush=True)
            return LocalGMMSliceSampler(
                geometry, num_slices=options.slices, iso_prob=options.iso_prob,
                local_selection=options.selection == "local",
            )

        sampler = refresh(state, -np.inf, 0.0)
        state = run_with_refresh(runner, state, sampler, refresh, options.refresh_efolds)
        result = jax.block_until_ready(state.to_result())
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - logsumexp(log_weights))
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
        record = {
            **metadata, "seed": seed, "samples": count,
            "evaluations": int(result.total_num_likelihood_evaluations),
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
