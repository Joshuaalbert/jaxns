"""Incrementally audit NN representations on the archived GP contour populations.

This is a surrogate-fit experiment, not a nested-sampling run. A second arm
adds explicitly counted true value observations around the stored points to
test whether learning normal curvature needs off-contour information.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--relative-loss", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.hessian_sampler import negative_hessian_block
    from benchmarks.issue_312.nn_geometry import (
        fit_nn,
        negative_predictive_hessian,
        predictive_mean,
    )

    root = options.output_dir
    root.mkdir(parents=True, exist_ok=True)
    previous = None
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))

    def true_values(points):
        residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
        return peak - 0.5 * (
            (residual / 1e-5) ** 2 + np.sum(((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
        )

    @jax.jit
    def diagnostics(network, points):
        learned = jax.vmap(negative_predictive_hessian, in_axes=(None, 0))(network, points)
        blocks = jax.vmap(negative_hessian_block, in_axes=(0, None, None))(points, 1e-5, 0.3)
        true = jnp.broadcast_to(jnp.diag(jnp.array([0., 0.] + [400.] * 14 + [0., 0.])),
                                learned.shape).at[:, :2, :2].set(blocks)
        le, lv = jnp.linalg.eigh(learned)
        te, tv = jnp.linalg.eigh(true)
        return {
            "hessian_relative_error": jnp.median(
                jnp.linalg.norm(learned - true, axis=(1, 2)) / jnp.linalg.norm(true, axis=(1, 2)),
            ),
            "stiff_alignment": jnp.median(jnp.abs(jnp.sum(lv[:, :, -1] * tv[:, :, -1], axis=1))),
            "stiff_eigenvalue_ratio": jnp.median(le[:, -1] / te[:, -1]),
        }

    # The same independent exact-contour set audits global generalisation for
    # every fit; it never enters selection, fitting or an optimiser objective.
    rng = np.random.default_rng(31204)
    exact = rng.uniform(size=(256, 18))
    ball = rng.normal(size=(256, 15))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    ball *= (np.sqrt(30) * rng.uniform(size=256) ** (1 / 15))[:, None]
    exact[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * exact[:, 1]) + 1e-5 * ball[:, 0]
    exact[:, 2:16] = 0.5 + 0.05 * ball[:, 1:]
    source = Path(__file__).with_name("gp_audit") / "seed-0"
    for update in (0, 1, 2, 9, 10, 19, 20):
        started = time.monotonic()
        archive = np.load(source / f"fit-{update:03d}.npz")
        points, values = archive["points"], archive["values"]
        selected = archive["training_indices"]
        training_points, training_values = points, values
        new_evaluations = 0
        if options.augment:
            centres = points[selected]
            # Symmetric, isotropic physical probes do not use a known stiff
            # axis or the true gradient. They deliberately query outside the
            # thin fitting population. All added true evaluations are counted.
            directions = rng.normal(size=(len(centres), 4, 18))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            offset = directions * np.array([1e-5, 1e-4, 1e-3, 1e-2])[None, :, None]
            probes = np.concatenate([
                (centres[:, None] + offset).reshape(-1, 18),
                (centres[:, None] - offset).reshape(-1, 18),
            ])
            probes = 1 - np.abs(1 - np.mod(probes, 2))
            training_points = np.concatenate([centres, probes])
            training_values = np.concatenate([values[selected], true_values(probes)])
            new_evaluations = len(probes)
        network, fit_record, indices = fit_nn(
            training_points, training_values, peak=peak,
            max_points=len(training_points) if options.augment else 128,
            max_iterations=options.iterations, previous=previous,
            relative_loss=options.relative_loss,
        )
        previous = network
        held_out = np.ones(len(points), dtype=bool)
        held_out[selected] = False
        local = points[held_out] if np.any(held_out) else points
        local_values = true_values(local)
        predicted = np.asarray(jax.jit(jax.vmap(predictive_mean, in_axes=(None, 0)))(
            network, jnp.asarray(local),
        ))
        local_diagnostics = diagnostics(network, jnp.asarray(local[:64]))
        global_diagnostics = diagnostics(network, jnp.asarray(exact))
        global_values = np.asarray(jax.jit(jax.vmap(predictive_mean, in_axes=(None, 0)))(
            network, jnp.asarray(exact),
        ))
        record = {
            "update": update, "augment": options.augment, "max_iterations": options.iterations,
            "new_true_evaluations": new_evaluations, **fit_record,
            "population_held_out_count": int(np.sum(held_out)),
            "population_audit_rmse": float(np.sqrt(np.mean((predicted - local_values) ** 2))),
            "global_audit_rmse": float(np.sqrt(np.mean((global_values - true_values(exact)) ** 2))),
            "local": {name: float(value) for name, value in local_diagnostics.items()},
            "global": {name: float(value) for name, value in global_diagnostics.items()},
            "seconds": time.monotonic() - started,
            "source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("nn_geometry.py", "audit_nn_fit.py")
            },
        }
        np.savez_compressed(
            root / f"fit-{update:03d}.npz", **network._asdict(),
            points=training_points, values=training_values, training_indices=indices,
        )
        with (root / "fits.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
