"""Inspect all learned axes, not only the dominant Hessian eigenvector."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=5)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.hessian_sampler import negative_hessian_block
    from benchmarks.issue_312.nn_geometry import NN, negative_predictive_hessian

    @jax.jit
    def inspect(network, points, scale):
        learned = jax.vmap(negative_predictive_hessian, in_axes=(None, 0))(network, points)
        block = jax.vmap(negative_hessian_block, in_axes=(0, None, None))(points, 1e-5, 0.3)
        true = jnp.broadcast_to(jnp.diag(jnp.array([0., 0.] + [400.] * 14 + [0., 0.])),
                                learned.shape).at[:, :2, :2].set(block)
        eigenvalues, rotation = jnp.linalg.eigh(learned)
        true_values, true_rotation = jnp.linalg.eigh(true)
        precision = jnp.maximum(eigenvalues / scale, 12.0)
        true_precision = jnp.maximum(true_values / scale, 12.0)
        covariance_diagonal = jnp.sum(rotation ** 2 / precision[:, None, :], axis=2)
        true_covariance_diagonal = jnp.sum(
            true_rotation ** 2 / true_precision[:, None, :], axis=2,
        )
        # Ratios to the variance in the true stiff direction remove any
        # irrelevant common scale: directions are normalized before slicing.
        stiff_projection = jnp.einsum("nij,ni->nj", rotation, true_rotation[:, :, -1])
        stiff_variance = jnp.sum(stiff_projection ** 2 / precision, axis=1)
        relative_to_stiff = covariance_diagonal / stiff_variance[:, None]
        true_relative_to_stiff = true_covariance_diagonal * true_precision[:, -1, None]
        return {
            "median_hessian_diagonal": jnp.median(jnp.diagonal(learned, axis1=1, axis2=2), axis=0),
            "median_gaussian_direction_sd": jnp.median(jnp.sqrt(covariance_diagonal), axis=0),
            "median_oracle_direction_sd": jnp.median(jnp.sqrt(true_covariance_diagonal), axis=0),
            "median_axis_to_stiff_variance_ratio": jnp.median(relative_to_stiff, axis=0),
            "median_oracle_axis_to_stiff_variance_ratio": jnp.median(
                true_relative_to_stiff, axis=0,
            ),
        }

    rows = []
    for directory in options.directories:
        records = [json.loads(line) for line in (directory / "fits.jsonl").read_text().splitlines()]
        for record in records:
            if record["contour"] is None:
                continue
            path = directory / f"fit-{record['update']:03d}.npz"
            with np.load(path) as saved:
                network = NN(*(jnp.asarray(saved[name]) for name in NN._fields))
                mask = np.ones(len(saved["points"]), dtype=bool)
                mask[saved["centres"]] = False
                points = saved["points"][mask] if np.any(mask) else saved["points"]
            scale = 2 * (float(network.peak) - record["contour"]) / 17
            diagnostics = inspect(network, jnp.asarray(points[:64]), jnp.asarray(scale))
            rows.append({
                "seed": record["seed"], "update": record["update"], "log_X": record["log_X"],
                "contour": record["contour"], "directory": str(directory),
                **{name: np.asarray(value).tolist() for name, value in diagnostics.items()},
            })
    options.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"fit_audits": len(rows), "output": str(options.output)}), flush=True)


if __name__ == "__main__":
    main()
