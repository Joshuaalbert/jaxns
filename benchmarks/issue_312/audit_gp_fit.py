"""Refit fixed archived populations to separate optimization from sampling."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[0, 2, 10, 20])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.gp_geometry import GP, fit_gp, negative_predictive_hessian
    from benchmarks.issue_312.hessian_sampler import negative_hessian_block

    records = [
        json.loads(line)
        for line in (options.directory / "fits.jsonl").read_text().splitlines()
    ]
    output = []
    for update in options.updates:
        with np.load(options.directory / f"fit-{update:03d}.npz") as saved:
            points, values = saved["points"], saved["values"]
            previous = None
            if update:
                with np.load(options.directory / f"fit-{update - 1:03d}.npz") as old:
                    previous = GP(*(jnp.asarray(old[name]) for name in GP._fields))
            gp, fit, indices = fit_gp(
                points, values, max_points=len(saved["coefficients"]),
                max_iterations=options.iterations, previous=previous,
            )
        selected = np.zeros(len(points), dtype=bool)
        selected[indices] = True
        audit = points[~selected] if np.any(~selected) else points
        audit = audit[:64]
        predicted = np.asarray(jax.jit(jax.vmap(
            negative_predictive_hessian, in_axes=(None, 0),
        ))(gp, jnp.asarray(audit)))
        truth = np.broadcast_to(np.diag([0., 0.] + [400.] * 14 + [0., 0.]), predicted.shape).copy()
        truth[:, :2, :2] = jax.vmap(negative_hessian_block, in_axes=(0, None, None))(
            jnp.asarray(audit), 1e-5, 0.3,
        )
        le, lv = np.linalg.eigh(predicted)
        te, tv = np.linalg.eigh(truth)
        row = {
            "update": update, "log_X": records[update]["log_X"],
            "max_iterations": options.iterations,
            "baseline": {name: records[update][name] for name in (
                "negative_log_marginal_likelihood", "held_out_rmse", "stiff_alignment",
                "stiff_eigenvalue_ratio", "hessian_relative_error", "iterations",
            )},
            "refit": {
                **fit,
                "hessian_relative_error": float(np.median(
                    np.linalg.norm(predicted - truth, axis=(1, 2))
                    / np.linalg.norm(truth, axis=(1, 2)),
                )),
                "stiff_alignment": float(np.median(np.abs(np.sum(
                    lv[:, :, -1] * tv[:, :, -1], axis=1,
                )))),
                "stiff_eigenvalue_ratio": float(np.median(le[:, -1] / te[:, -1])),
            },
        }
        output.append(row)
        print(json.dumps(row), flush=True)
    options.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
