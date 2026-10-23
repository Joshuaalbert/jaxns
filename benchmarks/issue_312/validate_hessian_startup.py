"""Measure straight-ridge movement on a broad early contour."""

import hashlib
import json
import os
from pathlib import Path


def main():
    os.sched_setaffinity(0, {3})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler

    count, steps, gap = 8192, 90, 2e8
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    contour = peak - gap

    def likelihood(point):
        return peak - 0.5 * (
            ((point[0] - 0.5) / 1e-5) ** 2
            + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2)
        )

    # The exact constrained set lies inside x0 in [0.3,0.7] times the other
    # unit intervals. Rejection from this rectangle gives independent seeds;
    # we do not assume the tiny Gaussian-coordinate edge correction is zero.
    rng = np.random.default_rng(312003)
    chunks, collected = [], 0
    while collected < count:
        proposed = rng.uniform(size=(count - collected, 18))
        proposed[:, 0] = 0.3 + 0.4 * proposed[:, 0]
        selected = np.asarray(jax.vmap(likelihood)(proposed)) > contour
        chunks.append(proposed[selected])
        collected += int(selected.sum())
    initial = np.concatenate(chunks)
    output = Path(__file__).with_name("raw") / "hessian-startup-movement.jsonl"
    with output.open("w") as stream:
        for scaled in (False, True):
            sampler = ToyHessianSliceSampler(
                num_slices=steps, curvature=0.0, scale_to_contour=scaled,
            )

            def one(key, point, sampler=sampler):
                final, _ = sampler.run_chain(
                    key, point, likelihood(point), jnp.asarray(contour), likelihood,
                )
                return final.point, final.accepted, final.evaluations

            points, accepted, evaluations = jax.block_until_ready(jax.jit(jax.vmap(one))(
                jax.random.split(jax.random.PRNGKey(312004), count),
                jnp.asarray(initial),
            ))
            points = np.asarray(points)
            assert np.all(np.asarray(jax.vmap(likelihood)(points)) > contour)
            assert np.all((points >= 0) & (points <= 1))
            assert np.all(np.asarray(accepted) == steps)
            record = {
                "contour_scaled": scaled, "gap": gap,
                "chains": count, "slices": steps,
                "rms_x0_displacement": float(np.sqrt(np.mean(
                    (points[:, 0] - initial[:, 0]) ** 2,
                ))),
                "x0_endpoint_correlation": float(
                    np.corrcoef(points[:, 0], initial[:, 0])[0, 1],
                ),
                "initial_x0_sd": float(np.std(initial[:, 0])),
                "final_x0_sd": float(np.std(points[:, 0])),
                "calls_per_slice": float(np.mean(evaluations) / steps),
                "acceptance_fraction": float(np.mean(accepted) / steps),
                "sampler_sha256": hashlib.sha256(
                    Path(__file__).with_name("hessian_sampler.py").read_bytes(),
                ).hexdigest(),
            }
            stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
