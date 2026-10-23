"""Stationarity and displacement for one archived, frozen GP direction law."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--chains", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gap", type=float, default=15.0)
    options = parser.parse_args()
    if options.chains % options.batch_size or not 0 < options.gap < 50:
        parser.error("Use complete batches and an untruncated contour gap in (0, 50).")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.stats import beta, kstest

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.gp_geometry import GP
    from benchmarks.issue_312.gp_hessian_sampler import GPHessianSliceSampler

    sigma = 1e-5
    peak = -np.log(sigma * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    with np.load(options.fit) as saved:
        gp = GP(*(jnp.asarray(saved[name]) for name in GP._fields))
    sampler = GPHessianSliceSampler(gp, float(peak))
    rng = np.random.default_rng(312001)
    radius = np.sqrt(2 * options.gap)
    ball = rng.normal(size=(options.chains, 15))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    ball *= (radius * rng.uniform(size=options.chains) ** (1 / 15))[:, None]
    initial = rng.uniform(size=(options.chains, 18))
    initial[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * initial[:, 1]) + sigma * ball[:, 0]
    initial[:, 2:16] = 0.5 + 0.05 * ball[:, 1:]

    def likelihood(point):
        residual = point[0] - (0.5 + 0.3 * jnp.sin(2 * jnp.pi * point[1]))
        return peak - 0.5 * (
            (residual / sigma) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2)
        )

    def one_chain(key, point):
        final, _ = sampler.run_chain(
            key, point, likelihood(point), jnp.asarray(peak - options.gap), likelihood,
        )
        return final.point, final.accepted, final.evaluations

    kernel = jax.jit(jax.vmap(one_chain))
    keys = jax.random.split(jax.random.PRNGKey(312002), options.chains)
    outputs = []
    started = time.monotonic()
    for start in range(0, options.chains, options.batch_size):
        outputs.append(jax.block_until_ready(kernel(
            keys[start:start + options.batch_size],
            jnp.asarray(initial[start:start + options.batch_size]),
        )))
    points, accepted, evaluations = (
        np.concatenate([np.asarray(output[i]) for output in outputs]) for i in range(3)
    )
    gaussian = np.concatenate([
        ((points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])) / sigma)[:, None],
        (points[:, 2:16] - 0.5) / 0.05,
    ], axis=1)
    pits = np.concatenate([
        points[:, [1, 16, 17]], beta.cdf((gaussian / radius + 1) / 2, 8, 8),
        (np.linalg.norm(gaussian, axis=1) / radius)[:, None] ** 15,
    ], axis=1)
    ks = [float(kstest(column, "uniform").statistic) for column in pits.T]
    assert np.all((points >= 0) & (points <= 1))
    assert np.all(np.sum(gaussian ** 2, axis=1) < radius ** 2)
    assert max(ks) < 5 / np.sqrt(options.chains)
    record = {
        "fit_sha256": hashlib.sha256(options.fit.read_bytes()).hexdigest(),
        "chains": options.chains, "slices": 90, "gap": options.gap,
        "seconds": time.monotonic() - started,
        "max_ks": max(ks), "ks_statistics": ks, "ks_limit": 5 / np.sqrt(options.chains),
        "mh_acceptance": float(accepted.mean() / 90),
        "mean_likelihood_evaluations": float(evaluations.mean()),
        "rms_displacement": np.sqrt(np.mean((points - initial) ** 2, axis=0)).tolist(),
        "endpoint_correlation": [
            float(np.corrcoef(points[:, i], initial[:, i])[0, 1]) for i in range(18)
        ],
    }
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
