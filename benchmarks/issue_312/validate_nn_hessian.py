"""Stationarity and movement of one frozen NN kernel on exact ridge draws."""

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
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--chains", type=int, default=8192)
    parser.add_argument("--gap", type=float, default=15.0)
    options = parser.parse_args()
    if options.chains % 256 or not 0 < options.gap < 50:
        parser.error("Use a multiple of 256 chains and an interior contour gap in (0, 50).")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.stats import beta, kstest

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.nn_geometry import NN
    from benchmarks.issue_312.nn_hessian_sampler import NNHessianSliceSampler

    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    with np.load(options.fit) as saved:
        network = NN(*(jnp.asarray(saved[name]) for name in NN._fields))
    sampler = NNHessianSliceSampler(network, float(peak))
    rng = np.random.default_rng(312001)
    radius = np.sqrt(2 * options.gap)
    ball = rng.normal(size=(options.chains, 15))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    ball *= (radius * rng.uniform(size=options.chains) ** (1 / 15))[:, None]
    initial = rng.uniform(size=(options.chains, 18))
    initial[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * initial[:, 1]) + 1e-5 * ball[:, 0]
    initial[:, 2:16] = 0.5 + 0.05 * ball[:, 1:]

    def likelihood(point):
        residual = point[0] - 0.5 - 0.3 * jnp.sin(2 * jnp.pi * point[1])
        return peak - 0.5 * ((residual / 1e-5) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2))

    @jax.jit
    @jax.vmap
    def chain(key, point):
        final, _ = sampler.run_chain(
            key, point, likelihood(point), jnp.asarray(peak - options.gap), likelihood,
        )
        return final.point, final.accepted, final.evaluations

    keys = jax.random.split(jax.random.PRNGKey(312002), options.chains)
    outputs = []
    started = time.monotonic()
    for start in range(0, options.chains, 256):
        outputs.append(jax.block_until_ready(chain(
            keys[start:start + 256], jnp.asarray(initial[start:start + 256]),
        )))
    points, accepted, evaluations = (
        np.concatenate([np.asarray(output[i]) for output in outputs]) for i in range(3)
    )
    gaussian = np.column_stack([
        (points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5,
        (points[:, 2:16] - 0.5) / 0.05,
    ])
    pits = np.column_stack([
        points[:, [1, 16, 17]], beta.cdf((gaussian / radius + 1) / 2, 8, 8),
        (np.linalg.norm(gaussian, axis=1) / radius) ** 15,
    ])
    statistics = [float(kstest(column, "uniform").statistic) for column in pits.T]
    assert np.all((points >= 0) & (points <= 1))
    assert np.all(np.sum(gaussian ** 2, axis=1) < radius ** 2)
    assert max(statistics) < 5 / np.sqrt(options.chains)
    record = {
        "fit_sha256": hashlib.sha256(options.fit.read_bytes()).hexdigest(),
        "chains": options.chains, "slices": 90, "gap": options.gap,
        "seconds": time.monotonic() - started,
        "max_ks": max(statistics), "ks_statistics": statistics,
        "ks_limit": 5 / np.sqrt(options.chains),
        "mh_acceptance": float(accepted.mean() / 90),
        "mean_likelihood_evaluations": float(evaluations.mean()),
        "rms_displacement": np.sqrt(np.mean((points - initial) ** 2, axis=0)).tolist(),
        "endpoint_correlation": [
            float(np.corrcoef(points[:, i], initial[:, i])[0, 1]) for i in range(18)
        ],
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("nn_hessian_sampler.py", "nn_geometry.py", "hessian_chain.py")
        },
    }
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
