"""Replay archived 100-iteration EM fits through the independent NumPy path."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=10)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from jaxns.sampling.gmm import GaussianMixture, em_gmm_reference, initialise_gmm

    def read_mixture(path):
        with np.load(path) as saved:
            return GaussianMixture(*(jnp.asarray(saved[name]) for name in (
                "centres", "covariances", "log_masses", "valid",
            )))

    fits = [json.loads(line) for line in
            (options.directory / "fits.jsonl").read_text().splitlines()]
    records = []
    for fit in (fits[0], fits[-1]):
        path = options.directory / f"fit-{fit['update']:03d}.npz"
        observed = read_mixture(path)
        with np.load(path) as saved:
            points = saved["points"]
        capacity = fit["capacity"]
        data = np.zeros((capacity, 18))
        data[:len(points)] = points
        mask = np.arange(capacity) < len(points)
        if options.warm_start and fit["update"] > 0:
            initial = read_mixture(options.directory / f"fit-{fit['update'] - 1:03d}.npz")
        else:
            initial = initialise_gmm(
                jax.random.PRNGKey(31227), jnp.asarray(data), 10, mask=jnp.asarray(mask),
                regularisation=1e-6,
            )
        reference = em_gmm_reference(data, initial, mask.astype(float), mask, 100, 1e-6)
        for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(observed)):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
        records.append({
            "update": fit["update"], "population": len(points), "iterations": 100,
            "warm_start": options.warm_start, "max_centre_error": float(np.max(np.abs(
                np.asarray(observed.centres) - np.asarray(reference.centres),
            ))),
            "max_covariance_error": float(np.max(np.abs(
                np.asarray(observed.covariances) - np.asarray(reference.covariances),
            ))),
            "max_log_mass_error": float(np.max(np.abs(
                np.asarray(observed.log_masses) - np.asarray(reference.log_masses),
            ))),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    options.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
