"""Check stationarity and displacement from independent exact-contour draws."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument("--chains", type=int, default=8192)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--gap", type=float, default=15.0)
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument(
        "--hessian-contour-scale", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not 0 < options.gap < 50:
        parser.error("The analytic untruncated contour requires 0 < gap < 50")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.stats import beta, kstest

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler

    sigma = 1e-5
    peak = -np.log(sigma * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    radius = np.sqrt(2 * options.gap)
    rng = np.random.default_rng(312001)
    # The shear x0 = f(x1) + sigma*z0 has constant Jacobian. A uniform
    # 15-ball times three independent uniform coordinates therefore gives
    # exact constrained-prior draws, without an MCMC reference approximation.
    ball = rng.normal(size=(options.chains, 15))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    ball *= (radius * rng.uniform(size=options.chains) ** (1 / 15))[:, None]
    initial = rng.uniform(size=(options.chains, 18))
    initial[:, 0] = (
        0.5 + options.curvature * np.sin(2 * np.pi * initial[:, 1]) + sigma * ball[:, 0]
    )
    initial[:, 2:16] = 0.5 + 0.05 * ball[:, 1:]

    def likelihood(point):
        residual = point[0] - (0.5 + options.curvature * jnp.sin(2 * jnp.pi * point[1]))
        return peak - 0.5 * (
            (residual / sigma) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2)
        )

    def distribution_checks(points):
        gaussian = np.concatenate([
            ((points[:, 0] - 0.5 - options.curvature * np.sin(
                2 * np.pi * points[:, 1],
            )) / sigma)[:, None],
            (points[:, 2:16] - 0.5) / 0.05,
        ], axis=1)
        # Each standardized ball coordinate maps through Beta(8,8); the
        # radius to the power 15 is also uniform. Include all 18 marginals.
        pits = np.concatenate([
            points[:, [1, 16, 17]],
            beta.cdf((gaussian / radius + 1) / 2, 8, 8),
            (np.linalg.norm(gaussian, axis=1) / radius)[:, None] ** 15,
        ], axis=1)
        statistics = [float(kstest(column, "uniform").statistic) for column in pits.T]
        return {
            "ks_statistics": statistics,
            "max_ks": max(statistics),
            "positive_residual_fraction": float(np.mean(gaussian[:, 0] > 0)),
            "std_x1": float(np.std(points[:, 1])),
            "std_x16": float(np.std(points[:, 16])),
            "std_x17": float(np.std(points[:, 17])),
            "mean_squared_radius": float(np.mean(np.sum(gaussian ** 2, axis=1))),
        }

    options.output.parent.mkdir(parents=True, exist_ok=True)
    for corrected in (True, False):
        sampler = ToyHessianSliceSampler(
            num_slices=options.slices, curvature=options.curvature,
            correct_mh=corrected,
            scale_to_contour=options.hessian_contour_scale,
        )

        def one_chain(key, point, sampler=sampler):
            final, _ = sampler.run_chain(
                key, point, likelihood(point), jnp.asarray(peak - options.gap), likelihood,
            )
            return final.point, final.accepted, final.evaluations

        started = time.monotonic()
        points, accepted, evaluations = jax.block_until_ready(jax.jit(jax.vmap(one_chain))(
            jax.random.split(jax.random.PRNGKey(312002), options.chains),
            jnp.asarray(initial),
        ))
        points = np.asarray(points)
        assert np.all((points >= 0) & (points <= 1))
        assert np.all(np.asarray(jax.vmap(likelihood)(points)) > peak - options.gap)
        checks = distribution_checks(points)
        limit = 5 / np.sqrt(options.chains)
        record = {
            "correct_mh": corrected,
            "chains": options.chains,
            "slices": options.slices,
            "gap": options.gap,
            "curvature": options.curvature,
            "precision_floor": sampler.precision_floor,
            "hessian_contour_scale": sampler.scale_to_contour,
            "sampler_sha256": hashlib.sha256(
                Path(__file__).with_name("hessian_sampler.py").read_bytes(),
            ).hexdigest(),
            "initial": distribution_checks(initial),
            "final": checks,
            "ks_limit": limit,
            "stationarity_pass": bool(checks["max_ks"] < limit),
            "acceptance_fraction": float(np.mean(accepted) / options.slices),
            "evaluations_per_slice": float(np.mean(evaluations) / options.slices),
            "rms_displacement": np.sqrt(np.mean((points - initial) ** 2, axis=0)).tolist(),
            "endpoint_correlations": [
                float(np.corrcoef(initial[:, index], points[:, index])[0, 1])
                for index in (1, 16, 17, 5)
            ],
            "seconds_including_compilation": time.monotonic() - started,
        }
        with options.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if corrected:
            assert checks["max_ks"] < limit, "Corrected kernel failed stationarity check"


if __name__ == "__main__":
    main()
