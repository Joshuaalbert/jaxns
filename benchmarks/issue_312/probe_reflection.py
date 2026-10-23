"""Independent-contour stationarity and movement grid for midpoint reflections."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--chains", type=int, default=8192)
    parser.add_argument("--gap", type=float, default=15.0)
    parser.add_argument("--cycles", type=int, default=9)
    parser.add_argument("--slices", type=int, default=10)
    parser.add_argument("--bounces", nargs="+", type=int, default=[10])
    parser.add_argument("--steps", nargs="+", type=float, default=[0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not 0 < options.gap < 50 or options.chains % 256:
        parser.error("Use 0 < gap < 50 and a multiple of 256 chains.")
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.stats import beta, kstest

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.reflection_sampler import ReflectionSampler

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
        return -0.5 * ((residual / 1e-5) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2))

    keys = jax.random.split(jax.random.PRNGKey(312002), options.chains)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for bounces in options.bounces:
        for step in options.steps:
            sampler = ReflectionSampler(
                num_cycles=options.cycles, slices_per_cycle=options.slices,
                num_bounces=bounces, step_size=step, boundary_only=options.boundary_only,
            )

            @jax.jit
            @jax.vmap
            def chain(key, point, sampler=sampler):
                result, _ = sampler.run_chain(
                    key, point, likelihood(point), jnp.asarray(-options.gap), likelihood,
                )
                return result

            started = time.monotonic()
            batches = [chain(keys[i:i + 256], jnp.asarray(initial[i:i + 256]))
                       for i in range(0, options.chains, 256)]
            result = jax.tree.map(lambda *values: np.concatenate(values), *batches)
            points = result.point
            assert np.all((points >= 0) & (points <= 1))
            assert np.all(np.asarray(jax.vmap(likelihood)(points)) > -options.gap)
            gaussian = np.column_stack([
                (points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5,
                (points[:, 2:16] - 0.5) / 0.05,
            ])
            pits = np.column_stack([
                points[:, [1, 16, 17]], beta.cdf((gaussian / radius + 1) / 2, 8, 8),
                (np.linalg.norm(gaussian, axis=1) / radius) ** 15,
            ])
            statistics = [float(kstest(column, "uniform").statistic) for column in pits.T]
            record = {
                "chains": options.chains, "gap": options.gap,
                "cycles": options.cycles, "slices_per_cycle": options.slices,
                "bounces": bounces, "step_size": step, "boundary_only": options.boundary_only,
                "max_ks": max(statistics), "ks_statistics": statistics,
                "ks_limit": 5 / np.sqrt(options.chains),
                "acceptance": float(np.mean(result.accepted) / options.cycles),
                "mean_evaluations": float(np.mean(result.evaluations)),
                "mean_gradients": float(np.mean(result.gradients)),
                "rms_displacement": np.sqrt(np.mean((points - initial) ** 2, axis=0)).tolist(),
                "correlations_x1_x16_x17_x5": [
                    float(np.corrcoef(initial[:, i], points[:, i])[0, 1])
                    for i in (1, 16, 17, 5)
                ],
                "seconds": time.monotonic() - started,
                "sampler_sha256": hashlib.sha256(
                    Path(__file__).with_name("reflection_sampler.py").read_bytes(),
                ).hexdigest(),
            }
            with options.output.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            assert max(statistics) < record["ks_limit"], "Stationarity check failed"


if __name__ == "__main__":
    main()
