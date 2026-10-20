"""Compare only the direction law on the reported 18-dimensional ridge."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from scipy.special import gammaln
from tensorflow_probability.substrates.jax import distributions as tfpd

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.priors import Prior


def prior_model(sigma_stiff: float, curvature: float):
    """Use curvature=0 to isolate linear anisotropy from a curved ridge."""
    x = Prior(
        tfpd.Uniform(low=jnp.zeros(18), high=jnp.ones(18)), name="x",
    ).realise()
    centre = 0.5 + curvature * jnp.sin(2 * jnp.pi * x[1])
    return (
        tfpd.Normal(centre, sigma_stiff).log_prob(x[0])
        + jnp.sum(tfpd.Normal(0.5, 0.05).log_prob(x[2:16]))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("isotropic", "covariance"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sigma-stiff", type=float, default=1e-5)
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument("--regularisation", type=float, default=1e-8)
    parser.add_argument("--refresh-efolds", type=float, default=1.0)
    parser.add_argument("--roots", type=int, default=150)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.cpu is not None:
        os.sched_setaffinity(0, {options.cpu})
    jax.config.update("jax_enable_x64", True)
    model = Model(prior_model=partial(
        prior_model, options.sigma_stiff, options.curvature,
    ))
    runner = NestedSampler(
        model=model,
        root_allocation_degree=options.roots,
        sampler=UniDimSliceSampler(num_slices=options.slices),
        replacement_width=options.width,
    )

    def run(seed):
        state = runner.initialise(key=jax.random.PRNGKey(seed))
        if options.mode == "covariance":
            state = state.covariance_directions(
                relative_regularisation=options.regularisation,
                refresh_efolds=options.refresh_efolds,
            )
        # Both arms finish the same first complete default-depth epoch.
        state = runner.resume_until_goal(
            state, lambda current: int(current.goal_loop_iter) > 0,
        )
        return jax.block_until_ready(state.to_result()), state

    metadata = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"],
        )).hexdigest(),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": options.mode,
        "sigma_stiff": options.sigma_stiff,
        "curvature": options.curvature,
        "regularisation": options.regularisation,
        "refresh_efolds": options.refresh_efolds,
        "roots": options.roots,
        "slices": options.slices,
        "width": options.width,
        "float64": True,
    }
    if options.warmup:
        started = time.monotonic()
        run(options.seeds[0])
        metadata["warmup_seconds_including_compilation"] = time.monotonic() - started
    options.output.parent.mkdir(parents=True, exist_ok=True)
    peak = (
        -np.log(options.sigma_stiff * np.sqrt(2 * np.pi))
        - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    )
    for seed in options.seeds:
        started = time.monotonic()
        result, state = run(seed)
        elapsed = time.monotonic() - started
        count = int(result.total_num_samples)
        points = np.asarray(result.X_samples["x"])[:count]
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        mean = np.sum(weights[:, None] * points, axis=0)
        std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
        gap = peak - np.asarray(result.log_L)[:count]
        interior = (gap > 0) & (gap < 50)
        # A volume-preserving shear turns each interior contour into a
        # 15-ball times three unit intervals. The Gaussian cube truncation
        # error in log Z is negligible (moderate coordinates are 10 sigma in).
        log_X_true = (
            7.5 * np.log(np.pi) - gammaln(8.5)
            + np.log(options.sigma_stiff) + 14 * np.log(0.05)
            + 7.5 * np.log(2 * gap[interior])
        )
        volume_error = np.average(
            np.asarray(result.log_X_mean)[:count][interior] - log_X_true,
            weights=weights[interior],
        )
        record = {
            **metadata,
            "seed": seed,
            "unit_dtypes": sorted({
                str(leaf.dtype) for leaf in jax.tree.leaves(result.U_samples)
            }),
            "samples": count,
            "evaluations": int(result.total_num_likelihood_evaluations),
            "seconds": elapsed,
            "timing": "warmed" if options.warmup else "including_compilation",
            "log_Z": float(result.log_Z_mean),
            "log_Z_uncert": float(result.log_Z_uncert),
            "peak_gap": float(peak - result.log_L_supremum),
            "std_x1": float(std[1]),
            "std_x16": float(std[16]),
            "std_x17": float(std[17]),
            "std_x5": float(std[5]),
            "posterior_mean_log_X_error": float(volume_error),
            "geometry_updates": (
                0 if state.covariance_data is None
                else int(state.covariance_data.num_updates)
            ),
            "termination_reason": int(state.termination_reason),
            "depth_reached": bool(state.depth_reached),
        }
        with options.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
