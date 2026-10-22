"""Verify the reporter's Dynesty rslice control on the analytic 18-D ridge."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--live-points", type=int, default=150)
    parser.add_argument("--slices", type=int, default=90)
    parser.add_argument("--sigma-stiff", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import dynesty
    import numpy as np
    import scipy
    from scipy.special import logsumexp
    from scipy.stats import norm, wasserstein_distance

    if dynesty.__version__ != "3.0.0":
        raise ValueError("This reproduction requires the reporter's Dynesty 3.0.0.")
    if options.output.exists():
        raise FileExistsError(f"Refusing to overwrite a completed run: {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    dimension, sigma, curvature = 18, 0.05, 0.3
    sigma_stiff = options.sigma_stiff
    peak = -np.log(sigma_stiff * np.sqrt(2 * np.pi)) - 14 * np.log(sigma * np.sqrt(2 * np.pi))
    calls = 0

    def log_likelihood(point):
        nonlocal calls
        calls += 1
        centre = 0.5 + curvature * np.sin(2 * np.pi * point[1])
        residual = (point[0] - centre) / sigma_stiff
        others = (point[2:16] - 0.5) / sigma
        return peak - 0.5 * residual ** 2 - 0.5 * np.sum(others ** 2)

    def prior_transform(point):
        return point

    # Check the NumPy rewrite against normalized Gaussian factors, including
    # points near the posterior. Validation uses a separate RNG and is not
    # counted as sampling work or allowed to consume the sampler's random keys.
    validation = np.random.default_rng(120312).uniform(size=(64, dimension))
    validation[:32, 0] = 0.5 + curvature * np.sin(2 * np.pi * validation[:32, 1])
    validation[:32, 2:16] = np.random.default_rng(120313).normal(0.5, sigma, (32, 14))
    for point in validation:
        centre = 0.5 + curvature * np.sin(2 * np.pi * point[1])
        expected = norm.logpdf(point[0], centre, sigma_stiff)
        expected += norm.logpdf(point[2:16], 0.5, sigma).sum()
        np.testing.assert_allclose(log_likelihood(point), expected, rtol=1e-12, atol=1e-10)
    calls = 0
    started = time.monotonic()
    last_progress = started
    last_delta = None
    progress_path = options.output.with_suffix(".progress.jsonl")

    def print_progress(result, iteration, ncall, *, dlogz, logl_max, add_live_it=None):
        nonlocal last_progress, last_delta
        if add_live_it is None:
            last_delta = float(result.delta_logz)
        current = time.monotonic()
        if current - last_progress < 30 and add_live_it != options.live_points:
            return
        row = {
            "seed": options.seed, "iterations": iteration,
            "seconds": current - started, "evaluations": calls,
            "log_Z": float(result.logz), "delta_log_Z": float(result.delta_logz),
            "added_live": add_live_it,
        }
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        last_progress = current

    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, dimension,
        nlive=options.live_points, bound="multi", sample="rslice", slices=options.slices,
        rstate=np.random.default_rng(options.seed),
    )
    # Preserve the reported stopping rule and all other Dynesty defaults. This
    # dlogz is a remaining-evidence criterion, not a 0.01 uncertainty goal.
    sampler.run_nested(dlogz=0.01, print_func=print_progress)
    result = sampler.results
    points = np.asarray(result.samples)
    weights = np.exp(result.logwt - logsumexp(result.logwt))
    mean = np.sum(weights[:, None] * points, axis=0)
    std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
    if not sampler.added_live or len(points) != result.niter + options.live_points:
        raise ValueError("The completed result must include all remaining live points.")
    if last_delta is None or last_delta > 0.011:
        # The iterator's last yielded row precedes its final stopping check;
        # allow one nlive=150 compression step when checking the recorded delta.
        raise ValueError(f"Unexpected final remaining-evidence estimate: {last_delta}")
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    archive = options.output.with_suffix(".npz")
    np.savez_compressed(
        archive, samples=points, log_weights=np.asarray(result.logwt),
        log_likelihood=np.asarray(result.logl), log_volume=np.asarray(result.logvol),
    )
    metadata = {
        "seed": options.seed, "sampler": "dynesty", "dynesty": dynesty.__version__,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "u_dtype": str(points.dtype),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "dimension": dimension, "live_points": options.live_points,
        "slices": options.slices, "bound": "multi", "sample": "rslice", "dlogz": 0.01,
        "sigma_stiff": sigma_stiff, "sigma": sigma, "curvature": curvature,
        "samples": len(points), "iterations": int(result.niter),
        "evaluations": calls, "sum_result_ncall": int(np.sum(result.ncall)),
        "sampler_ncall": int(sampler.ncall), "seconds": time.monotonic() - started,
        "log_Z": float(result.logz[-1]), "log_Z_uncert": float(result.logzerr[-1]),
        "peak_gap": float(peak - np.max(result.logl)),
        "ess": float(1 / np.sum(weights ** 2)),
        "posterior_mean": mean.tolist(), "posterior_std": std.tolist(),
        "last_yielded_delta_log_Z": last_delta, "added_live": bool(sampler.added_live),
        "normalized_gaussian_reference_passed": True,
    }
    uniform_grid = (np.arange(16384) + 0.5) / 16384
    for coordinate in (1, 5, 16, 17):
        metadata[f"std_x{coordinate}"] = float(std[coordinate])
        if coordinate != 5:
            metadata[f"w1_x{coordinate}"] = float(wasserstein_distance(
                points[:, coordinate], uniform_grid, u_weights=weights,
            ))
    options.output.write_text(json.dumps(metadata) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
