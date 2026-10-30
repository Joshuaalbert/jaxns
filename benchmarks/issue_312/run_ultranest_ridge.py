"""Run an adaptive UltraNest slice control on the normalized 18-D ridge."""

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
    parser.add_argument("--live-points", type=int, default=1000)
    parser.add_argument("--initial-slices", type=int, default=72)
    parser.add_argument("--max-slices", type=int, default=144)
    parser.add_argument("--sigma-stiff", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import numpy as np
    import scipy
    import ultranest
    from scipy.special import logsumexp
    from scipy.stats import norm, wasserstein_distance
    from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction

    if ultranest.__version__ != "4.6.3":
        raise ValueError("This control requires the pinned UltraNest 4.6.3.")
    if options.output.exists() or options.output.with_suffix(".progress.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    dimension, sigma, curvature = 18, 0.05, 0.3
    sigma_stiff = options.sigma_stiff
    peak = -np.log(sigma_stiff * np.sqrt(2 * np.pi)) - 14 * np.log(sigma * np.sqrt(2 * np.pi))
    calls = 0

    def log_likelihood(points):
        nonlocal calls
        calls += len(points)
        centre = 0.5 + curvature * np.sin(2 * np.pi * points[:, 1])
        residual = (points[:, 0] - centre) / sigma_stiff
        others = (points[:, 2:16] - 0.5) / sigma
        return peak - 0.5 * residual ** 2 - 0.5 * np.sum(others ** 2, axis=1)

    def prior_transform(points):
        return points

    # Check the vectorized rewrite against independently normalized factors.
    # These draws neither advance UltraNest's global RNG nor count as run work.
    validation = np.random.default_rng(120312).uniform(size=(64, dimension))
    validation[:32, 0] = 0.5 + curvature * np.sin(2 * np.pi * validation[:32, 1])
    validation[:32, 2:16] = np.random.default_rng(120313).normal(0.5, sigma, (32, 14))
    expected = norm.logpdf(validation[:, 0],
                           0.5 + curvature * np.sin(2 * np.pi * validation[:, 1]), sigma_stiff)
    expected += norm.logpdf(validation[:, 2:16], 0.5, sigma).sum(axis=1)
    np.testing.assert_allclose(log_likelihood(validation), expected, rtol=1e-12, atol=1e-10)
    calls = 0
    np.random.seed(options.seed)
    started = time.monotonic()
    progress_path = options.output.with_suffix(".progress.jsonl")

    def record_region(points, info, region, transformLayer, region_fresh):
        row = {
            "seed": options.seed, "iterations": int(info["it"]),
            "seconds": time.monotonic() - started, "evaluations": calls,
            "sampler_ncall": int(info["ncall"]), "log_Z": float(info["logz"]),
            "log_volume": float(info["logvol"]),
            "live_points": len(points["u"]), "slices": int(step_sampler.nsteps),
            "live_std": np.std(points["u"], axis=0).tolist(),
            "max_log_L": float(np.max(points["logl"])),
        }
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    step_sampler = SliceSampler(
        nsteps=options.initial_slices, max_nsteps=options.max_slices,
        generate_direction=generate_mixture_random_direction,
        adaptive_nsteps="move-distance", check_nsteps=False,
    )
    sampler = ultranest.ReactiveNestedSampler(
        [f"x{i}" for i in range(dimension)], log_likelihood, prior_transform,
        vectorized=True, log_dir=None,
    )
    constructor_calls = calls
    sampler.stepsampler = step_sampler
    # Keep reactive allocation and its default uncertainty/ESS requirements.
    # 1000 is a minimum, not a cap; no evaluation or improvement limit is set.
    result = sampler.run(
        min_num_live_points=options.live_points, frac_remain=0.01,
        show_status=False, viz_callback=record_region,
    )
    weighted = result["weighted_samples"]
    points = np.asarray(weighted["points"])
    log_likelihoods = np.asarray(weighted["logl"])
    # UltraNest's logw contains quadrature volume weights, before likelihood.
    # Store evidence contributions in the same convention as the Dynesty archive.
    log_weights = np.asarray(weighted["logw"]) + log_likelihoods
    np.testing.assert_allclose(logsumexp(log_weights), result["logz"], atol=1e-10)
    weights = np.exp(log_weights - logsumexp(log_weights))
    np.testing.assert_allclose(weights, weighted["weights"], rtol=1e-10, atol=1e-12)
    mean = np.sum(weights[:, None] * points, axis=0)
    std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
    step_log = np.asarray(step_sampler.logstat)
    sequence = sampler.run_sequence
    archive = options.output.with_suffix(".npz")
    np.savez_compressed(
        archive, samples=points, log_weights=log_weights, log_likelihood=log_likelihoods,
        step_log=step_log, sequence_nlive=np.asarray(sequence["nlive"]),
        sequence_log_volume=np.asarray(sequence["logvol"]),
        sequence_log_likelihood=np.asarray(sequence["logl"]),
    )
    metadata = {
        "seed": options.seed, "sampler": "ultranest", "ultranest": ultranest.__version__,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "rng": "numpy.random.seed", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "u_dtype": str(points.dtype),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "dimension": dimension, "min_live_points": options.live_points,
        "final_min_live_points": int(sampler.min_num_live_points),
        "final_roots": len(sampler.root.children),
        "max_live_points": int(np.max(sequence["nlive"])),
        "initial_slices": options.initial_slices, "max_slices": options.max_slices,
        "final_slices": int(step_sampler.nsteps),
        "mean_slices": float(np.mean(step_log[:, 2])),
        "fraction_at_max_slices": float(np.mean(step_log[:, 2] == options.max_slices)),
        "direction": "generate_mixture_random_direction", "adaptive_nsteps": "move-distance",
        "check_nsteps": False, "region_filter": False, "region": "MLFriends",
        "vectorized": True, "num_bootstraps": 30,
        "frac_remain": 0.01, "dlogz": 0.5, "dKL": 0.5, "min_ess": 400,
        "max_ncalls": None, "max_iters": None, "max_num_improvement_loops": -1,
        "sigma_stiff": sigma_stiff, "sigma": sigma, "curvature": curvature,
        "samples": len(points), "iterations": int(result["niter"]),
        "evaluations": calls, "constructor_evaluations": constructor_calls,
        "sampler_ncall": int(result["ncall"]), "seconds": time.monotonic() - started,
        "log_Z": float(result["logz"]), "log_Z_uncert": float(result["logzerr"]),
        "log_Z_bootstrap": float(result["logz_bs"]),
        "log_Z_uncert_bootstrap": float(result["logzerr_bs"]),
        "tail_fraction": float(np.expm1(result["logzerr_tail"])),
        "peak_gap": float(peak - np.max(log_likelihoods)),
        "ess": float(1 / np.sum(weights ** 2)),
        "posterior_mean": mean.tolist(), "posterior_std": std.tolist(),
        "step_log_labels": step_sampler.logstat_labels,
        "jump_distance_ratio": float(step_sampler.mean_jump_distance),
        "fraction_far_enough": float(step_sampler.far_enough_fraction),
        "insertion_order_converged": bool(result["insertion_order_MWW_test"]["converged"]),
        "completed": True, "normalized_gaussian_reference_passed": True,
    }
    uniform_grid = (np.arange(16384) + 0.5) / 16384
    for coordinate in (1, 5, 16, 17):
        metadata[f"std_x{coordinate}"] = float(std[coordinate])
        if coordinate != 5:
            metadata[f"w1_x{coordinate}"] = float(wasserstein_distance(
                points[:, coordinate], uniform_grid, u_weights=weights,
            ))
    options.output.write_text(json.dumps(metadata, allow_nan=False) + "\n")
    print(json.dumps(metadata, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
