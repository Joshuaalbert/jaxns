"""Compare only the direction law on the reported 18-dimensional ridge."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path


def prior_model(sigma_stiff: float, curvature: float):
    """Use curvature=0 to isolate linear anisotropy from a curved ridge."""
    from jax import numpy as jnp
    from tensorflow_probability.substrates.jax import distributions as tfpd

    from jaxns.priors import Prior

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
    parser.add_argument("--unit-dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument(
        "--allocation-target", choices=("uniform", "evidence_improving"),
        default="uniform",
    )
    parser.add_argument("--log-z-uncert", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.log_z_uncert is not None and (
        not math.isfinite(options.log_z_uncert) or options.log_z_uncert <= 0
    ):
        parser.error("--log-z-uncert must be finite and positive")
    if options.cpu is not None:
        os.sched_setaffinity(0, {options.cpu})
    # Pin before importing libraries that initialise backend/thread pools.
    # Changing only the main thread's affinity afterwards is insufficient.
    import jax
    import numpy as np
    from scipy.special import gammaln

    jax.config.update("jax_enable_x64", True)
    from jaxns.constrained_sampler import UniDimSliceSampler
    from jaxns.core import NestedSampler
    from jaxns.model import Model

    if options.unit_dtype is not None:
        import jaxctx

        import jaxns.model as model_module

        # This process-local benchmark override reaches every JAXCTX init/apply
        # call before tracing. Merely enabling JAX x64 leaves its default cube
        # dtype at float32. Production Model and sampler APIs stay unchanged.
        model_module.transform = partial(
            jaxctx.transform,
            base_dtype={"float32": jax.numpy.float32, "float64": jax.numpy.float64}[
                options.unit_dtype
            ],
        )

    model = Model(prior_model=partial(
        prior_model, options.sigma_stiff, options.curvature,
    ))
    runner = NestedSampler(
        model=model,
        root_allocation_degree=options.roots,
        sampler=UniDimSliceSampler(num_slices=options.slices),
        replacement_width=options.width,
        allocation_target=options.allocation_target,
        # A requested uncertainty goal must not silently become a sample-cap
        # comparison. The runner grows storage at its existing safe boundaries.
        unlimited_samples=options.log_z_uncert is not None,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = options.output.with_suffix(".progress.jsonl")

    def run(seed, record_progress=True):
        started = time.monotonic()
        state = runner.initialise(key=jax.random.PRNGKey(seed))
        if options.mode == "covariance":
            state = state.covariance_directions(
                relative_regularisation=options.regularisation,
                refresh_efolds=options.refresh_efolds,
            )

        def goal(current):
            if options.log_z_uncert is None:
                # Preserve the original single-epoch direction comparison.
                return int(current.goal_loop_iter) > 0
            if int(current.goal_loop_iter) == 0:
                return False
            uncertainty = float(current.expected_log_Z_uncert)
            if record_progress:
                progress = {
                    "seed": seed,
                    "mode": options.mode,
                    "allocation_target": options.allocation_target,
                    "epoch": int(current.goal_loop_iter),
                    "samples": int(current.num_samples),
                    "evaluations": int(current.total_num_likelihood_evaluations),
                    "log_Z": float(current.expected_log_Z_mean),
                    "log_Z_uncert": uncertainty,
                    "seconds": time.monotonic() - started,
                    "geometry_updates": (
                        0 if current.covariance_data is None
                        else int(current.covariance_data.num_updates)
                    ),
                }
                with progress_path.open("a") as stream:
                    stream.write(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
            return uncertainty <= options.log_z_uncert

        checkpoint_dir = None
        if record_progress and options.checkpoint_dir is not None:
            checkpoint_dir = options.checkpoint_dir / f"seed-{seed}"
        state = runner.resume_until_goal(
            state, goal, checkpoint_dir=checkpoint_dir, checkpoint_cadence=0,
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
        "jaxctx": version("jaxctx"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        "allocation_target": options.allocation_target,
        "target_log_Z_uncert": options.log_z_uncert,
        "unlimited_samples": options.log_z_uncert is not None,
        "float64": True,
        "requested_unit_dtype": options.unit_dtype,
    }
    if options.warmup:
        started = time.monotonic()
        run(options.seeds[0], record_progress=False)
        metadata["warmup_seconds_including_compilation"] = time.monotonic() - started
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
        unit_dtypes = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(result.U_samples)})
        if options.unit_dtype is not None and unit_dtypes != [options.unit_dtype]:
            raise ValueError(f"Cube dtype override failed: {unit_dtypes}")
        log_weights = np.asarray(result.log_dp)[:count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        if options.archive_dir is not None:
            options.archive_dir.mkdir(parents=True, exist_ok=True)
            archive = options.archive_dir / f"seed-{seed}.npz"
            if archive.exists():
                raise FileExistsError(f"Refusing to overwrite {archive}")
            np.savez_compressed(
                archive,
                samples=points,
                unit_samples=np.asarray(jax.tree.leaves(result.U_samples)[0])[:count],
                log_weights=log_weights,
                log_likelihood=np.asarray(result.log_L)[:count],
                log_volume=np.asarray(result.log_X_mean)[:count],
            )
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
            "unit_dtypes": unit_dtypes,
            "physical_dtype": str(points.dtype),
            "log_likelihood_dtype": str(result.log_L.dtype),
            "evidence_dtype": str(result.log_Z_mean.dtype),
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
            "epochs": int(state.goal_loop_iter),
            "goal_reached": (
                bool(state.depth_reached) and int(state.termination_reason) == 0
                and (
                    options.log_z_uncert is None
                    or float(state.expected_log_Z_uncert) <= options.log_z_uncert
                )
            ),
        }
        with options.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
