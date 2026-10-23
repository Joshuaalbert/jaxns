"""Audit density fitting on the known ridge and saved posterior populations."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

# Historical checkpoints reference the evidence runner's __main__ model.
from run_ridge import prior_model  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("oracle", "saved-0", "saved-1", "saved-2"),
                        required=True)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "gmm_fit")
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import jax.numpy as jnp
    import numpy as np
    import scipy
    import sklearn
    from scipy.stats import norm, wasserstein_distance

    jax.config.update("jax_enable_x64", True)
    from gmm_split_reference import (
        DensityFit,
        draw_mixture,
        log_density,
        marginal,
        split_fit,
    )

    from jaxns.checkpoint import CheckpointManager
    from jaxns.sampling.gmm import (
        GaussianMixture,
        em_gmm_reference,
        fit_gmm,
        initialise_gmm,
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    seed = 20261021 + (0 if options.dataset == "oracle" else 1 + int(options.dataset[-1]))
    rng = np.random.default_rng(seed)
    data_path = options.output_dir / f"{options.dataset}.npz"
    metadata_path = options.output_dir / f"{options.dataset}.input.json"
    sigma = 1e-5
    peak = -np.log(sigma * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))

    def exact_samples(count: int) -> np.ndarray:
        points = rng.uniform(size=(count, 18))
        points[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * points[:, 1]) + sigma * rng.normal(size=count)
        points[:, 2:16] = rng.normal(0.5, 0.05, size=(count, 14))
        # Rejection, rather than clipping, respects the cube support. For this
        # target the Gaussian tail probability is negligible at this scale.
        invalid = np.any((points < 0) | (points > 1), axis=1)
        if np.any(invalid):
            points[invalid] = exact_samples(int(invalid.sum()))
        return points.astype(np.float32)

    if data_path.exists():
        with np.load(data_path) as archive:
            train, test = archive["train"], archive["test"]
        input_metadata = json.loads(metadata_path.read_text())
        if len(train) != options.samples or input_metadata["seed"] != seed:
            raise ValueError("Existing data belong to a different configuration.")
    else:
        input_metadata = {"dataset": options.dataset, "seed": seed, "samples": options.samples}
        if options.dataset == "oracle":
            train, test = exact_samples(options.samples), exact_samples(options.samples)
            input_metadata["source"] = "Independent analytic posterior draws, cast to U float32"
        else:
            run_seed = int(options.dataset[-1])
            checkpoint = (options.checkpoint_root
                          / f"issue312-gmm10-180-{run_seed}/evidence/seed-{run_seed}")
            with CheckpointManager(checkpoint) as manager:
                state = manager.load()
            if state is None:
                raise ValueError(f"Missing evidence checkpoint {checkpoint}")
            results = jax.block_until_ready(state.to_result())
            count = int(results.total_num_samples)
            points = np.asarray(results.X_samples["x"])[:count]
            log_weights = np.asarray(results.log_dp)[:count]
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()
            train = points[rng.choice(count, options.samples, p=weights)]
            test = points[rng.choice(count, options.samples, p=weights)]
            input_metadata.update(
                source="Independent resamples of the same saved weighted empirical posterior",
                checkpoint=str(checkpoint), source_samples=count,
                checkpoint_checksum=json.loads((checkpoint / "CHECKPOINT").read_text())["checksum"],
                source_ess=float(results.ess), source_log_Z=float(results.log_Z_mean),
                source_log_Z_uncert=float(results.log_Z_uncert),
            )
            del state, results
        np.savez_compressed(data_path, train=train, test=test)
        metadata_path.write_text(json.dumps(input_metadata, indent=2) + "\n")
    data_checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    train_jax = jnp.asarray(train)
    mask = jnp.ones(len(train), dtype=bool)
    # Even equal-weight controls exercise the production weighted initializer:
    # mass quantiles of input order, rather than the historical random path.
    log_weights = jnp.zeros(len(train), dtype=jnp.float64)
    regularisation = 1e-6
    key = jax.random.PRNGKey(0)
    compiled_fit = jax.jit(fit_gmm, static_argnames=("n_components", "n_iters", "regularisation"))
    models = {}
    traces = {}
    timings = {}

    def numpy_fit(fit: GaussianMixture) -> DensityFit:
        return DensityFit(np.asarray(fit.centres), np.asarray(fit.covariances),
                          np.exp(np.asarray(fit.log_masses)))

    def refine(initial: GaussianMixture, label: str) -> GaussianMixture:
        started = time.monotonic()
        current = initial
        trace = []
        previous_iteration = 0
        for iteration in sorted({10, 20, 50, 100, 200, options.iterations}):
            if iteration > options.iterations:
                continue
            current, _, _ = jax.block_until_ready(compiled_fit(
                key, train_jax, len(initial.centres), mask=mask,
                log_sample_weights=log_weights, initial=current,
                n_iters=iteration - previous_iteration, regularisation=regularisation,
            ))
            converted = numpy_fit(current)
            trace.append({"iteration": iteration,
                          "train_mean_log_density": float(log_density(train, converted).mean()),
                          "test_mean_log_density": float(log_density(test, converted).mean())})
            if iteration in (10, 100):
                models[f"{label}-{iteration}"] = converted
            previous_iteration = iteration
            print(json.dumps({"dataset": options.dataset, "fit": label, **trace[-1]}), flush=True)
        traces[label] = trace
        timings[label] = time.monotonic() - started
        models[f"{label}-{options.iterations}"] = numpy_fit(current)
        return current

    initial = initialise_gmm(key, train_jax, options.components, mask=mask,
                             log_sample_weights=log_weights, regularisation=regularisation)
    refine(initial, "current")
    split_started = time.monotonic()
    split, split_trace = split_fit(train, options.components, seed, regularisation)
    timings["split"] = time.monotonic() - split_started
    models["split"] = split
    split_initial = GaussianMixture(
        centres=jnp.asarray(split.centres), covariances=jnp.asarray(split.covariances),
        log_masses=jnp.log(jnp.asarray(split.masses)), valid=jnp.ones(len(split.masses), bool),
    )
    refine(split_initial, "split-em")
    if options.dataset != "oracle":
        saved_path = (Path(__file__).parent / "raw"
                      / f"staged-gmm10-180-{options.dataset[-1]}.gmm.fit.json")
        saved = json.loads(saved_path.read_text())
        models["original-weighted-10"] = DensityFit(
            np.asarray(saved["centres"]), np.asarray(saved["covariances"]),
            np.asarray(saved["mixture_masses"]),
        )

    # Compare arithmetic with a separate NumPy implementation on this ridge,
    # rather than inferring an EM bug from a poor density approximation alone.
    small = train[:512].astype(np.float64)
    small_initial = initialise_gmm(
        key, jnp.asarray(small), 3, mask=jnp.ones(512, bool),
        log_sample_weights=jnp.zeros(512),
    )
    check, _, _ = jax.block_until_ready(compiled_fit(
        key, jnp.asarray(small), 3, mask=jnp.ones(512, bool),
        log_sample_weights=jnp.zeros(512), initial=small_initial, n_iters=10,
        regularisation=regularisation,
    ))
    reference = em_gmm_reference(
        small, small_initial, np.ones(512) / 512, np.ones(512, bool), n_iters=10,
    )
    for actual, expected in ((check.centres, reference.centres),
                             (check.covariances, reference.covariances),
                             (check.log_masses, reference.log_masses)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    records = []
    uniform_grid = (np.arange(16384) + 0.5) / 16384
    test_log_likelihood = (
        peak - 0.5 * ((test[:, 0].astype(float) - 0.5
                      - 0.3 * np.sin(2 * np.pi * test[:, 1].astype(float))) / sigma) ** 2
        - 0.5 * np.sum(((test[:, 2:16].astype(float) - 0.5) / 0.05) ** 2, axis=1)
    )
    for name, fit in models.items():
        draws = draw_mixture(fit, 65536, seed + 100)
        residual = draws[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * draws[:, 1])
        log_q = log_density(test, fit)
        record = {
            "name": name, "components": len(fit.masses),
            "train_mean_log_density": float(log_density(train, fit).mean()),
            "test_mean_log_density": float(log_q.mean()),
            "oracle_kl_estimate": (float(np.mean(test_log_likelihood - log_q))
                                   if options.dataset == "oracle" else None),
            "mass_sum": float(fit.masses.sum()),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "fraction_residual_within_3sigma": float(np.mean(np.abs(residual) < 3 * sigma)),
            "fraction_outside_cube": float(np.mean(np.any((draws < 0) | (draws > 1), axis=1))),
            "largest_mass": float(fit.masses.max()),
            "mean": (fit.masses @ fit.centres).tolist(),
            "centres": fit.centres.tolist(), "covariances": fit.covariances.tolist(),
            "masses": fit.masses.tolist(),
        }
        for dimension in (0, 1, 5, 16, 17):
            record[f"w1_data_x{dimension}"] = float(wasserstein_distance(
                test[:, dimension], draws[:, dimension],
            ))
        for dimension in (1, 16, 17):
            record[f"w1_uniform_x{dimension}"] = float(wasserstein_distance(
                uniform_grid, draws[:, dimension],
            ))
        # Check marginal normalization independently of the sampling routine.
        one = marginal(fit, [5])
        width = np.sqrt(one.covariances[:, 0, 0])
        integral = np.sum(one.masses * (
            norm.cdf((10 - one.centres[:, 0]) / width)
            - norm.cdf((-10 - one.centres[:, 0]) / width)
        ))
        np.testing.assert_allclose(integral, 1, atol=1e-10)
        records.append(record)
    output = {
        "dataset": options.dataset, "input": input_metadata, "data_checksum": data_checksum,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            Path(__file__).with_name("gmm_split_reference.py").read_bytes(),
        ).hexdigest(),
        "jax": jax.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "python": platform.python_version(),
        "u_dtype": str(train.dtype), "regularisation": regularisation,
        "max_components": options.components, "em_iterations": options.iterations,
        "split_trace": split_trace, "em_traces": traces, "seconds_including_compilation": timings,
        "numpy_em_reference_passed": True, "models": records,
    }
    (options.output_dir / f"{options.dataset}.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps({"dataset": options.dataset, "models": [{
        key: record[key] for key in (
            "name", "components", "test_mean_log_density",
            "residual_rms", "fraction_residual_within_3sigma",
        )
    } for record in records]}), flush=True)


if __name__ == "__main__":
    main()
