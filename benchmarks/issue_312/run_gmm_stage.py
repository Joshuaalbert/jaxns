"""Continue a completed ridge evidence run to a fixed multiple of its Kish ESS."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

# The evidence runner's checkpoints pickle this model as __main__.prior_model.
# Importing it also keeps the same explicit model symbol for our checkpoints.
from run_ridge import prior_model  # noqa: F401


def posterior_summary(result, peak):
    """Report actual posterior spread alongside the weight-only ESS goal."""
    import numpy as np
    from scipy.stats import wasserstein_distance

    count = int(result.total_num_samples)
    points = np.asarray(result.X_samples["x"])[:count]
    log_weights = np.asarray(result.log_dp)[:count]
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    mean = np.sum(weights[:, None] * points, axis=0)
    std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
    summary = {
        "samples": count,
        "evaluations": int(result.total_num_likelihood_evaluations),
        "log_Z": float(result.log_Z_mean),
        "log_Z_uncert": float(result.log_Z_uncert),
        "ess": float(result.ess),
        "peak_gap": float(peak - result.log_L_supremum),
        "posterior_mean": mean.tolist(),
        "posterior_std": std.tolist(),
    }
    uniform_grid = (np.arange(4096) + 0.5) / 4096
    for dimension in (1, 16, 17, 5):
        summary[f"std_x{dimension}"] = float(std[dimension])
        if dimension != 5:
            summary[f"w1_x{dimension}"] = float(wasserstein_distance(
                points[:, dimension], uniform_grid, u_weights=weights,
            ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-record", type=Path, required=True)
    parser.add_argument("--evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gmm", "isotropic"), required=True)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--iso-prob", type=float, default=0.01)
    parser.add_argument("--regularisation", type=float, default=1e-6)
    parser.add_argument("--ess-multiple", type=float, default=2.0)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not math.isfinite(options.ess_multiple) or options.ess_multiple <= 1:
        parser.error("--ess-multiple must be finite and greater than one")
    os.sched_setaffinity(0, {options.cpu})
    # Backend imports follow affinity, just as in the evidence-stage runner.
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from jaxns.checkpoint import CheckpointInUseError, CheckpointManager
    from jaxns.constrained_sampler import UniDimSliceSampler
    from jaxns.core import NestedSampler
    from jaxns.sampling.ellipsoid import component_probabilities

    started = time.monotonic()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_diff = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()
    rows = options.evidence_record.read_text().splitlines()
    if len(rows) != 1:
        raise ValueError("Expected exactly one completed evidence-stage record.")
    evidence = json.loads(rows[0])
    if not (
        evidence["mode"] == "isotropic" and evidence["goal_reached"]
        and evidence["allocation_target"] == "evidence_improving"
        and evidence["target_log_Z_uncert"] == 0.1
        and evidence["termination_reason"] == 0
    ):
        raise ValueError("The first stage must reach uncertainty 0.1 with isotropic directions.")
    # Paired arms briefly need the same checkpoint lock while reading their
    # common baseline. Retry only this read; continuation directories differ.
    for attempt in range(6):
        try:
            with CheckpointManager(options.evidence_checkpoint) as manager:
                baseline_state = manager.load()
            break
        except CheckpointInUseError:
            if attempt == 5:
                raise
            time.sleep(0.25 * 2 ** attempt)
    if baseline_state is None:
        raise ValueError("Missing evidence-stage checkpoint.")
    if baseline_state.covariance_data is not None or baseline_state.sampler_data is not None:
        raise ValueError("The baseline must not contain covariance or GMM directions.")
    if int(baseline_state.num_samples) != evidence["samples"]:
        raise ValueError("Checkpoint and evidence record have different sample counts.")
    peak = (
        -np.log(evidence["sigma_stiff"] * np.sqrt(2 * np.pi))
        - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    )
    baseline = posterior_summary(jax.block_until_ready(baseline_state.to_result()), peak)
    if baseline["evaluations"] != evidence["evaluations"] or baseline["log_Z_uncert"] > 0.1:
        raise ValueError("Checkpoint does not match the completed evidence goal.")
    target_ess = options.ess_multiple * baseline["ess"]
    baseline_epoch = int(baseline_state.goal_loop_iter)
    manifest = json.loads((options.evidence_checkpoint / "CHECKPOINT").read_text())
    configuration = {
        "baseline_checksum": manifest["checksum"],
        "mode": options.mode,
        "components": options.components,
        "fit_iterations": options.fit_iterations,
        "iso_prob": options.iso_prob,
        "regularisation": options.regularisation,
        "target_ess": target_ess,
        "slices": evidence["slices"],
    }
    options.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = options.checkpoint_dir / "experiment.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise ValueError("Continuation directory belongs to another configuration.")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2) + "\n")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    state = baseline_state
    fit_seconds = 0.0
    fit_record = None
    if options.mode == "gmm":
        fit_started = time.monotonic()
        state = jax.block_until_ready(state.fit_gmm_directions(
            num_components=options.components,
            num_iterations=options.fit_iterations,
            iso_prob=options.iso_prob,
            regularisation=options.regularisation,
        ))
        fit_seconds = time.monotonic() - fit_started
        np.testing.assert_array_equal(state.random_key, baseline_state.random_key)
        np.testing.assert_array_equal(state.goal_key, baseline_state.goal_key)
        data = state.sampler_data
        centres = np.asarray(data.centres)
        residual = (
            centres[:, 0] - 0.5
            - evidence["curvature"] * np.sin(2 * np.pi * centres[:, 1])
        )
        # This analytic diagnostic is separate from fitting and consumes no
        # likelihood calls or sampler randomness. It tests the fitted heights.
        true_heights = (
            peak - 0.5 * (residual / evidence["sigma_stiff"]) ** 2
            - 0.5 * np.sum(((centres[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
        )
        fit_record = {
            "fit_seconds": fit_seconds,
            "centres": centres.tolist(),
            "covariances": np.asarray(data.mixture.covariances).tolist(),
            "mixture_masses": np.exp(np.asarray(data.mixture.log_masses)).tolist(),
            "valid": np.asarray(data.valid).tolist(),
            "log_L_at_mean": np.asarray(data.log_L_at_mean).tolist(),
            "analytic_log_L_at_mean": true_heights.tolist(),
            "num_updates": int(data.num_updates),
            "selection_probabilities_by_peak_gap": {
                str(gap): np.asarray(component_probabilities(data, peak - gap)).tolist()
                for gap in (1, 2, 5, 10, 15, 20, 30, 50)
            },
        }
        options.output.with_suffix(".fit.json").write_text(
            json.dumps(fit_record, indent=2) + "\n",
        )
        print(json.dumps({"seed": evidence["seed"], "fit_seconds": fit_seconds,
                          "valid_components": int(np.sum(data.valid)),
                          "baseline_ess": baseline["ess"], "target_ess": target_ess}), flush=True)

    runner = NestedSampler(
        model=state.model,
        root_allocation_degree=evidence["roots"],
        sampler=UniDimSliceSampler(num_slices=evidence["slices"]),
        replacement_width=evidence["width"],
        allocation_target="posterior_improving",
        unlimited_samples=True,
    )
    progress_path = options.output.with_suffix(".progress.jsonl")

    def goal(current):
        if current.covariance_data is not None:
            raise ValueError("Covariance whitening must remain disabled.")
        summary = posterior_summary(jax.block_until_ready(current.to_result()), peak)
        progress = {
            "seed": evidence["seed"], "mode": options.mode,
            "epoch": int(current.goal_loop_iter),
            "stage_epochs": int(current.goal_loop_iter) - baseline_epoch,
            "seconds": time.monotonic() - started,
            "additional_evaluations": summary["evaluations"] - baseline["evaluations"],
            "target_ess": target_ess,
            **summary,
        }
        if current.sampler_data is not None:
            progress["num_directions"] = int(current.sampler_data.num_directions)
            progress["num_isotropic"] = int(current.sampler_data.num_isotropic)
            progress["fit_updates"] = int(current.sampler_data.num_updates)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        print(json.dumps(progress), flush=True)
        return summary["ess"] >= target_ess

    state = runner.resume_until_goal(
        state, goal, checkpoint_dir=options.checkpoint_dir, checkpoint_cadence=0,
    )
    result = jax.block_until_ready(state.to_result())
    final = posterior_summary(result, peak)
    record = {
        **configuration,
        "seed": evidence["seed"],
        "commit": source_commit,
        "diff_sha256": source_diff,
        "evidence_commit": evidence["commit"],
        "jax": jax.__version__, "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "allocation_target": "posterior_improving",
        "covariance_whitening": False,
        "fit_seconds": fit_seconds,
        "fit_valid_components": None if fit_record is None else sum(fit_record["valid"]),
        "baseline": baseline,
        "final": final,
        "stage_epochs": int(state.goal_loop_iter) - baseline_epoch,
        "additional_evaluations": final["evaluations"] - baseline["evaluations"],
        "seconds_including_compilation": time.monotonic() - started,
        "termination_reason": int(state.termination_reason),
        "goal_reached": final["ess"] >= target_ess and int(state.termination_reason) == 0,
        "num_directions": (
            0 if state.sampler_data is None else int(state.sampler_data.num_directions)
        ),
        "num_isotropic": (
            0 if state.sampler_data is None else int(state.sampler_data.num_isotropic)
        ),
    }
    options.output.write_text(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if not record["goal_reached"]:
        raise RuntimeError("Continuation ended without doubling the baseline ESS.")


if __name__ == "__main__":
    main()
