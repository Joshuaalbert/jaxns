"""Calibrate the SS8 separation against two evidence-uncertainty goals.

This is an exploratory companion to the canonical paper experiment.  One
process owns one seed so independent seeds can run concurrently without
sharing checkpoints or output files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.paper_phantom_conditioning import cases
from jaxns.checkpoint import CheckpointManager
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition

DIMENSION = 8
ROOT_DEGREE = 30 * DIMENSION
REPLACEMENT_WIDTH = 10 * DIMENSION
NUM_SLICES = 10 * DIMENSION
RETAINED_PHANTOMS = 9 * DIMENSION
DEPTH_DLOG_Z = 1e-3


def _normal_log_density(x, mean, covariance):
    """Evaluate one multivariate-normal log density in host arithmetic."""
    displacement = x - mean
    _, log_determinant = np.linalg.slogdet(covariance)
    quadratic = displacement @ np.linalg.solve(covariance, displacement)
    return (
        -0.5 * mean.size * np.log(2.0 * np.pi)
        - 0.5 * log_determinant
        - 0.5 * quadratic
    )


def _normal_log_density_batch(x, mean, covariance):
    """Evaluate one multivariate-normal log density for many rows."""
    displacement = x - mean  # [N, D]
    _, log_determinant = np.linalg.slogdet(covariance)
    precision = np.linalg.inv(covariance)  # [D, D]
    quadratic = np.einsum(
        "...i,ij,...j->...",
        displacement,
        precision,
        displacement,
    )  # [N]
    return (
        -0.5 * mean.size * np.log(2.0 * np.pi)
        - 0.5 * log_determinant
        - 0.5 * quadratic
    )


def _set_geometry(
        separation: float,
        spike_scale: float,
        prior_variance: float,
        equal_evidence: bool,
) -> None:
    """Set the SS8 geometry and optionally balance component evidence."""
    means = np.zeros((2, DIMENSION), dtype=np.float64)  # [mode, D]
    means[0, 0] = separation
    means[1, 0] = -separation
    cases.SS8_COMPONENT_MEANS = jnp.asarray(
        means,
        dtype=cases.PAPER_DTYPE,
    )
    slab_covariance = jnp.asarray(
        cases.G8_LIKELIHOOD_COVARIANCE,
        dtype=cases.PAPER_DTYPE,
    )  # [D, D]
    cases.SS8_COMPONENT_COVARIANCES = jnp.stack([
        spike_scale * slab_covariance,
        slab_covariance,
    ])  # [mode, D, D]
    cases.SS8_PRIOR_COVARIANCE = (
        prior_variance
        * jnp.eye(DIMENSION, dtype=cases.PAPER_DTYPE)
    )  # [D, D]
    if equal_evidence:
        # Weight inversely by each unweighted component integral.  Normalising
        # the two factors to sum to two preserves the former overall amplitude
        # scale while making the exact evidence contributions equal.
        prior_mean = np.asarray(cases.SS8_PRIOR_MEAN)  # [D]
        prior_covariance = np.asarray(
            cases.SS8_PRIOR_COVARIANCE
        )  # [D, D]
        component_means = np.asarray(cases.SS8_COMPONENT_MEANS)  # [mode, D]
        component_covariances = np.asarray(
            cases.SS8_COMPONENT_COVARIANCES
        )  # [mode, D, D]
        integrals = np.asarray([
            np.exp(_normal_log_density(
                mean,
                prior_mean,
                prior_covariance + covariance,
            ))
            for mean, covariance in zip(
                component_means,
                component_covariances,
                strict=True,
            )
        ])  # [mode]
        inverse_integrals = 1.0 / integrals  # [mode]
        weights = (
            integrals.size
            * inverse_integrals
            / np.sum(inverse_integrals)
        )  # [mode]
    else:
        weights = np.ones(2, dtype=np.float64)  # [mode]
    cases.SS8_WEIGHTS = jnp.asarray(weights, dtype=cases.PAPER_DTYPE)


def _component_evidence() -> tuple[np.ndarray, np.ndarray]:
    """Return each exact component integral and its share of total evidence."""
    prior_mean = np.asarray(cases.SS8_PRIOR_MEAN)  # [D]
    prior_covariance = np.asarray(cases.SS8_PRIOR_COVARIANCE)  # [D, D]
    means = np.asarray(cases.SS8_COMPONENT_MEANS)  # [mode, D]
    covariances = np.asarray(cases.SS8_COMPONENT_COVARIANCES)  # [mode, D, D]
    weights = np.asarray(cases.SS8_WEIGHTS)  # [mode]
    log_integrals = np.asarray([
        np.log(weight) + _normal_log_density(
            mean,
            prior_mean,
            prior_covariance + covariance,
        )
        for weight, mean, covariance in zip(
            weights,
            means,
            covariances,
            strict=True,
        )
    ])  # [mode]
    integrals = np.exp(log_integrals)  # [mode]
    return integrals, integrals / np.sum(integrals)


def _mode_summary(state) -> dict:
    """Measure whether the classic race tree represents the narrow mode."""
    results = state.to_result().trim()
    samples = np.asarray(jax.tree.leaves(results.X_samples)[0])  # [N, D]
    means = np.asarray(cases.SS8_COMPONENT_MEANS)  # [mode, D]
    covariances = np.asarray(cases.SS8_COMPONENT_COVARIANCES)  # [mode, D, D]
    weights = np.asarray(cases.SS8_WEIGHTS)  # [mode]
    component_log_density = np.stack([
        np.log(weight) + _normal_log_density_batch(
            samples,
            mean,
            covariance,
        )
        for weight, mean, covariance in zip(
            weights,
            means,
            covariances,
            strict=True,
        )
    ], axis=1)  # [N, mode]
    assignments = np.argmax(component_log_density, axis=1)  # [N]
    posterior_weights = np.exp(np.asarray(results.log_dp))  # [N]
    component_integrals, component_shares = _component_evidence()
    narrow = assignments == 0  # [N]
    num_samples = int(results.total_num_samples)
    active_lineages = np.asarray(
        results.num_live_points_per_sample
    )[:num_samples]  # [N]
    return {
        "classic_samples": num_samples,
        "goal_loop_iterations": int(state.goal_loop_iter),
        "allocation_loop_iterations": int(state.allocation_loop_iter),
        "maximum_active_lineages": int(np.max(active_lineages)),
        "likelihood_evaluations": int(
            results.total_num_likelihood_evaluations
        ),
        "expected_log_Z": float(state.expected_log_Z_mean),
        "expected_log_Z_uncert": float(state.expected_log_Z_uncert),
        "narrow_classic_samples": int(np.sum(narrow)),
        "narrow_posterior_mass": float(np.sum(posterior_weights[narrow])),
        "narrow_posterior_mass_truth": float(component_shares[0]),
        "component_evidence": component_integrals.tolist(),
        "component_evidence_share": component_shares.tolist(),
        "lost_narrow_log_Z_bias": float(np.log(component_shares[1])),
    }


def _run_to_goal(
        nested_sampler: NestedSampler,
        state,
        target: float,
        seed: int,
        checkpoint_dir: Path,
):
    """Run or resume one state until its expected uncertainty crosses target."""
    started = time.perf_counter()

    def goal_condition(current_state) -> bool:
        if int(current_state.goal_loop_iter) == 0:
            return False
        uncertainty = float(current_state.expected_log_Z_uncert)
        print(
            f"seed {seed}, target {target:.3f}: "
            f"{int(current_state.num_samples):,} classic, "
            f"uncertainty {uncertainty:.6f}",
            flush=True,
        )
        return uncertainty < target

    depth_condition = DepthCondition(
        dlogZ=jnp.log1p(jnp.asarray(DEPTH_DLOG_Z, dtype=jnp.float64)),
    )
    if state is None:
        completed = nested_sampler.run_until_goal(
            goal_condition,
            depth_cond=depth_condition,
            key=jax.random.PRNGKey(seed),
            checkpoint_dir=checkpoint_dir,
        )
    else:
        completed = nested_sampler.resume_until_goal(
            state,
            goal_condition,
            depth_cond=depth_condition,
            checkpoint_dir=checkpoint_dir,
        )
    jax.block_until_ready(completed)
    return completed, time.perf_counter() - started


def _publish_json(path: Path, record: dict) -> None:
    """Atomically publish one completed calibration record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    """Run one separation/seed calibration cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--separation", type=float, required=True)
    parser.add_argument("--spike-scale", type=float, default=0.01)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--equal-evidence", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-state-root",
        type=Path,
        help="Optional canonical 0.05 state root for the unchanged geometry.",
    )
    args = parser.parse_args()
    if args.separation <= 0.0:
        raise ValueError("separation must be positive.")
    if not 0.0 < args.spike_scale < 1.0:
        raise ValueError("spike-scale must be strictly between zero and one.")
    if args.prior_variance <= 0.0:
        raise ValueError("prior-variance must be positive.")
    if args.seed < 0:
        raise ValueError("seed must be non-negative.")

    _set_geometry(
        args.separation,
        args.spike_scale,
        args.prior_variance,
        args.equal_evidence,
    )
    model, truth = cases.build_ss8()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=NUM_SLICES,
        collect_phantom_samples=True,
        max_phantom_samples=RETAINED_PHANTOMS,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=ROOT_DEGREE,
        shell_size=REPLACEMENT_WIDTH,
        collect_phantom_samples=True,
        sampler=sampler,
        allocation_target="uniform",
        unlimited_samples=True,
    )

    source_state = None
    if args.source_state_root is not None:
        source_dir = (
            args.source_state_root
            / "isotropic"
            / "spike_slab"
            / f"seed-{args.seed:02d}"
        )
        with CheckpointManager(source_dir) as manager:
            source_state = manager.load()
        if source_state is None:
            raise FileNotFoundError(f"No source state in {source_dir}.")

    stage_root = (
        args.state_root
        / (
            f"separation-{args.separation:g}"
            f"-scale-{args.spike_scale:g}"
            f"-prior-{args.prior_variance:g}"
            f"-equal-{int(args.equal_evidence)}"
        )
        / f"seed-{args.seed:02d}"
    )
    if source_state is None:
        state_005, seconds_005 = _run_to_goal(
            nested_sampler,
            None,
            0.05,
            args.seed,
            stage_root / "goal-0.05",
        )
    else:
        state_005 = source_state
        seconds_005 = 0.0
    summary_005 = _mode_summary(state_005)
    summary_005["run_seconds"] = seconds_005
    stage_record = {
        "separation": args.separation,
        "spike_scale": args.spike_scale,
        "prior_variance": args.prior_variance,
        "equal_evidence": args.equal_evidence,
        "component_weights": np.asarray(cases.SS8_WEIGHTS).tolist(),
        "seed": args.seed,
        "truth_log_Z": float(truth),
        "goal_0.05": summary_005,
    }
    stage_output = args.output.with_name(
        f"{args.output.stem}-goal-0.05{args.output.suffix}"
    )
    _publish_json(stage_output, stage_record)
    print(json.dumps(stage_record, indent=2, sort_keys=True), flush=True)

    state_002, seconds_002 = _run_to_goal(
        nested_sampler,
        state_005,
        0.02,
        args.seed,
        stage_root / "goal-0.02",
    )
    summary_002 = _mode_summary(state_002)
    summary_002["run_seconds"] = seconds_002

    record = {
        "separation": args.separation,
        "spike_scale": args.spike_scale,
        "prior_variance": args.prior_variance,
        "equal_evidence": args.equal_evidence,
        "component_weights": np.asarray(cases.SS8_WEIGHTS).tolist(),
        "seed": args.seed,
        "truth_log_Z": float(truth),
        "goal_0.05": summary_005,
        "goal_0.02": summary_002,
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
            "cpu_count": os.cpu_count(),
        },
    }
    _publish_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
