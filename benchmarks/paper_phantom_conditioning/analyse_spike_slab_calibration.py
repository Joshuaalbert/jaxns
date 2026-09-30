"""Evaluate classic and phantom-prefix evidence for one SS8 calibration seed."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.paper_phantom_conditioning import cases
from benchmarks.paper_phantom_conditioning.prefix_sweep import (
    sample_phantom_prefix_sweep_reference,
)
from jaxns.checkpoint import CheckpointManager
from jaxns.results import _incoming_lineages_per_sample

DIMENSION = 8
NUM_GROUPS = 9
MC_DRAWS = 2048
EVIDENCE_CAPACITY_QUANTUM = 8192


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


def _set_geometry(
        separation: float,
        spike_scale: float,
        prior_variance: float,
        equal_evidence: bool,
) -> None:
    """Set the model constants used to compute the exact reference evidence."""
    means = np.zeros((2, DIMENSION), dtype=np.float64)  # [mode, D]
    means[0, 0] = separation
    means[1, 0] = -separation
    cases.SS8_COMPONENT_MEANS = jax.numpy.asarray(
        means,
        dtype=cases.PAPER_DTYPE,
    )
    slab_covariance = jax.numpy.asarray(
        cases.G8_LIKELIHOOD_COVARIANCE,
        dtype=cases.PAPER_DTYPE,
    )  # [D, D]
    cases.SS8_COMPONENT_COVARIANCES = jax.numpy.stack([
        spike_scale * slab_covariance,
        slab_covariance,
    ])  # [mode, D, D]
    cases.SS8_PRIOR_COVARIANCE = (
        prior_variance
        * jax.numpy.eye(DIMENSION, dtype=cases.PAPER_DTYPE)
    )  # [D, D]
    if equal_evidence:
        # Match the generation model exactly: inverse-integral weighting gives
        # both likelihood components one half of the analytic evidence.
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
    cases.SS8_WEIGHTS = jax.numpy.asarray(
        weights,
        dtype=cases.PAPER_DTYPE,
    )


def _analysis_results(state):
    """Trim unused capacity and restore a small reusable padding quantum."""
    state = state.trim()
    num_samples = int(state.num_samples)
    capacity = (
        (num_samples + EVIDENCE_CAPACITY_QUANTUM - 1)
        // EVIDENCE_CAPACITY_QUANTUM
        * EVIDENCE_CAPACITY_QUANTUM
    )
    return state.resize(capacity).to_result(), capacity


def main() -> None:
    """Load one 0.02 state and write its paired evidence-prefix result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--separation", type=float, required=True)
    parser.add_argument("--spike-scale", type=float, default=0.01)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--equal-evidence", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mc-workers", type=int, default=16)
    parser.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    args = parser.parse_args()
    if args.mc_workers < 1:
        raise ValueError("mc-workers must be positive.")
    if args.mc_draws < 2:
        raise ValueError("mc-draws must be at least two.")
    if not 0.0 < args.spike_scale < 1.0:
        raise ValueError("spike-scale must be strictly between zero and one.")
    if args.prior_variance <= 0.0:
        raise ValueError("prior-variance must be positive.")

    _set_geometry(
        args.separation,
        args.spike_scale,
        args.prior_variance,
        args.equal_evidence,
    )
    _, truth = cases.build_ss8()
    checkpoint_dir = (
        args.state_root
        / (
            f"separation-{args.separation:g}"
            f"-scale-{args.spike_scale:g}"
            f"-prior-{args.prior_variance:g}"
            f"-equal-{int(args.equal_evidence)}"
        )
        / f"seed-{args.seed:02d}"
        / "goal-0.02"
    )
    with CheckpointManager(checkpoint_dir) as manager:
        state = manager.load()
    if state is None:
        raise FileNotFoundError(f"No completed state in {checkpoint_dir}.")

    results, capacity = _analysis_results(state)
    valid_blocks = np.asarray(results.block_data.valid)  # [G_capacity]
    if not np.all(np.asarray(results.block_data.size)[valid_blocks] == 1):
        raise RuntimeError("SS8 calibration unexpectedly contains a plateau.")
    started = time.perf_counter()
    log_Z, gates = sample_phantom_prefix_sweep_reference(
        key=jax.random.fold_in(jax.random.PRNGKey(args.seed), 1),
        log_L_constraints=results.log_L_constraints,
        K_classic=_incoming_lineages_per_sample(results),
        valid_phantom=results.valid_phantom,
        log_L_phantom=results.log_L_phantom,
        num_samples=results.total_num_samples,
        block_state=results.block_data.to_block_state(),
        dimension=DIMENSION,
        num_draws=args.mc_draws,
        batch_size=1,
        num_workers=args.mc_workers,
        num_groups=NUM_GROUPS,
    )
    evidence = {}
    for prefix in range(NUM_GROUPS + 1):
        draws = log_Z[:, prefix]  # [M]
        evidence[str(prefix)] = {
            "log_Z_mean": float(np.mean(draws)),
            "log_Z_uncert": float(np.std(draws, ddof=1)),
            "log_Z_bias": float(np.mean(draws) - float(truth)),
            "gate_active_fraction": (
                0.0
                if prefix == 0
                else float(np.mean(gates[prefix - 1, valid_blocks]))
            ),
        }
    record = {
        "separation": args.separation,
        "spike_scale": args.spike_scale,
        "prior_variance": args.prior_variance,
        "equal_evidence": args.equal_evidence,
        "component_weights": np.asarray(cases.SS8_WEIGHTS).tolist(),
        "seed": args.seed,
        "goal_log_Z_uncert": 0.02,
        "truth_log_Z": float(truth),
        "classic_samples": int(results.total_num_samples),
        "goal_loop_iterations": int(state.goal_loop_iter),
        "allocation_loop_iterations": int(state.allocation_loop_iter),
        "maximum_active_lineages": int(np.max(
            np.asarray(_incoming_lineages_per_sample(results))[valid_blocks]
        )),
        "likelihood_evaluations": int(
            results.total_num_likelihood_evaluations
        ),
        "evidence_capacity": capacity,
        "mc_draws": args.mc_draws,
        "mc_workers": args.mc_workers,
        "analysis_seconds": time.perf_counter() - started,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
