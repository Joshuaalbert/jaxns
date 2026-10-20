"""Separate slice-bracket cost, disconnected-line mixing, and cube-face bounds."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

# Checkpoints made by run_ridge.py refer to __main__.prior_model. Keep that
# explicit symbol available while loading those locally generated states.
from run_ridge import prior_model  # noqa: F401


def shrink_draws(half_width, bracket_half_width, distant_centre=None):
    """Apply the production sign-based shrink rule to simple known line slices."""
    rng = np.random.default_rng(312)
    count = 100_000
    left = np.full(count, -bracket_half_width)
    right = np.full(count, bracket_half_width)
    accepted = np.zeros(count)
    evaluations = np.zeros(count, dtype=int)
    active = np.arange(count)
    for _ in range(1000):
        proposal = rng.uniform(left[active], right[active])
        valid = np.abs(proposal) < half_width
        if distant_centre is not None:
            valid |= np.abs(proposal - distant_centre) < half_width
        evaluations[active] += 1
        accepted[active[valid]] = proposal[valid]
        rejected = active[~valid]
        rejected_proposal = proposal[~valid]
        left[rejected] = np.where(
            rejected_proposal < 0, rejected_proposal, left[rejected],
        )
        right[rejected] = np.where(
            rejected_proposal > 0, rejected_proposal, right[rejected],
        )
        active = rejected
        if active.size == 0:
            break
    else:
        raise RuntimeError("Shrink diagnostic exceeded 1000 proposals.")
    return {
        "draws": count,
        "half_width": half_width,
        "bracket_half_width": bracket_half_width,
        "mean_evaluations": float(evaluations.mean()),
        "accepted_std": float(accepted.std()),
        "distant_interval_fraction": (
            None if distant_centre is None
            else float(np.mean(np.abs(accepted - distant_centre) < half_width))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    from jax import numpy as jnp

    from jaxns.checkpoint import CheckpointManager
    from jaxns.pytree import TreeField
    from jaxns.sampling.slice import _new_proposal, _slice_bounds

    bounds = []
    for point, direction, expected in (
        ([0.25, 0.5], [1.0, 0.1], [-0.25, 0.75]),
        ([0.0, 0.5], [1.0, 0.1], [0.0, 1.0]),
        ([1.0, 0.5], [-1.0, 0.1], [0.0, 1.0]),
    ):
        anchor = TreeField(jnp.asarray(point))
        heading = TreeField(jnp.asarray(direction))
        actual = _slice_bounds(anchor, heading)
        child, _, calls, _, _ = _new_proposal(
            jax.random.PRNGKey(0), anchor, heading, jnp.asarray(0.0),
            lambda value: jnp.asarray(1.0),
        )
        bounds.append({
            "point": point, "direction": direction, "expected_bounds": expected,
            "actual_bounds": [float(value) for value in actual],
            "constant_likelihood_child": np.asarray(child.tree).tolist(),
            "evaluations": int(calls),
        })
    checkpoint_boundaries = []
    for directory in options.checkpoint_dir:
        with CheckpointManager(directory) as manager:
            state = manager.load()
        if state is None:
            raise ValueError(f"No checkpoint found in {directory}.")
        count = int(state.num_samples)
        points = np.concatenate([
            np.asarray(leaf)[:count].reshape(count, -1)
            for leaf in jax.tree.leaves(state.samples.U_samples)
        ], axis=1)
        result = state.to_result()
        result_count = int(result.total_num_samples)
        result_points = np.concatenate([
            np.asarray(leaf)[:result_count].reshape(result_count, -1)
            for leaf in jax.tree.leaves(result.U_samples)
        ], axis=1)
        on_boundary = (result_points == 0) | (result_points == 1)
        boundary_rows = np.flatnonzero(np.any(on_boundary, axis=1))
        log_weights = np.asarray(result.log_dp)[:result_count]
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        checkpoint_boundaries.append({
            "directory": str(directory), "samples": count,
            "rows_on_boundary": int(np.sum(np.any((points == 0) | (points == 1), axis=1))),
            "rows_outside_cube": int(np.sum(np.any((points < 0) | (points > 1), axis=1))),
            "posterior_boundary_weight": float(weights[boundary_rows].sum()),
            "boundary_rows": [
                {
                    "dimensions": np.flatnonzero(on_boundary[row]).tolist(),
                    "log_L": float(result.log_L[row]),
                    "posterior_weight": float(weights[row]),
                } for row in boundary_rows
            ],
        })
    covariance = np.array([[0.3 ** 2 / 2 + 1e-10, -0.3 / (2 * np.pi)],
                           [-0.3 / (2 * np.pi), 1 / 12]])
    output = {
        "production_bounds": bounds,
        "connected_full_cube": shrink_draws(1e-5, 0.5),
        "connected_tighter_bracket": shrink_draws(1e-5, 1e-4),
        "disconnected_full_cube": shrink_draws(1e-3, 0.5, 0.3),
        "toy_global_covariance_eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
        "checkpoint_boundaries": checkpoint_boundaries,
    }
    options.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
