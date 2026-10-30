"""Audit new-format raw runs, replay diagnostics, and condition their evidence.

Schema2 records include per-child variable work; schema3 adds controller ledgers.
Historical archives stay
reproducible with their original executed source commits; no field inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def reflection_work_statistics(
    bounce_counts: np.ndarray,
    gradients: np.ndarray,
    realised_steps: np.ndarray,
) -> dict[str, int | float]:
    """Describe retained paths; independently counted gradients guard count mixups."""
    import numpy as np

    if (bounce_counts.ndim != 1 or gradients.shape != bounce_counts.shape
            or realised_steps.shape != (len(bounce_counts), 9) or not len(bounce_counts)):
        raise ValueError("Expected nonempty [N] counts/gradients and [N, 9] realised steps.")
    if not np.array_equal(gradients, 9 * bounce_counts):
        raise ValueError("Reflection counts disagree with independently recorded gradients.")
    durations = realised_steps * bounce_counts[:, None]  # [N, 9]
    return {
        "bounce_count_min": int(bounce_counts.min()),
        "bounce_count_max": int(bounce_counts.max()),
        "mean_bounce_count": float(bounce_counts.mean()),
        "mean_realised_duration": float(durations.mean()),
        "duration_min": float(durations.min()),
        "duration_max": float(durations.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--skip-evidence", action="store_true")
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp
    from scipy.special import logsumexp
    from scipy.stats import norm, wasserstein_distance

    from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
    from benchmarks.issue_312.numpy_allocation_evidence import draw_numpy_evidence
    from benchmarks.issue_312.duration_common import make_arm, numpy_parameters
    from benchmarks.issue_312.tuning_common import (
        SOURCE_ROOT,
        atomic_json,
        atomic_npz,
        diagnostic_correlation,
        make_model,
        numpy_log_likelihood,
    )
    from jaxns.pytree import pytree_ravel

    folder = options.run
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    completed = json.loads((folder / "COMPLETE.json").read_text())
    assert completed["core_complete"]
    online_comparison = manifest.get("comparison") == "online"
    assert manifest["record_schema"] == (3 if online_comparison else 2)
    if not online_comparison and options.schedule is None:
        parser.error("Historical fixed-schedule experiments require --schedule.")
    assert manifest["kernel"] == dict(
        num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=0.003,
    )
    for relative, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((SOURCE_ROOT / relative).read_bytes()).hexdigest() == digest, (
            relative
        )
    schedule_hash = (hashlib.sha256(options.schedule.read_bytes()).hexdigest()
                     if options.schedule is not None else None)
    assert schedule_hash == manifest["schedule_sha256"]
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122799)))

    def likelihood(point):
        return model.log_likelihood(unravel(point))

    schedule = (json.loads(options.schedule.read_text())
                if options.schedule is not None else None)
    peak = manifest["peak"] if online_comparison else schedule["peak"]
    arm = "short" if manifest["arm"] == "tuned" else manifest["arm"]
    if online_comparison:
        from benchmarks.issue_312.online_audit import audit_online_history
        from benchmarks.issue_312.reflection_sampler import ReflectionSampler

        sampler = ReflectionSampler(**manifest["kernel"])
        parameters = None
    else:
        sampler, parameters = make_arm(arm, schedule)

    @jax.jit
    @jax.vmap
    def replay(key, point, value, contour, count, step):
        if online_comparison:
            return sampler.run_chain(key, point, value, contour, likelihood,
                                     contour_num_bounces=count, contour_step_size=step)
        return sampler.run_chain(key, point, value, contour, likelihood)

    @jax.jit
    @jax.vmap
    def trajectory_steps(key, contour, recorded_h):
        if online_comparison:
            base_h = recorded_h
        else:
            _, base_h = parameters(contour)

        def jitter(cycle_key):
            _, _, time_key = jax.random.split(cycle_key, 3)
            return base_h * jax.random.uniform(
                time_key, dtype=jnp.float64, minval=0.8, maxval=1.2
            )

        return jax.vmap(jitter)(jax.random.split(key, 9))

    started = time.monotonic()
    final_budget = manifest["budgets"][-1]
    with np.load(folder / f"calls-{final_budget}.npz") as saved:
        raw = dict(saved)
    with np.load(folder / f"inputs-{final_budget}.npz") as saved_inputs:
        final_phantom_values = saved_inputs["log_L_phantom"]
    controller_audit = (audit_online_history(folder, manifest, raw, options.schedule)
                        if online_comparison else None)
    n = len(raw["samples"])
    roots = raw["seed_indices"] < 0
    indices = np.flatnonzero(~roots)
    accepted = np.full(n, -1, np.int32)
    realised_steps = np.zeros((n, 9))
    replay_evaluations, replay_gradients = 0, 0
    # Match the production width; padding is excluded from all scientific counts.
    # These replay calls are diagnostic overhead, never sampling/evidence budget.
    for start in range(0, len(indices), 150):
        selected = indices[start:start + 150]
        padded = np.pad(selected, (0, 150 - len(selected)), mode="edge")
        seed_idx = raw["seed_indices"][padded]
        result, (_, values) = jax.block_until_ready(
            replay(
                jnp.asarray(raw["chain_keys"][padded]),
                jnp.asarray(raw["unit_samples"][seed_idx]),
                jnp.asarray(raw["log_likelihood"][seed_idx]),
                jnp.asarray(raw["log_L_constraints"][padded]),
                jnp.asarray(raw["bounce_counts"][padded]),
                jnp.asarray(raw["nominal_steps"][padded]),
            )
        )
        size = len(selected)
        np.testing.assert_array_equal(
            np.asarray(result.point)[:size], raw["unit_samples"][selected]
        )
        np.testing.assert_array_equal(
            np.asarray(result.log_likelihood)[:size], raw["log_likelihood"][selected]
        )
        np.testing.assert_array_equal(
            np.asarray(result.evaluations)[:size], raw["evaluations_per_sample"][selected]
        )
        np.testing.assert_array_equal(
            np.asarray(result.gradients)[:size], raw["gradient_evaluations_per_sample"][selected],
        )
        if online_comparison:
            # The preceding chronological audit verified these choices before replay.
            expected_b = raw["bounce_counts"][selected]
            expected_h = raw["nominal_steps"][selected]
        else:
            expected_b, expected_h = numpy_parameters(
                arm, schedule, raw["log_L_constraints"][selected],
            )
        np.testing.assert_array_equal(raw["bounce_counts"][selected], expected_b)
        np.testing.assert_allclose(raw["nominal_steps"][selected], expected_h, rtol=2e-14, atol=0)
        np.testing.assert_array_equal(np.asarray(result.gradients)[:size], 9 * expected_b)
        # Verify that the retained prefix includes every repeated cycle state.
        np.testing.assert_array_equal(
            np.asarray(values)[:size, :8], final_phantom_values[selected]
        )
        accepted[selected] = np.asarray(result.accepted)[:size]
        realised_steps[selected] = np.asarray(
            trajectory_steps(
                jnp.asarray(raw["chain_keys"][padded]),
                jnp.asarray(raw["log_L_constraints"][padded]),
                jnp.asarray(raw["nominal_steps"][padded]),
            )
        )[:size]
        replay_evaluations += int(np.asarray(result.evaluations)[:size].sum())
        replay_gradients += int(np.asarray(result.gradients)[:size].sum())
    replay_seconds = time.monotonic() - started
    diagnostic_path = folder / "chain-diagnostics.npz"
    atomic_npz(
        diagnostic_path, reflection_accepted=accepted, realised_steps=realised_steps,
        bounce_counts=raw["bounce_counts"], nominal_steps=raw["nominal_steps"],
        gradient_evaluations_per_sample=raw["gradient_evaluations_per_sample"],
        realised_durations=realised_steps * raw["bounce_counts"][:, None],
    )
    audit_rows = []
    coordinates = (1, 16, 17, 5)
    quantiles = (np.arange(8192) + 0.5) / 8192
    bins = ((0, 5), (5, 15), (15, 40), (40, 100), (100, 1000), (1000, np.inf))
    for budget in manifest["budgets"]:
        row = json.loads((folder / f"CORE-{budget}.json").read_text())
        archive = folder / f"calls-{budget}.npz"
        input_path = folder / f"inputs-{budget}.npz"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
        assert (
            hashlib.sha256(input_path.read_bytes()).hexdigest() == row["phantom_archive_sha256"]
        )
        with np.load(archive) as loaded:
            saved = dict(loaded)
        count = row["samples"]
        points = saved["samples"]
        values = saved["log_likelihood"]
        constraints = saved["log_L_constraints"]
        seed_idx, parent_idx, batch_idx = (
            saved[name]
            for name in (
                "seed_indices",
                "parent_indices",
                "batch_indices",
            )
        )
        root = seed_idx < 0
        assert points.dtype == np.float64 and points.shape == (count, 18)
        np.testing.assert_array_equal(points, saved["unit_samples"])
        np.testing.assert_array_equal(points, raw["samples"][:count])
        assert np.all((points >= 0) & (points <= 1)) and np.all(values > constraints)
        np.testing.assert_array_equal(root, parent_idx < 0)
        np.testing.assert_array_equal(root, np.isneginf(constraints))
        assert root[:150].all() and root.sum() == row["roots"]
        assert np.all(seed_idx[~root] < np.flatnonzero(~root))
        assert np.all(batch_idx[seed_idx[~root]] < batch_idx[~root])
        np.testing.assert_array_equal(values[parent_idx[~root]], constraints[~root])
        assert np.all(values[seed_idx[~root]] > constraints[~root])
        assert np.all(constraints[seed_idx[~root]] <= constraints[~root])
        np.testing.assert_array_equal(
            saved["out_degrees"], np.bincount(parent_idx[~root], minlength=count)
        )
        evaluations = saved["evaluations_per_sample"]
        assert evaluations.sum() == row["evaluations"] and np.all(evaluations[root] == 1)
        assert row["evaluations_before_last_batch"] < budget <= row["evaluations"]
        assert evaluations[batch_idx == batch_idx.max()].sum() == row["last_batch_evaluations"]
        counts = saved["bounce_counts"]
        gradients = saved["gradient_evaluations_per_sample"]
        np.testing.assert_array_equal(gradients, 9 * counts)
        assert np.all(counts[root] == 0) and np.all(gradients[root] == 0)
        assert np.all(saved["nominal_steps"][root] == 0)
        assert np.all(saved["value_only_evaluations_per_sample"][root] == 1)
        np.testing.assert_array_equal(
            saved["value_only_evaluations_per_sample"], evaluations - gradients,
        )
        assert row["gradient_evaluations"] == gradients.sum()
        assert row["value_only_evaluations"] == (evaluations - gradients).sum()
        assert np.all(evaluations[~root] >= gradients[~root] + 18)
        error = np.max(np.abs(values - numpy_log_likelihood(points)))
        assert error < 5e-5, error
        initial_hash = hashlib.sha256(
            points[:150].tobytes() + values[:150].tobytes()
        ).hexdigest()
        assert initial_hash == row["initial_sha256"]
        inputs = EvidenceInputs.load(input_path)
        np.testing.assert_array_equal(np.asarray(inputs.log_L_classic)[:count], values)
        np.testing.assert_array_equal(np.asarray(inputs.log_L_constraints)[:count], constraints)
        valid = np.asarray(inputs.valid_phantom)
        np.testing.assert_array_equal(valid[:count], ~root)
        assert not valid[count:].any() and inputs.log_L_phantom.shape[1] == 8
        assert np.all(
            np.asarray(inputs.log_L_phantom)[valid]
            > np.asarray(inputs.log_L_constraints)[valid, None]
        )
        block = inputs.block_state
        block_valid = np.asarray(block.valid)
        unique, inverse, block_counts = np.unique(
            values, return_inverse=True, return_counts=True,
        )
        outgoing = np.bincount(inverse, weights=saved["out_degrees"]).astype(np.int64)
        incoming = row["roots"] + np.concatenate([[0], np.cumsum(outgoing - block_counts)[:-1]])
        np.testing.assert_array_equal(np.asarray(block.log_L_blocks)[block_valid], unique)
        np.testing.assert_array_equal(np.asarray(block.block_size)[block_valid], block_counts)
        np.testing.assert_array_equal(np.asarray(block.block_out_degree)[block_valid], outgoing)
        np.testing.assert_array_equal(np.asarray(block.incoming_K)[block_valid], incoming)
        weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
        mean = np.sum(points * weights[:, None], axis=0)
        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
        np.testing.assert_allclose(mean, row["mean"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(std, row["std"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(1 / np.sum(weights**2), row["kish_ess"])
        metrics = {}
        for i in coordinates:
            target = norm.ppf(quantiles, 0.5, 0.05) if i == 5 else quantiles
            order = np.argsort(points[:, i])
            cdf = np.cumsum(weights[order])
            truth = norm.cdf(points[order, i], 0.5, 0.05) if i == 5 else points[order, i]
            metrics[f"w1_x{i}"] = float(
                wasserstein_distance(points[:, i], target, u_weights=weights)
            )
            metrics[f"cdf_error_x{i}"] = float(
                max(np.max(abs(cdf - truth)), np.max(abs(cdf - weights[order] - truth)))
            )
        movement = []
        gaps = peak - constraints
        for low, high in bins:
            select = (~root) & (gaps > low) & (gaps <= high)
            starts, ends = points[seed_idx[select]], points[select]
            number = int(select.sum())
            item = {
                "gap_low": low,
                "gap_high": high if np.isfinite(high) else None,
                "pairs": number,
            }
            if number:
                item["reflection_accepted"] = int(accepted[:count][select].sum())
                item["reflection_attempted"] = 9 * number
                item["acceptance"] = item["reflection_accepted"] / (9 * number)
                item["mean_realised_h"] = float(realised_steps[:count][select].mean())
                item["h_min"] = float(realised_steps[:count][select].min())
                item["h_max"] = float(realised_steps[:count][select].max())
                item["gradient_evaluations"] = int(gradients[select].sum())
                item.update(reflection_work_statistics(
                    counts[select], gradients[select], realised_steps[:count][select],
                ))
            if number >= 30:
                for i in range(18) if online_comparison else coordinates:
                    rms = float(np.sqrt(np.mean((ends[:, i] - starts[:, i]) ** 2)))
                    item[f"rms_dx{i}"] = rms
                    squared_sum = float(np.sum((ends[:, i] - starts[:, i]) ** 2))
                    item[f"squared_displacement_sum_x{i}"] = squared_sum
                    item[f"esjd_dx{i}_per_gradient"] = squared_sum / int(gradients[select].sum())
                    item[f"corr_x{i}"] = diagnostic_correlation(starts[:, i], ends[:, i])
                    if online_comparison:
                        item[f"esjd_dx{i}_per_total_call"] = squared_sum / int(
                            evaluations[select].sum(),
                        )
                        item[f"squared_coordinate_jump_sum_x{i}"] = float(np.sum(
                            (ends[:, i]**2 - starts[:, i]**2)**2,
                        ))
                        item[f"corr_squared_x{i}"] = diagnostic_correlation(
                            starts[:, i]**2, ends[:, i]**2,
                        )
            movement.append(item)
        log_tail = float(
            peak
            + saved["log_volume"][np.argmax(values)]
            - row["log_Z"]
            - 0.5 * row["log_Z_uncert"] ** 2
        )
        # Early snapshots can have an enormous valid upper bound. Preserve its
        # log value; JSON null plus a flag denotes linear overflow, never clipping.
        tail_overflow = log_tail > np.log(np.finfo(np.float64).max)
        tail = None if tail_overflow else float(np.exp(log_tail))
        analysis = {
            **row,
            **metrics,
            "movement": movement,
            "audit_passed": True,
            "max_likelihood_discrepancy": float(error),
            "classic_tail_upper_fraction": tail,
            "classic_tail_log_upper_fraction": log_tail,
            "classic_tail_upper_fraction_overflow": bool(tail_overflow),
            "replay_exact": True,
            "replay_evaluations_full_run": replay_evaluations,
            "replay_gradients_full_run": replay_gradients,
            "replay_seconds_full_run": replay_seconds,
            "schedule_sha256": schedule_hash,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
        if online_comparison:
            analysis["controller_audit"] = controller_audit["snapshots"][str(budget)]
            analysis["controller_audit_seconds_full_run"] = controller_audit["seconds"]
        if not options.skip_evidence:
            duration_comparison = manifest.get("comparison", "tuning") == "duration"
            key = jax.random.PRNGKey(
                3122902 if online_comparison else 3122713 if duration_comparison else 3122705,
            )
            for value in (
                manifest["seed"],
                budget,
                ("uniform", "evidence_improving").index(manifest["allocation"]),
                (("fixed", "online", "frozen") if online_comparison else
                 ("fixed", "short", "duration") if duration_comparison else
                 ("fixed", "tuned")).index(manifest["arm"]),
            ):
                key = jax.random.fold_in(key, value)
            draws = {}
            analysis.update(mc_draws=4096, C_min=20, mc_batch_size=64)
            for mode, condition in (("classic_mc", False), ("phantom", True)):
                evidence_start = time.monotonic()
                entropy = np.asarray(jax.random.fold_in(key, int(condition))).tolist()
                ensemble = draw_numpy_evidence(
                    inputs, np.random.SeedSequence(entropy), phantoms=condition, draws=4096
                )
                z = np.asarray(ensemble.log_Z_samples)
                assert z.shape == (4096,) and np.isfinite(z).all()
                draws[f"{mode}_log_Z_samples"] = z
                analysis.update(
                    {
                        f"{mode}_log_Z": float(z.mean()),
                        f"{mode}_log_Z_uncert": float(z.std()),
                        f"{mode}_mcse": float(z.std(ddof=1) / 64),
                        f"{mode}_percentile95": np.quantile(z, [0.025, 0.975]).tolist(),
                        f"{mode}_seconds": time.monotonic() - evidence_start,
                        f"{mode}_entropy": entropy,
                    }
                )
                if condition:
                    draws["phantom_gate_active"] = ensemble.phantom_gate_active
                    draws["kish_participating_cluster_counts"] = (
                        ensemble.kish_participating_cluster_counts
                    )
                    draws["phantom_log_dZ_mean"] = ensemble.log_dZ_mean
            draw_path = folder / f"evidence-{budget}.npz"
            atomic_npz(draw_path, **draws)
            analysis["evidence_sha256"] = hashlib.sha256(draw_path.read_bytes()).hexdigest()
        atomic_json(folder / f"ANALYSIS-{budget}.json", analysis)
        audit_rows.append(analysis)
        print(
            f"audit passed: {manifest['arm']} {manifest['allocation']} "
            f"seed={manifest['seed']} budget={budget}",
            flush=True,
        )
    atomic_json(
        folder / "ANALYSIS-COMPLETE.json",
        {
            "passed": True,
            "evidence_complete": not options.skip_evidence,
            "records": audit_rows,
            "seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "analysis_source_sha256": {
                name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "benchmarks/issue_312/analyse_reflection_tuning.py",
                    "benchmarks/issue_312/numpy_allocation_evidence.py",
                    "benchmarks/issue_312/duration_common.py",
                    "benchmarks/issue_312/duration_schedule.py",
                ) + (("benchmarks/issue_312/online_audit.py",) if online_comparison else ())
            },
        },
    )


if __name__ == "__main__":
    main()
