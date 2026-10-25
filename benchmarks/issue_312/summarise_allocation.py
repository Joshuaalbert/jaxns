"""Audit matched-work allocation runs, including actual seed-to-child movement."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, wasserstein_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("/tmp"))
    options = parser.parse_args()
    folder = Path(__file__).resolve().parent
    allocations = ("uniform", "evidence_improving", "posterior_improving")
    coordinates = (1, 16, 17, 5)
    intervals = ((0, 5), (5, 15), (15, 40), (40, 100), (100, 1000), (1000, np.inf))
    rows, movement_rows = [], []
    initial_hashes, snapshots = {}, {}
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    quantiles = (np.arange(8192) + 0.5) / 8192
    maximum_error = 0.0
    for kernel in ("hessian", "reflection"):
        for allocation in allocations:
            for letter in "ab":
                stem = f"allocation-{kernel}-{allocation}-{letter}"
                raw = folder / f"raw/{stem}.jsonl"
                for line in raw.read_text().splitlines():
                    row = json.loads(line)
                    assert row["kernel"] == kernel and row["allocation"] == allocation
                    assert row["initial_roots"] == row["replacement_width"] == row["delta_K"] == 150
                    assert row["budgets"] == [2000000, 4000000]
                    assert row["phantoms"] == 0 and row["unit_dtype"] == "float64"
                    assert row["unlimited_samples"] and not row["scientific_goal_claimed"]
                    assert row["termination_reason"] == 0 and row["sigma_stiff"] == 1e-5
                    assert row["curvature"] == 0.3 and row["dlogZ"] == float(np.log1p(1e-3))
                    assert (
                        row["evaluations_before_last_batch"] < row["budget"] <= row["evaluations"]
                    )
                    assert row["last_batch_evaluations"] == (
                        row["evaluations"] - row["evaluations_before_last_batch"]
                    )
                    expected_config = (
                        {"num_slices": 90, "scale_to_contour": True, "correct_mh": True}
                        if kernel == "hessian" else {
                            "num_cycles": 9, "slices_per_cycle": 1, "num_bounces": 10,
                            "step_size": 0.003, "boundary_only": False,
                        }
                    )
                    assert row["kernel_config"] == expected_config
                    for source, digest in row["source_sha256"].items():
                        path = (folder.parents[1] / source
                                if source.startswith("src/") else folder / source)
                        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                    archive = (
                        options.archive_root / f"jaxns-{stem}" / f"seed-{row['seed']}"
                        / f"calls-{row['budget']}.npz"
                    )
                    assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
                    with np.load(archive) as saved:
                        points, values = saved["samples"], saved["log_likelihood"]
                        seeds, parents = saved["seed_indices"], saved["parent_indices"]
                        constraints, batches = saved["log_L_constraints"], saved["batch_indices"]
                        evaluations = saved["evaluations_per_sample"]
                        assert points.dtype == np.float64 and points.shape == (row["samples"], 18)
                        np.testing.assert_array_equal(points, saved["unit_samples"])
                        assert np.all((points >= 0) & (points <= 1))
                        assert np.all(values > constraints)
                        root = seeds < 0
                        np.testing.assert_array_equal(root, parents < 0)
                        np.testing.assert_array_equal(root, np.isneginf(constraints))
                        assert root.sum() == row["roots"] and root[:150].all()
                        assert np.all(seeds[~root] < np.flatnonzero(~root))
                        assert np.all(batches[seeds[~root]] < batches[~root])
                        np.testing.assert_array_equal(values[parents[~root]], constraints[~root])
                        assert np.all(values[seeds[~root]] > constraints[~root])
                        assert np.all(constraints[seeds[~root]] <= constraints[~root])
                        assert evaluations.sum() == row["evaluations"]
                        assert np.all(evaluations[root] == 1)
                        last = batches == batches.max()
                        assert evaluations[last].sum() == row["last_batch_evaluations"]
                        assert row["gradient_evaluations"] == (~root).sum() * (
                            90 if kernel == "reflection" else 0
                        )
                        assert row["evaluations"] == (
                            row["value_only_evaluations"] + row["gradient_evaluations"]
                        )
                        digest = hashlib.sha256(
                            points[:150].tobytes() + values[:150].tobytes(),
                        ).hexdigest()
                        assert digest == row["initial_sha256"]
                        initial_hashes.setdefault(row["seed"], set()).add(digest)
                        residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
                        reference = peak - 0.5 * (
                            (residual / 1e-5) ** 2
                            + np.sum(((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
                        )
                        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
                        maximum_error = max(maximum_error, float(np.max(abs(values - reference))))
                        np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
                        # Additional finite-budget audit: bound the tail using
                        # the known peak and the classic expected volume. Its
                        # lognormal moment conversion gives log E[Z] = mu+s²/2.
                        # This is conditional on the shrinkage model, not a
                        # bound robust to poor constrained-prior mixing.
                        log_expected_z = row["log_Z"] + 0.5 * row["log_Z_uncert"] ** 2
                        row["classic_tail_upper_fraction"] = float(np.exp(
                            peak + saved["log_volume"][np.argmax(values)] - log_expected_z,
                        ))
                        weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                        mean = np.sum(points * weights[:, None], axis=0)
                        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
                        np.testing.assert_allclose(1 / np.sum(weights ** 2), row["kish_ess"])
                        for coordinate in coordinates:
                            np.testing.assert_allclose(std[coordinate], row[f"std_x{coordinate}"])
                            target = (norm.ppf(quantiles, 0.5, 0.05)
                                      if coordinate == 5 else quantiles)
                            row[f"w1_x{coordinate}"] = float(wasserstein_distance(
                                points[:, coordinate], target, u_weights=weights,
                            ))
                        gaps = peak - constraints
                        for low, high in intervals:
                            selected = (~root) & (gaps > low) & (gaps <= high)
                            starts, ends = points[seeds[selected]], points[selected]
                            movement = {
                                "kernel": kernel, "allocation": allocation, "seed": row["seed"],
                                "budget": row["budget"], "gap_low": low,
                                "gap_high": high if np.isfinite(high) else None,
                                "pairs": int(selected.sum()),
                            }
                            for coordinate in coordinates:
                                start, end = starts[:, coordinate], ends[:, coordinate]
                                enough = len(start) >= 30
                                movement[f"rms_dx{coordinate}"] = (
                                    float(np.sqrt(np.mean((end - start) ** 2))) if enough else None
                                )
                                spread = len(start) > 1 and start.std() > 0 and end.std() > 0
                                movement[f"corr_x{coordinate}"] = (
                                    float(np.corrcoef(start, end)[0, 1])
                                    if enough and spread else None
                                )
                            movement_rows.append(movement)
                        if row["budget"] == 4000000:
                            snapshots[kernel, allocation, row["seed"]] = (points, weights)
                    rows.append(row)
    assert len(rows) == 360 and len(snapshots) == 180
    assert sorted(initial_hashes) == list(range(30))
    assert all(len(hashes) == 1 for hashes in initial_hashes.values())

    summary, bootstraps = {}, {}
    indices = np.random.default_rng(312043).integers(0, 30, (10000, 30))
    for kernel in ("hessian", "reflection"):
        for budget in (2000000, 4000000):
            for allocation in allocations:
                selected = sorted(
                    [row for row in rows if row["kernel"] == kernel
                     and row["budget"] == budget and row["allocation"] == allocation],
                    key=lambda row: row["seed"],
                )
                assert [row["seed"] for row in selected] == list(range(30))
                z = np.array([row["log_Z"] for row in selected])
                sd = np.array([row["log_Z_uncert"] for row in selected])
                bootstrap = np.sqrt(np.mean(z[indices] ** 2, axis=1))
                bootstraps[kernel, budget, allocation] = bootstrap
                item = {
                    "seeds": 30, "mean_log_Z": float(z.mean()),
                    "rmse": float(np.sqrt(np.mean(z*z))),
                    "rmse_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "mean_quoted_sd": float(sd.mean()),
                    "within_two_quoted_sd": int(np.sum(abs(z) <= 2 * sd)),
                    "mean_evaluations": float(np.mean([row["evaluations"] for row in selected])),
                    "max_budget_overshoot": max(row["evaluations"] - budget for row in selected),
                    "mean_gradient_evaluations": float(np.mean(
                        [row["gradient_evaluations"] for row in selected],
                    )),
                    "median_ess": float(np.median([row["kish_ess"] for row in selected])),
                    "median_roots": float(np.median([row["roots"] for row in selected])),
                    "median_peak_gap": float(np.median([row["peak_gap"] for row in selected])),
                    "max_classic_tail_upper_fraction": max(
                        row["classic_tail_upper_fraction"] for row in selected
                    ),
                    "runs_with_completed_depth": sum(
                        row["completed_epochs"] > 0 for row in selected
                    ),
                    "median_widths": {
                        f"x{i}": float(np.median([row[f"std_x{i}"] for row in selected]))
                        for i in coordinates
                    },
                    "median_wasserstein": {
                        f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                        for i in coordinates
                    },
                    "movement": {},
                }
                for low, _ in intervals:
                    moves = [move for move in movement_rows if move["kernel"] == kernel
                             and move["allocation"] == allocation and move["budget"] == budget
                             and move["gap_low"] == low]
                    metrics = {"median_pairs": float(np.median([move["pairs"] for move in moves]))}
                    for coordinate in coordinates:
                        for prefix in ("rms_d", "corr_"):
                            field = f"{prefix}x{coordinate}"
                            values = [move[field] for move in moves if move[field] is not None]
                            metrics[field] = float(np.median(values)) if values else None
                            metrics[f"runs_{field}"] = len(values)
                    item["movement"][str(low)] = metrics
                summary[f"{kernel}_{allocation}_{budget}"] = item
            for allocation in allocations[1:]:
                difference = (bootstraps[kernel, budget, allocation]
                              - bootstraps[kernel, budget, "uniform"])
                interval_name = f"{kernel}_{allocation}_minus_uniform_{budget}_rmse_95_percent"
                summary[interval_name] = np.quantile(
                    difference, [0.025, 0.975],
                ).tolist()
    summary["validation"] = {
        "archives": len(rows), "initial_populations": len(initial_hashes),
        "maximum_likelihood_discrepancy": maximum_error,
        "bootstrap_draws": 10000, "bootstrap_seed": 312043,
        "movement_minimum_pairs_per_run_bin": 30,
    }
    (folder / "allocation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = (
        "kernel", "allocation", "seed", "budget", "evaluations", "gradient_evaluations",
        "log_Z", "log_Z_uncert", "kish_ess", "roots", "completed_epochs", "peak_gap",
        "std_x1", "std_x16", "std_x17", "std_x5", "w1_x1", "w1_x16", "w1_x17", "w1_x5",
        "initial_sha256", "archive_sha256",
        "classic_tail_upper_fraction",
    )
    for filename, selected, fieldnames in (
        ("allocation_summary.csv", rows, fields),
        ("allocation_movement.csv", movement_rows, list(movement_rows[0])),
    ):
        with (folder / filename).open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(selected)
    for kernel in ("hessian", "reflection"):
        for allocation in allocations:
            name = f"{kernel}_{allocation}_4000000"
            print(json.dumps({"group": name, **{
                key: value for key, value in summary[name].items() if key != "movement"
            }}))
    print(json.dumps(summary["validation"]))

    colours = ("#777777", "#287fa4", "#bc6b26")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for r, kernel in enumerate(("hessian", "reflection")):
        for a, (allocation, colour) in enumerate(zip(allocations, colours, strict=True)):
            primary = summary[f"{kernel}_{allocation}_4000000"]
            selected = sorted([row for row in rows if row["kernel"] == kernel
                               and row["allocation"] == allocation and row["budget"] == 4000000],
                              key=lambda row: row["seed"])
            axes[r, 0].errorbar(
                np.arange(30) + (a-1)*0.2, [row["log_Z"] for row in selected],
                yerr=[row["log_Z_uncert"] for row in selected], fmt=".", color=colour,
                label=allocation.removesuffix("_improving"),
            )
            axes[r, 1].plot(
                [2, 4], [summary[f"{kernel}_{allocation}_{b}"]["rmse"] for b in (2000000, 4000000)],
                "o-", color=colour, label=allocation.removesuffix("_improving"),
            )
            axes[r, 2].plot(
                range(4), [primary["median_widths"][f"x{i}"] / (0.05 if i == 5 else np.sqrt(1/12))
                           for i in coordinates], "o-", color=colour,
            )
        axes[r, 0].axhline(0, color="black", ls="--", lw=1)
        axes[r, 0].set(title=kernel.title(), xlabel="Seed", ylabel="Log evidence (truth ≈ 0)")
        axes[r, 0].legend(fontsize=8)
        axes[r, 1].set(
            xlabel="Model-call budget (millions)", ylabel="Log-evidence RMSE", xticks=[2, 4],
        )
        axes[r, 2].axhline(1, color="black", ls="--", lw=1)
        axes[r, 2].set(xticks=range(4), xticklabels=[f"x{i}" for i in coordinates],
                       ylabel="Median posterior SD / true SD at 4M", ylim=(0, 1.1))
    fig.suptitle(
        "Allocation comparison: 30 matched seeds per cell, identical kernels and work budgets",
    )
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"allocation_comparison.{extension}", dpi=160)

    for kernel in ("hessian", "reflection"):
        fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
        for r, (allocation, colour) in enumerate(zip(allocations, colours, strict=True)):
            for seed in range(30):
                points, weights = snapshots[kernel, allocation, seed]
                for c, coordinate in enumerate(coordinates):
                    order = np.argsort(points[:, coordinate])
                    axes[r, c].plot(points[order, coordinate], np.cumsum(weights[order]),
                                    color=colour, alpha=0.3, lw=0.7)
            for c, coordinate in enumerate(coordinates):
                grid = np.linspace(0.3, 0.7, 200) if coordinate == 5 else np.linspace(0, 1, 200)
                axes[r, c].plot(grid, norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid,
                                "k--", lw=1.4)
                axes[r, c].set(
                    title=f"{allocation.removesuffix('_improving')}: x{coordinate}",
                    xlim=(grid.min(), grid.max()), ylim=(0, 1),
                )
        fig.suptitle(
            f"{kernel.title()}, 4M model calls: all 30 posterior CDFs per allocation strategy",
        )
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"allocation_{kernel}_marginals.{extension}", dpi=160)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for r, kernel in enumerate(("hessian", "reflection")):
        for allocation, colour in zip(allocations, colours, strict=True):
            movement = summary[f"{kernel}_{allocation}_4000000"]["movement"]
            for c, coordinate in enumerate((1, 16)):
                distances = [movement[str(low)][f"rms_dx{coordinate}"] for low, _ in intervals]
                normalised = [value * np.sqrt(12) if value is not None else np.nan
                              for value in distances]
                axes[r, c].plot(range(6), normalised, "o-", color=colour,
                                label=allocation.removesuffix("_improving"))
        for c, coordinate in enumerate((1, 16)):
            axes[r, c].axhline(np.sqrt(2), color="black", ls="--", lw=1,
                               label="Independent uniform draws (reference)")
            axes[r, c].set(
                title=f"{kernel.title()}: x{coordinate}", yscale="log", ylim=(0.003, 2.0),
                xticks=range(6),
                xticklabels=["0–5", "5–15", "15–40", "40–100", "100–1000", ">1000"],
                xlabel="Parent peak-gap bin (nats)", ylabel="Median RMS move / full uniform SD",
            )
            axes[r, c].legend(fontsize=7)
    fig.suptitle("Actual seed-to-child movement at 4M calls; medians across runs within each bin")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"allocation_movement.{extension}", dpi=160)


if __name__ == "__main__":
    main()
