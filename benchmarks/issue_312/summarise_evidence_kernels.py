"""Validate and compare 30-seed evidence-goal runs with the existing controls."""

import argparse
import csv
import gzip
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
    baseline_path = folder / "reflection_summary.csv"
    rows = []
    with baseline_path.open() as stream:
        for source in csv.DictReader(stream):
            if source["group"] not in ("Hessian, contour scaled", "Reflection + 9 slices"):
                continue
            kernel = "hessian" if source["group"].startswith("Hessian") else "reflection"
            row = {
                key: float(value) for key, value in source.items()
                if key not in ("group", "archive_sha256")
            }
            rows.append({
                **row, "kernel": kernel, "allocation": "uniform", "seed": int(row["seed"]),
                "epochs": 1, "roots": 150, "archive_sha256": source["archive_sha256"],
            })

    # Every run was started before a lint-only explicit binding of the goal
    # callback's loop variables. Preserve that executed source, not just HEAD.
    original = gzip.decompress((folder / "raw/run_evidence_kernels.executed.py.gz").read_bytes())
    runner_hashes = {
        hashlib.sha256(original).hexdigest(),
        hashlib.sha256((folder / "run_evidence_kernels.py").read_bytes()).hexdigest(),
    }
    snapshots = {}
    maximum_error = 0.0
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    quantiles = (np.arange(8192) + 0.5) / 8192
    for kernel in ("hessian", "reflection"):
        for letter in "abc":
            raw = folder / f"raw/evidence-{kernel}-{letter}.jsonl"
            for line in raw.read_text().splitlines():
                row = json.loads(line)
                assert row["source_sha256"]["run_evidence_kernels.py"] in runner_hashes
                for source in ("hessian_sampler.py", "reflection_sampler.py", "run_ridge.py"):
                    assert row["source_sha256"][source] == hashlib.sha256(
                        (folder / source).read_bytes(),
                    ).hexdigest()
                assert row["kernel"] == kernel
                expected_config = (
                    {"num_slices": 90, "scale_to_contour": True, "correct_mh": True}
                    if kernel == "hessian" else {
                        "num_cycles": 9, "slices_per_cycle": 1, "num_bounces": 10,
                        "step_size": 0.003, "boundary_only": False,
                    }
                )
                assert row["kernel_config"] == expected_config
                assert row["initial_roots"] == row["replacement_width"] == row["delta_K"] == 150
                assert row["allocation_target"] == "evidence_improving"
                assert row["target_log_Z_uncert"] == 0.4 and row["log_Z_uncert"] <= 0.4
                assert row["goal_reached"] and row["depth_reached"] and row["unlimited_samples"]
                assert row["termination_reason"] == 0 and row["phantoms"] == 0
                assert row["unit_dtype"] == "float64" and row["sigma_stiff"] == 1e-5
                assert row["curvature"] == 0.3 and row["dlogZ"] == float(np.log1p(1e-3))
                history = row["history"]
                assert [item["epoch"] for item in history] == list(range(1, row["epochs"] + 1))
                assert all(item["log_Z_uncert"] > 0.4 for item in history[:-1])
                for field in ("log_Z", "log_Z_uncert", "samples", "roots", "evaluations"):
                    np.testing.assert_allclose(history[-1][field], row[field], rtol=1e-12)
                assert row["gradient_evaluations"] == (
                    (row["samples"] - row["roots"]) * (90 if kernel == "reflection" else 0)
                )
                assert row["evaluations"] == (
                    row["value_only_evaluations"] + row["gradient_evaluations"]
                )
                archive = (
                    options.archive_root / f"jaxns-evidence-{kernel}-{letter}"
                    / f"seed-{row['seed']}.npz"
                )
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
                with np.load(archive) as saved:
                    points = saved["samples"]
                    values = saved["log_likelihood"]
                    assert points.dtype == np.float64 and points.shape == (row["samples"], 18)
                    np.testing.assert_array_equal(points, saved["unit_samples"])
                    assert np.all((points >= 0) & (points <= 1))
                    assert np.all(values > saved["log_L_constraints"])
                    assert np.sum(np.isneginf(saved["log_L_constraints"])) == row["roots"]
                    residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
                    reference = peak - 0.5 * (
                        (residual / 1e-5) ** 2
                        + np.sum(((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
                    )
                    np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
                    maximum_error = max(maximum_error, float(np.max(abs(values - reference))))
                    np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
                    weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                    np.testing.assert_allclose(1 / np.sum(weights ** 2), row["kish_ess"])
                    mean = np.sum(weights[:, None] * points, axis=0)
                    std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
                    for coordinate in (1, 16, 17, 5):
                        np.testing.assert_allclose(std[coordinate], row[f"std_x{coordinate}"])
                        reference = (norm.ppf(quantiles, loc=0.5, scale=0.05)
                                     if coordinate == 5 else quantiles)
                        row[f"w1_x{coordinate}"] = float(wasserstein_distance(
                            points[:, coordinate], reference, u_weights=weights,
                        ))
                    snapshots[kernel, row["seed"]] = (points, weights)
                rows.append({**row, "allocation": "evidence_improving"})

    summary = {}
    # Resample paired seed indices, using the same indices for each kernel's
    # before/after RMSE difference. This measures finite-cohort uncertainty.
    bootstrap_indices = np.random.default_rng(312040).integers(0, 30, (10000, 30))
    paired = {}
    for kernel in ("hessian", "reflection"):
        for allocation in ("uniform", "evidence_improving"):
            selected = sorted(
                [row for row in rows
                 if row["kernel"] == kernel and row["allocation"] == allocation],
                key=lambda row: row["seed"],
            )
            assert [row["seed"] for row in selected] == list(range(30))
            errors = np.array([row["log_Z"] for row in selected])
            uncertainty = np.array([row["log_Z_uncert"] for row in selected])
            rmse_bootstrap = np.sqrt(np.mean(errors[bootstrap_indices] ** 2, axis=1))
            paired[kernel, allocation] = rmse_bootstrap
            summary[f"{kernel}_{allocation}"] = {
                "seeds": 30, "mean_log_Z": float(errors.mean()),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "rmse_bootstrap_95_percent": np.quantile(rmse_bootstrap, [0.025, 0.975]).tolist(),
                "empirical_sd": float(errors.std(ddof=1)),
                "range_log_Z": [float(errors.min()), float(errors.max())],
                "mean_quoted_uncertainty": float(uncertainty.mean()),
                "range_quoted_uncertainty": [float(uncertainty.min()), float(uncertainty.max())],
                "mean_evaluations": float(np.mean([row["evaluations"] for row in selected])),
                "mean_gradient_evaluations": float(np.mean(
                    [row["gradient_evaluations"] for row in selected],
                )),
                "within_one_quoted_sd": int(np.sum(abs(errors) <= uncertainty)),
                "within_two_quoted_sd": int(np.sum(abs(errors) <= 2 * uncertainty)),
                "normalised_rmse": float(np.sqrt(np.mean((errors / uncertainty) ** 2))),
                "median_peak_gap": float(np.median([row["peak_gap"] for row in selected])),
                "range_epochs": [min(row["epochs"] for row in selected),
                                 max(row["epochs"] for row in selected)],
                "mean_epochs": float(np.mean([row["epochs"] for row in selected])),
                "range_roots": [min(row["roots"] for row in selected),
                                max(row["roots"] for row in selected)],
                "median_widths": {
                    f"x{i}": float(np.median([row[f"std_x{i}"] for row in selected]))
                    for i in (1, 16, 17, 5)
                },
                "median_wasserstein": {
                    f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                    for i in (1, 16, 17, 5)
                },
            }
        delta = paired[kernel, "evidence_improving"] - paired[kernel, "uniform"]
        summary[f"{kernel}_paired_rmse_change_95_percent"] = np.quantile(
            delta, [0.025, 0.975],
        ).tolist()
    summary["validation"] = {
        "new_archives": len(snapshots), "max_likelihood_error": maximum_error,
        "baseline_csv_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "bootstrap_draws": 10000, "bootstrap_seed": 312040,
    }
    (folder / "evidence_kernels_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = (
        "kernel", "allocation", "seed", "log_Z", "log_Z_uncert", "evaluations",
        "gradient_evaluations", "epochs", "roots", "peak_gap", "std_x1", "std_x16", "std_x17",
        "std_x5", "w1_x1", "w1_x16", "w1_x17", "w1_x5", "archive_sha256",
    )
    with (folder / "evidence_kernels_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row_index, kernel in enumerate(("hessian", "reflection")):
        for allocation, colour, offset, label in (
            ("uniform", "#888888", -0.15, "Uniform, one epoch"),
            ("evidence_improving", "#287fa4", 0.15, "Evidence goal 0.4"),
        ):
            selected = sorted(
                [row for row in rows
                 if row["kernel"] == kernel and row["allocation"] == allocation],
                key=lambda row: row["seed"],
            )
            axes[row_index, 0].errorbar(
                np.arange(30) + offset, [row["log_Z"] for row in selected],
                yerr=[row["log_Z_uncert"] for row in selected], fmt=".", color=colour, label=label,
            )
            axes[row_index, 1].scatter(
                [row["evaluations"] / 1e6 for row in selected],
                [abs(row["log_Z"]) for row in selected], color=colour, s=20, label=label,
            )
            widths = summary[f"{kernel}_{allocation}"]["median_widths"]
            axes[row_index, 2].plot(
                range(4), [widths[f"x{i}"] / (0.05 if i == 5 else np.sqrt(1 / 12))
                           for i in (1, 16, 17, 5)], "o-", color=colour, label=label,
            )
        axes[row_index, 0].axhline(0, color="black", ls="--", lw=1)
        axes[row_index, 0].set(
            xlabel="Seed", ylabel="Log evidence (truth ≈ 0)", title=kernel.title(),
        )
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 1].set(xlabel="Model calls incl. gradients (millions)",
                               ylabel="Absolute log-evidence error")
        axes[row_index, 2].axhline(1, color="black", ls="--", lw=1)
        axes[row_index, 2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                               ylabel="Median posterior SD / true SD")
    fig.suptitle("18D ridge: 30 matched seeds per kernel, evidence allocation to quoted SD ≤ 0.4")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"evidence_kernels_comparison.{extension}", dpi=160)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for row_index, kernel in enumerate(("hessian", "reflection")):
        for seed in range(30):
            points, weights = snapshots[kernel, seed]
            for column, coordinate in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, coordinate])
                axes[row_index, column].plot(
                    points[order, coordinate], np.cumsum(weights[order]),
                    color="#287fa4", alpha=0.3, lw=0.8,
                )
        for column, coordinate in enumerate((1, 16, 17, 5)):
            grid = np.linspace(0.3, 0.7, 200) if coordinate == 5 else np.linspace(0, 1, 200)
            axes[row_index, column].plot(
                grid, norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid, "k--", lw=1.5,
            )
            axes[row_index, column].set(title=f"{kernel.title()}: x{coordinate}", ylim=(0, 1),
                                        xlim=(grid.min(), grid.max()))
    fig.suptitle("Evidence goal 0.4: all 30 posterior CDFs; dashed line is the exact marginal")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"evidence_kernels_marginals.{extension}", dpi=160)


if __name__ == "__main__":
    main()
