"""Validate and plot the six-seed Dynesty reproduction of the reported toy."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp


def main():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(__file__).parent
    metrics = (
        "seed", "log_Z", "log_Z_uncert", "peak_gap", "evaluations", "sum_result_ncall",
        "seconds", "ess", "std_x1", "std_x16", "std_x17", "std_x5",
        "w1_x1", "w1_x16", "w1_x17",
    )
    rows = []
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    curve_figure, curve_axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    grid = np.linspace(0, 1, 1001)
    for seed in range(6):
        path = directory / "raw" / f"dynesty-rslice90-150-{seed}.jsonl"
        lines = path.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError(f"Expected one complete record in {path}.")
        row = json.loads(lines[0])
        if not (
            row["seed"] == seed and row["dynesty"] == "3.0.0"
            and row["live_points"] == 150 and row["slices"] == 90
            and row["bound"] == "multi" and row["sample"] == "rslice"
            and row["dlogz"] == 0.01 and row["sigma_stiff"] == 1e-5
            and row["sigma"] == 0.05 and row["curvature"] == 0.3
            and row["dimension"] == 18 and row["u_dtype"] == "float64"
            and row["added_live"] and row["normalized_gaussian_reference_passed"]
            and row["samples"] == row["iterations"] + 150
            and row["last_yielded_delta_log_Z"] <= 0.011
            and row["sampler_ncall"] == row["sum_result_ncall"] >= row["evaluations"]
        ):
            raise ValueError(f"Unexpected configuration or incomplete result in {path}.")
        if hashlib.sha256((directory / "run_dynesty_ridge.py").read_bytes()).hexdigest() != (
            row["script_sha256"]
        ):
            raise ValueError(f"Runner checksum does not match {path}.")
        archive = path.with_suffix(".npz")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row["sample_archive_sha256"]:
            raise ValueError(f"Sample checksum does not match {path}.")
        with np.load(archive) as data:
            points = data["samples"]
            log_weights = data["log_weights"]
            log_likelihood = data["log_likelihood"]
        if len(points) != row["samples"] or np.any((points < 0) | (points > 1)):
            raise ValueError(f"Invalid posterior sample archive {archive}.")
        np.testing.assert_allclose(logsumexp(log_weights), row["log_Z"], atol=1e-10)
        weights = np.exp(log_weights - logsumexp(log_weights))
        mean = np.sum(weights[:, None] * points, axis=0)
        std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
        np.testing.assert_allclose(mean, row["posterior_mean"], atol=1e-12)
        np.testing.assert_allclose(std, row["posterior_std"], atol=1e-12)
        peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
        # Independently re-evaluate every stored likelihood after sampling.
        # This has no influence on the sampler or its instrumented call count.
        residual = (points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5
        expected = peak - 0.5 * residual ** 2 - 0.5 * np.sum(
            ((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1,
        )
        np.testing.assert_allclose(expected, log_likelihood, rtol=1e-10, atol=1e-7)
        rows.append({key: row[key] for key in metrics})

        for axis, coordinate in zip(axes.flat[2:], (1, 16, 17, 5)):
            bins = np.linspace(0, 1, 81) if coordinate != 5 else np.linspace(0.25, 0.75, 61)
            density, edges = np.histogram(points[:, coordinate], bins=bins,
                                          weights=weights, density=True)
            axis.stairs(density, edges, color=f"C{seed}", label=f"Seed {seed}", linewidth=1.2)
        axis = curve_axes.flat[seed]
        selected = np.random.default_rng(4300 + seed).choice(len(points), 2000, p=weights)
        axis.plot(grid, 0.5 + 0.3 * np.sin(2 * np.pi * grid), color="black", linewidth=0.8)
        axis.scatter(points[selected, 1], points[selected, 0], s=2, color=f"C{seed}", alpha=0.2)
        axis.set_title(f"Seed {seed}: log Z = {row['log_Z']:+.3f}")
        axis.set_xlabel("x1: uniform ridge coordinate")
        axis.set_ylabel("x0")
        axis.set_xlim(0, 1)
        axis.set_ylim(0.15, 0.85)

    seed_values = np.asarray([row["seed"] for row in rows])
    errors = np.asarray([row["log_Z"] for row in rows])
    uncertainties = np.asarray([row["log_Z_uncert"] for row in rows])
    axes[0, 0].errorbar(seed_values, errors, yerr=uncertainties, fmt="o", capsize=3)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("Seed")
    axes[0, 0].set_ylabel("log Z ± quoted SD (truth: 0)")
    for coordinate, marker in ((1, "o"), (16, "s"), (17, "^")):
        axes[0, 1].plot(seed_values, [row[f"std_x{coordinate}"] * np.sqrt(12) for row in rows],
                        marker=marker, label=f"x{coordinate}")
    axes[0, 1].axhline(1, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("Seed")
    axes[0, 1].set_ylabel("Posterior width / uniform-prior width")
    axes[0, 1].legend(fontsize=8)
    for axis, coordinate in zip(axes.flat[2:], (1, 16, 17, 5)):
        if coordinate != 5:
            axis.plot((0, 1), (1, 1), color="black", linestyle="--", label="Known posterior")
            axis.set_xlim(0, 1)
        else:
            normal_grid = np.linspace(0.25, 0.75, 1001)
            axis.plot(normal_grid, np.exp(-0.5 * ((normal_grid - 0.5) / 0.05) ** 2)
                      / (0.05 * np.sqrt(2 * np.pi)), color="black", linestyle="--",
                      label="Known posterior")
            axis.set_xlim(0.25, 0.75)
        axis.set_xlabel(f"x{coordinate}")
        axis.set_ylabel("Weighted posterior density")
    axes[1, 2].legend(fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    figure.suptitle("Dynesty 3.0.0: 150 live points, 90 rslice steps, dlogz=0.01")
    curve_figure.suptitle("Dynesty posterior draws along the known curve")
    for plot, stem in ((figure, "dynesty_ridge_summary"), (curve_figure, "dynesty_ridge_coverage")):
        plot.savefig(directory / f"{stem}.png", dpi=180)
        plot.savefig(directory / f"{stem}.pdf")
        plt.close(plot)
    with (directory / "dynesty_ridge_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "runs": rows, "mean_log_Z": float(errors.mean()),
        "rmse_log_Z": float(np.sqrt(np.mean(errors ** 2))),
        "within_two_sigma": int(np.sum(np.abs(errors) <= 2 * uncertainties)),
        "mean_likelihood_evaluations": float(np.mean([row["evaluations"] for row in rows])),
        "mean_dynesty_ncall": float(np.mean([row["sum_result_ncall"] for row in rows])),
        "validation_passed": True,
    }
    (directory / "dynesty_ridge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
