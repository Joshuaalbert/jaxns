"""Validate and compare the six-seed Dynesty and UltraNest ridge controls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance


def main():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(__file__).parent
    groups = (
        ("Dynesty 150", "dynesty-rslice90-150", "run_dynesty_ridge.py", 150),
        ("Dynesty 1000", "dynesty-rslice90-1000", "run_dynesty_ridge.py", 1000),
        ("UltraNest ≥1000", "ultranest-adaptive72-144-1000", "run_ultranest_ridge.py", 1000),
    )
    metrics = (
        "group", "seed", "log_Z", "log_Z_uncert", "peak_gap", "evaluations", "sampler_ncall",
        "seconds", "ess", "std_x1", "std_x16", "std_x17", "std_x5",
        "w1_x1", "w1_x16", "w1_x17", "ks_x1", "ks_x16", "ks_x17",
        "min_gaussian_width_ratio", "max_gaussian_width_ratio",
        "stiff_residual_mean", "stiff_residual_std",
    )
    rows = []
    summaries = {}
    summary_figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    marginal_figure, marginal_axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    curve_figure, curve_axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)
    live_figure, live_axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    grid = np.linspace(0, 1, 1001)
    uniform_grid = (np.arange(16384) + 0.5) / 16384
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))

    for group_index, (label, stem, runner, live_points) in enumerate(groups):
        group_rows = []
        native = []
        for seed in range(6):
            path = directory / "raw" / f"{stem}-{seed}.jsonl"
            lines = path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError(f"Expected one completed record in {path}.")
            record = json.loads(lines[0])
            if not (
                record["seed"] == seed and record["dimension"] == 18
                and record["sigma_stiff"] == 1e-5 and record["sigma"] == 0.05
                and record["curvature"] == 0.3 and record["u_dtype"] == "float64"
                and record["normalized_gaussian_reference_passed"]
            ):
                raise ValueError(f"Unexpected target or precision in {path}.")
            if group_index < 2:
                valid = (
                    record["dynesty"] == "3.0.0" and record["live_points"] == live_points
                    and record["slices"] == 90 and record["bound"] == "multi"
                    and record["sample"] == "rslice" and record["dlogz"] == 0.01
                    and record["added_live"]
                    and record["samples"] == record["iterations"] + live_points
                    and record["last_yielded_delta_log_Z"] <= 0.011
                    and record["sampler_ncall"] == record["sum_result_ncall"]
                    and record["sampler_ncall"] >= record["evaluations"]
                )
            else:
                valid = (
                    record["ultranest"] == "4.6.3" and record["min_live_points"] == live_points
                    and record["initial_slices"] == 72 and record["max_slices"] == 144
                    and record["direction"] == "generate_mixture_random_direction"
                    and record["adaptive_nsteps"] == "move-distance"
                    and record["region"] == "MLFriends" and record["vectorized"]
                    and record["dlogz"] == 0.5 and record["dKL"] == 0.5
                    and record["frac_remain"] == 0.01 and record["min_ess"] == 400
                    and record["max_ncalls"] is None and record["max_iters"] is None
                    and record["max_num_improvement_loops"] == -1 and record["completed"]
                    and record["samples"] == record["iterations"]
                    and record["evaluations"] >= record["sampler_ncall"]
                )
            if not valid:
                raise ValueError(f"Unexpected configuration or incomplete result in {path}.")
            runner_hash = hashlib.sha256((directory / runner).read_bytes()).hexdigest()
            if runner_hash != record["script_sha256"]:
                raise ValueError(f"Runner checksum does not match {path}.")
            archive = path.with_suffix(".npz")
            if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sample_archive_sha256"]:
                raise ValueError(f"Sample checksum does not match {path}.")
            with np.load(archive) as data:
                points = data["samples"]
                log_weights = data["log_weights"]
                log_likelihood = data["log_likelihood"]
                if group_index == 2:
                    step_log = data["step_log"]
                    np.testing.assert_allclose(step_log[:, 2].mean(), record["mean_slices"])
                    if np.max(step_log[:, 2]) > 144 or np.min(step_log[:, 2]) < 1:
                        raise ValueError(f"Slice adaptation exceeded its bounds in {path}.")
                    if np.max(data["sequence_nlive"]) != record["max_live_points"]:
                        raise ValueError(f"Incorrect population summary in {path}.")
            if points.shape != (record["samples"], 18) or not np.all(np.isfinite(points)):
                raise ValueError(f"Invalid posterior sample archive {archive}.")
            if np.any((points < 0) | (points > 1)):
                raise ValueError(f"Samples outside the prior in {archive}.")
            np.testing.assert_allclose(logsumexp(log_weights), record["log_Z"], atol=1e-10)
            weights = np.exp(log_weights - logsumexp(log_weights))
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
            mean = np.sum(weights[:, None] * points, axis=0)
            std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
            np.testing.assert_allclose(mean, record["posterior_mean"], atol=1e-12)
            np.testing.assert_allclose(std, record["posterior_std"], atol=1e-12)
            np.testing.assert_allclose(1 / np.sum(weights ** 2), record["ess"], rtol=1e-12)
            # Re-evaluate every stored likelihood, independently of either sampler.
            residual = (points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5
            expected = peak - 0.5 * residual ** 2 - 0.5 * np.sum(
                ((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1,
            )
            np.testing.assert_allclose(expected, log_likelihood, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(peak - log_likelihood.max(), record["peak_gap"], atol=1e-10)
            row = {key: record[key] for key in metrics if key in record}
            row["group"] = label
            row["min_gaussian_width_ratio"] = float(std[2:16].min() / 0.05)
            row["max_gaussian_width_ratio"] = float(std[2:16].max() / 0.05)
            residual_mean = np.sum(weights * residual)
            row["stiff_residual_mean"] = float(residual_mean)
            row["stiff_residual_std"] = float(np.sqrt(
                np.sum(weights * (residual - residual_mean) ** 2),
            ))
            for coordinate in (1, 16, 17):
                distance = wasserstein_distance(points[:, coordinate], uniform_grid,
                                                u_weights=weights)
                np.testing.assert_allclose(distance, record[f"w1_x{coordinate}"], atol=1e-12)
                order = np.argsort(points[:, coordinate])
                cdf = np.cumsum(weights[order])
                # Compare both sides of each empirical CDF jump with the known
                # uniform CDF. A plausible width alone can hide missing intervals.
                row[f"ks_x{coordinate}"] = float(max(
                    np.max(np.abs(cdf - points[order, coordinate])),
                    np.max(np.abs(cdf - weights[order] - points[order, coordinate])),
                ))
            group_rows.append(row)
            rows.append(row)

            for column, coordinate in enumerate((1, 16, 17, 5)):
                axis = marginal_axes[group_index, column]
                bins = np.linspace(0, 1, 81) if coordinate != 5 else np.linspace(0.25, 0.75, 61)
                density, edges = np.histogram(points[:, coordinate], bins=bins,
                                              weights=weights, density=True)
                axis.stairs(density, edges, color=f"C{seed}", label=f"Seed {seed}", linewidth=1)
            if group_index > 0:
                axis = curve_axes[group_index - 1, seed]
                selected = np.random.default_rng(4312 + seed).choice(len(points), 2500, p=weights)
                axis.plot(grid, 0.5 + 0.3 * np.sin(2 * np.pi * grid), color="black", linewidth=0.7)
                axis.scatter(points[selected, 1], points[selected, 0], s=2,
                             color=f"C{seed}", alpha=0.15)
                axis.set_title(f"Seed {seed}: log Z {record['log_Z']:+.2f}", fontsize=10)
                axis.set_xlabel("x1")
                axis.set_ylabel(label + "\nx0" if seed == 0 else "x0")
                axis.set_xlim(0, 1)
                axis.set_ylim(0.15, 0.85)
            if group_index == 2:
                progress = [json.loads(line) for line in
                            path.with_suffix(".progress.jsonl").read_text().splitlines()]
                log_volumes = np.asarray([item["log_volume"] for item in progress])
                # A reactive rerun can revisit shallower contours. Break the line
                # at that boundary instead of implying one chronological descent.
                reset = np.r_[False, np.diff(log_volumes) > 0]
                for axis, coordinate in zip(live_axes.flat[:3], (1, 16, 17)):
                    spreads = np.asarray([item["live_std"][coordinate] for item in progress])
                    spreads[reset] = np.nan
                    axis.plot(-log_volumes, spreads, color=f"C{seed}", label=f"Seed {seed}")
                steps = np.asarray([item["slices"] for item in progress], dtype=float)
                steps[reset] = np.nan
                live_axes[1, 1].plot(-log_volumes, steps, color=f"C{seed}", label=f"Seed {seed}")
                native.append({key: record[key] for key in (
                    "seed", "final_roots", "final_min_live_points", "max_live_points",
                    "mean_slices", "final_slices", "fraction_at_max_slices", "tail_fraction",
                    "jump_distance_ratio", "fraction_far_enough", "insertion_order_converged",
                    "constructor_evaluations", "log_Z_uncert_bootstrap",
                )})

        errors = np.asarray([row["log_Z"] for row in group_rows])
        uncertainties = np.asarray([row["log_Z_uncert"] for row in group_rows])
        summaries[label] = {
            "mean_log_Z": float(errors.mean()),
            "rmse_log_Z": float(np.sqrt(np.mean(errors ** 2))),
            "replicate_std_log_Z": float(errors.std(ddof=1)),
            "within_two_quoted_uncertainties": int(np.sum(np.abs(errors) <= 2 * uncertainties)),
            "mean_likelihood_evaluations": float(np.mean(
                [row["evaluations"] for row in group_rows],
            )),
            "native_diagnostics": native,
        }
        positions = np.arange(6) + (group_index - 1) * 0.20
        color = f"C{group_index}"
        axes[0, 0].errorbar(positions, errors, yerr=uncertainties, fmt="o", capsize=3,
                            color=color, label=label)
        for axis, key, scale in (
            (axes[0, 1], "peak_gap", 1), (axes[0, 2], "evaluations", 1e-6),
            (axes[1, 0], "std_x1", np.sqrt(12)),
            (axes[1, 1], "std_x16", np.sqrt(12)),
            (axes[1, 2], "std_x17", np.sqrt(12)),
        ):
            axis.plot(positions, [row[key] * scale for row in group_rows], "o-",
                      color=color, label=label, linewidth=0.8)
        for column, coordinate in enumerate((1, 16, 17, 5)):
            axis = marginal_axes[group_index, column]
            if coordinate != 5:
                axis.plot((0, 1), (1, 1), color="black", linestyle="--", label="Known posterior")
                axis.set_xlim(0, 1)
            else:
                normal_grid = np.linspace(0.25, 0.75, 1001)
                axis.plot(normal_grid, np.exp(-0.5 * ((normal_grid - 0.5) / 0.05) ** 2)
                          / (0.05 * np.sqrt(2 * np.pi)), color="black", linestyle="--",
                          label="Known posterior")
                axis.set_xlim(0.25, 0.75)
            axis.set_title(f"{label}: x{coordinate}")
            axis.set_xlabel(f"x{coordinate}")
            axis.set_ylabel("Weighted density")
            axis.grid(alpha=0.15)

    titles = ("log Z ± quoted uncertainty (truth: 0)", "Best log L below known peak",
              "Actual likelihood calls (millions)", "x1: width / uniform width",
              "x16: width / uniform width", "x17: width / uniform width")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Seed")
        axis.set_xticks(range(6))
        axis.grid(alpha=0.15)
    for axis in axes[1, :]:
        axis.axhline(1, color="black", linewidth=0.8)
    axes[0, 0].legend(fontsize=8)
    marginal_axes[0, 3].legend(fontsize=7)
    summary_figure.suptitle("18-D curved ridge: six seeds per configuration")
    marginal_figure.suptitle("Known marginals versus complete weighted results")
    curve_figure.suptitle("Posterior coverage of the known ridge; 2500 weighted draws per run")
    for axis, coordinate in zip(live_axes.flat[:3], (1, 16, 17)):
        axis.axhline(1 / np.sqrt(12), color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"UltraNest live-population spread: x{coordinate}")
        axis.set_ylabel("Unweighted standard deviation")
    live_axes[1, 1].set_title("Adaptive slice count at region updates")
    live_axes[1, 1].axhline(144, color="black", linestyle="--", linewidth=0.8)
    live_axes[1, 1].set_ylabel("Accepted slice steps per chain")
    for axis in live_axes.flat:
        axis.set_xlabel("Compression: −log X")
        axis.grid(alpha=0.15)
    live_axes[1, 1].legend(fontsize=8)
    live_figure.suptitle("UltraNest evolution before posterior weighting")
    for figure, suffix in ((summary_figure, "summary"), (marginal_figure, "marginals"),
                           (curve_figure, "coverage"), (live_figure, "live_spread")):
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"large_population_{suffix}.{extension}", dpi=160)
        plt.close(figure)
    with (directory / "large_population_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"groups": summaries, "runs": rows, "validation_passed": True}
    (directory / "large_population_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
