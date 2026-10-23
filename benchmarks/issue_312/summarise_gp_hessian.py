"""Validate and plot the six-seed GP/Hessian comparison."""

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
    folder = Path(__file__).resolve().parent
    groups = {
        "Isotropic": (folder / "raw/precision-f64-0.jsonl", "/tmp/issue312-precision-f64"),
        "Analytic Hessian": (
            folder / "raw/curved-hessian-scaled-0.jsonl", "/tmp/issue312-curved-hessian-scaled-0",
        ),
    }
    records = []
    snapshots = {}
    gp_fits = []
    for name, (path, archive_root) in groups.items():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["seed"] < 6:
                records.append({
                    **row, "group": name,
                    "archive": f"{archive_root}/seed-{row['seed']}.npz",
                })
    for count in (128, 256):
        for suffix in ("a", "b"):
            root = Path(f"/tmp/jaxns-gp-final{count}-{suffix}")
            for line in (root / "results.jsonl").read_text().splitlines():
                row = json.loads(line)
                for name, checksum in row["source_sha256"].items():
                    sources = [(folder / name).read_bytes()]
                    if name in ("gp_hessian_sampler.py", "run_gp_hessian.py"):
                        # Preserve the executed GP source when its unchanged
                        # transition law is shared with the NN experiment.
                        sources.append(gzip.decompress(
                            (folder / f"raw/{name}.before_nn.gz").read_bytes(),
                        ))
                    assert checksum in {hashlib.sha256(source).hexdigest() for source in sources}
                assert row["training_points"] == count and row["fit_iterations"] == 100
                assert row["refresh_efolds"] == 1.0 and row["warm_start"]
                assert row["oracle_scale"]["quadratic_dimension"] == 15
                directory = root / f"seed-{row['seed']}"
                fits = [json.loads(line) for line in
                        (directory / "fits.jsonl").read_text().splitlines()]
                assert len(fits) == row["geometry_updates"]
                log_x = np.array([fit["log_X"] for fit in fits])
                assert log_x[0] == 0 and np.all(np.abs(np.diff(log_x)) >= 1)
                for fit in fits:
                    assert fit["training_points"] <= count
                    assert fit["negative_log_marginal_likelihood"] == min(fit["restart_objectives"])
                    gp_fits.append({**fit, "training_capacity": count})
                    with np.load(directory / f"fit-{fit['update']:03d}.npz") as saved:
                        assert np.all(np.isfinite(saved["coefficients"]))
                        np.testing.assert_allclose(
                            saved["input_mean"], saved["points"].mean(axis=0),
                        )
                        if fit["contour"] is not None:
                            assert np.all(saved["values"] > fit["contour"])
                        indices = saved["training_indices"]
                        assert len(np.unique(indices)) == len(indices) == fit["training_points"]
                records.append({
                    **row, "group": f"GP {count}", "archive": str(directory / "samples.npz"),
                })

    individual = []
    quantiles = (np.arange(8192) + 0.5) / 8192
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    maximum_likelihood_error = 0.0
    for row in records:
        assert row["roots"] == 150 and row["slices"] == 90
        assert row["allocation_target"] == "uniform"
        assert row["epochs"] == 1 and row["depth_reached"] and row["termination_reason"] == 0
        with np.load(row["archive"]) as saved:
            points, values = saved["samples"], saved["log_likelihood"]
            assert points.dtype == np.float64 and points.shape == (row["samples"], 18)
            assert saved["unit_samples"].dtype == np.float64
            np.testing.assert_array_equal(points, saved["unit_samples"])
            assert np.all((points >= 0) & (points <= 1))
            if "log_L_constraints" in saved.files:
                assert np.all(values > saved["log_L_constraints"])
            residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
            reference = peak - 0.5 * (residual / 1e-5) ** 2
            reference -= 0.5 * np.sum(((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
            np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
            maximum_likelihood_error = max(
                maximum_likelihood_error, float(np.max(abs(values - reference))),
            )
            np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
            weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
            mean = np.sum(weights[:, None] * points, axis=0)
            std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
            distances = {}
            for coordinate in (1, 16, 17, 5):
                np.testing.assert_allclose(std[coordinate], row[f"std_x{coordinate}"], atol=1e-12)
                reference = (
                    norm.ppf(quantiles, loc=0.5, scale=0.05) if coordinate == 5 else quantiles
                )
                distances[f"w1_x{coordinate}"] = float(wasserstein_distance(
                    points[:, coordinate], reference, u_weights=weights,
                ))
            snapshots[row["group"], row["seed"]] = (points, weights)
        individual.append({
            "group": row["group"], "seed": row["seed"],
            **{name: row[name] for name in (
                "log_Z", "log_Z_uncert", "evaluations", "peak_gap",
                "std_x1", "std_x16", "std_x17", "std_x5", "seconds",
            )}, **distances,
            "archive_sha256": hashlib.sha256(Path(row["archive"]).read_bytes()).hexdigest(),
        })
    names = ("Isotropic", "Analytic Hessian", "GP 128", "GP 256")
    summary = {}
    for name in names:
        selected = sorted(
            [row for row in individual if row["group"] == name], key=lambda row: row["seed"],
        )
        assert [row["seed"] for row in selected] == list(range(6))
        z = np.array([row["log_Z"] for row in selected])
        summary[name] = {
            "seeds": 6, "mean_log_Z": float(z.mean()), "rmse": float(np.sqrt(np.mean(z ** 2))),
            "log_Z_range": [float(z.min()), float(z.max())],
            "mean_evaluations": float(np.mean([row["evaluations"] for row in selected])),
            "median_peak_gap": float(np.median([row["peak_gap"] for row in selected])),
            "mean_quoted_uncertainty": float(np.mean([row["log_Z_uncert"] for row in selected])),
            "within_two_quoted_sd": sum(
                abs(row["log_Z"]) <= 2 * row["log_Z_uncert"] for row in selected
            ),
            "median_widths": {
                f"x{i}": float(np.median([row[f"std_x{i}"] for row in selected]))
                for i in (1, 16, 17, 5)
            },
            "median_wasserstein": {
                f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                for i in (1, 16, 17, 5)
            },
        }
    summary["validation"] = {"max_likelihood_error": maximum_likelihood_error}
    with (folder / "gp_hessian_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(individual[0]))
        writer.writeheader()
        writer.writerows(individual)
    (folder / "gp_hessian_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    colours = ["#888888", "#2b8c69", "#377eb8", "#d46a30"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for index, (name, colour) in enumerate(zip(names, colours)):
        selected = [row for row in individual if row["group"] == name]
        x = index + np.linspace(-0.12, 0.12, 6)
        axes[0].errorbar(x, [row["log_Z"] for row in selected],
                         yerr=[row["log_Z_uncert"] for row in selected], fmt="o", color=colour)
        axes[1].scatter([row["evaluations"] / 1e6 for row in selected],
                        [abs(row["log_Z"]) for row in selected], label=name, color=colour)
        widths = [[row[f"std_x{i}"] / (0.05 if i == 5 else np.sqrt(1 / 12))
                   for i in (1, 16, 17, 5)] for row in selected]
        axes[2].plot(range(4), np.median(widths, axis=0), "o-", color=colour, label=name)
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(xticks=range(4), xticklabels=["Isotropic", "Analytic\nHessian", "GP 128", "GP 256"],
                ylabel="log evidence (truth ≈ 0)")
    axes[1].set(xlabel="Likelihood calls (millions)", ylabel="Absolute log-evidence error")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Median posterior SD / true SD")
    fig.suptitle("18D ridge, matched seeds 0–5: 150 roots, 90 transitions, float64 cube")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"gp_hessian_comparison.{extension}", dpi=160)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for count, colour in zip((128, 256), colours[2:]):
        for seed in range(6):
            fits = [row for row in gp_fits
                    if row["training_capacity"] == count and row["seed"] == seed]
            x = [-row["log_X"] for row in fits]
            label = f"{count} training points" if seed == 0 else None
            axes[0].semilogy(x, [max(row["stiff_eigenvalue_ratio"], 1e-16) for row in fits],
                             color=colour, alpha=0.55, label=label)
            angles = [np.degrees(np.arccos(np.clip(row["stiff_alignment"], 0, 1)))
                      for row in fits]
            axes[1].plot(x, angles, color=colour, alpha=0.55)
            selected = [row for row in fits if row["held_out_rmse"] is not None]
            axes[2].semilogy([-row["log_X"] for row in selected],
                             [max(row["held_out_rmse"] / row["output_scale"], 1e-12)
                              for row in selected],
                             color=colour, alpha=0.55)
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set(ylabel="Predicted / true largest Hessian eigenvalue", ylim=(1e-12, 10))
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Median stiff-axis error (degrees)", ylim=(-2, 92))
    axes[2].set(ylabel="Held-out log-likelihood RMSE / training SD")
    for axis in axes:
        axis.set_xlabel("Compression −log X at GP refresh")
    fig.suptitle(
        "GP curvature audit: held-out points when available; analytic Hessian only for audit",
    )
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"gp_hessian_fit_quality.{extension}", dpi=160)

    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
    for row_index, name in enumerate(names[1:]):
        for seed in range(6):
            points, weights = snapshots[name, seed]
            for column, coordinate in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, coordinate])
                axes[row_index, column].plot(points[order, coordinate], np.cumsum(weights[order]),
                                             lw=1, alpha=0.8, label=f"Seed {seed}")
        for column, coordinate in enumerate((1, 16, 17, 5)):
            grid = np.linspace(0.3, 0.7, 200) if coordinate == 5 else np.linspace(0, 1, 200)
            axes[row_index, column].plot(grid,
                                         norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid,
                                         "k--", lw=1.5)
            axes[row_index, column].set(
                title=f"{name}: x{coordinate}", ylim=(0, 1), xlim=(grid.min(), grid.max()),
            )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Posterior CDFs; dashed curve is the exact marginal")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"gp_hessian_marginals.{extension}", dpi=160)


if __name__ == "__main__":
    main()
