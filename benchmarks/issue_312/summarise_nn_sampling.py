"""Validate actual NN sampling archives and compare the matched six seeds."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, wasserstein_distance


def log_likelihood(points: np.ndarray) -> np.ndarray:
    """Independent NumPy reference, including all 15 Gaussian normalisers."""
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
    return peak - 0.5 * (residual / 1e-5) ** 2 - 0.5 * np.sum(
        ((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1,
    )


def main():
    folder = Path(__file__).resolve().parent
    names = ("Isotropic", "Analytic Hessian", "GP 128", "NN stored", "NN probes")
    colours = ("#888888", "#2b8c69", "#377eb8", "#b45a8b", "#d46a30")
    metrics = ("log_Z", "log_Z_uncert", "evaluations", "peak_gap", "seconds",
               "std_x1", "std_x16", "std_x17", "std_x5",
               "w1_x1", "w1_x16", "w1_x17", "w1_x5")
    individual = []
    # These are the previously validated same-seed controls, not new runs.
    with (folder / "gp_hessian_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["group"] in names[:3]:
                individual.append({
                    "group": row["group"], "seed": int(row["seed"]),
                    **{key: float(row[key]) for key in metrics},
                    "archive_sha256": row["archive_sha256"],
                })
    fits = [json.loads(line) for line in
            (folder / "raw/nn-sampling-fits.jsonl").read_text().splitlines()]
    snapshots = {}
    quantiles = (np.arange(8192) + 0.5) / 8192
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    maximum_likelihood_error = 0.0
    for mode, group in (("stored", "NN stored"), ("augmented", "NN probes")):
        runs = [json.loads(line) for line in
                (folder / f"raw/nn-sampling-{mode}.jsonl").read_text().splitlines()]
        for row in runs:
            assert row["roots"] == 150 and row["slices"] == 90 and row["phantoms"] == 0
            assert row["allocation_target"] == "uniform" and row["epochs"] == 1
            assert row["depth_reached"] and row["termination_reason"] == 0
            assert row["dlogZ"] == np.log1p(1e-3) and row["unit_dtype"] == "float64"
            assert row["representatives"] == 128 and row["fit_iterations"] == 1000
            assert row["warm_start"] and not row["relative_loss"]
            assert row["refresh_efolds"] == 1 and row["augment"] == (mode == "augmented")
            assert row["oracle_scale"] == {"peak": peak, "quadratic_dimension": 15}
            for name, checksum in row["source_sha256"].items():
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == checksum
            directory = Path(row["directory"])
            selected = [fit for fit in fits if fit["group"] == mode and fit["seed"] == row["seed"]]
            assert len(selected) == row["geometry_updates"]
            log_x = np.array([fit["log_X"] for fit in selected])
            assert log_x[0] == 0 and np.all(np.abs(np.diff(log_x)) >= 1)
            assert row["training_evaluations"] == sum(
                fit["training_evaluations"] for fit in selected
            )
            assert row["evaluations"] == row["sampling_evaluations"] + row["training_evaluations"]
            for fit in selected:
                assert fit["loss"] == min(fit["restart_losses"])
                assert fit["parameters"] == 1526 and fit["hidden_features"] == 32
                assert max(fit["restart_iterations"]) <= 1000
                path = directory / f"fit-{fit['update']:03d}.npz"
                assert hashlib.sha256(path.read_bytes()).hexdigest() == fit["archive_sha256"]
                with np.load(path) as saved:
                    points, values = saved["points"], saved["values"]
                    if fit["contour"] is not None:
                        assert np.all(values > fit["contour"])
                    assert len(points) == fit["contour_population"]
                    training = saved["training_points"]
                    indices = saved["training_indices"]
                    assert len(indices) == fit["training_points"] == len(np.unique(indices))
                    assert np.all((training >= 0) & (training <= 1))
                    np.testing.assert_allclose(
                        saved["training_values"], log_likelihood(training), rtol=1e-12, atol=5e-5,
                    )
                    if mode == "augmented":
                        assert fit["training_evaluations"] == 1024 and len(training) == 1152
                        np.testing.assert_array_equal(training[:128], points[saved["centres"]])
                    else:
                        assert fit["training_evaluations"] == 0 and len(indices) == 128
                        np.testing.assert_array_equal(training, points)
                        # Two floating-point whitening paths can choose different
                        # duplicate indices. Verify they still select identical
                        # coordinate sets before interpreting the omitted-row audit.
                        np.testing.assert_array_equal(
                            np.unique(points[saved["centres"]], axis=0),
                            np.unique(points[indices], axis=0),
                        )
                    np.testing.assert_allclose(saved["input_mean"], training.mean(axis=0))
                    centred = training - training.mean(axis=0)
                    covariance = centred.T @ centred / (len(training) - 1)
                    ridge = 1e-8 * max(float(np.trace(covariance) / 18), 1e-15)
                    np.testing.assert_allclose(
                        saved["input_cholesky"] @ saved["input_cholesky"].T,
                        covariance + ridge * np.eye(18), rtol=1e-10, atol=1e-15,
                    )
            path = directory / "samples.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["archive_sha256"]
            with np.load(path) as saved:
                points, values = saved["samples"], saved["log_likelihood"]
                assert points.shape == (row["samples"], 18) and points.dtype == np.float64
                assert saved["unit_samples"].dtype == np.float64
                np.testing.assert_array_equal(points, saved["unit_samples"])
                assert np.all((points >= 0) & (points <= 1))
                assert np.all(values > saved["log_L_constraints"])
                reference = log_likelihood(points)
                np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
                maximum_likelihood_error = max(
                    maximum_likelihood_error, float(np.max(np.abs(values - reference))),
                )
                np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
                weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                mean = np.sum(points * weights[:, None], axis=0)
                std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
                distances = {}
                for coordinate in (1, 16, 17, 5):
                    np.testing.assert_allclose(
                        std[coordinate], row[f"std_x{coordinate}"], atol=1e-12,
                    )
                    target = norm.ppf(quantiles, 0.5, 0.05) if coordinate == 5 else quantiles
                    distances[f"w1_x{coordinate}"] = float(wasserstein_distance(
                        points[:, coordinate], target, u_weights=weights,
                    ))
                duplicate_fraction = 1 - len(np.unique(points, axis=0)) / len(points)
                _, tie_counts = np.unique(values, return_counts=True)
                snapshots[group, row["seed"]] = (points, weights)
            individual.append({
                "group": group, "seed": row["seed"],
                **{key: row[key] for key in metrics if not key.startswith("w1_")}, **distances,
                "archive_sha256": row["archive_sha256"],
                **{key: row[key] for key in (
                    "sampling_evaluations", "training_evaluations", "fit_seconds",
                    "chain_hessian_evaluations", "geometry_updates",
                )},
                "duplicate_point_fraction": float(duplicate_fraction),
                "max_equal_likelihood_block": int(tie_counts.max()),
            })

    summary = {}
    for name in names:
        selected = sorted([row for row in individual if row["group"] == name],
                          key=lambda row: row["seed"])
        assert [row["seed"] for row in selected] == list(range(6))
        z = np.array([row["log_Z"] for row in selected])
        summary[name] = {
            "seeds": 6, "mean_log_Z": float(z.mean()), "rmse": float(np.sqrt(np.mean(z ** 2))),
            "log_Z_range": [float(z.min()), float(z.max())],
            "mean_evaluations": float(np.mean([row["evaluations"] for row in selected])),
            "median_peak_gap": float(np.median([row["peak_gap"] for row in selected])),
            "mean_quoted_uncertainty": float(np.mean([row["log_Z_uncert"] for row in selected])),
            "within_two_quoted_sd": sum(abs(row["log_Z"]) <= 2 * row["log_Z_uncert"]
                                        for row in selected),
            "median_widths": {f"x{i}": float(np.median([row[f"std_x{i}"] for row in selected]))
                              for i in (1, 16, 17, 5)},
        }
        if name in names[3:]:
            summary[name].update({
                "mean_training_evaluations": float(np.mean(
                    [row["training_evaluations"] for row in selected],
                )),
                "median_duplicate_fraction": float(np.median(
                    [row["duplicate_point_fraction"] for row in selected],
                )),
                "median_seconds": float(np.median([row["seconds"] for row in selected])),
                "median_fit_seconds": float(np.median([row["fit_seconds"] for row in selected])),
            })
    summary["validation"] = {"max_likelihood_error": maximum_likelihood_error, "nn_fits": len(fits)}
    with (folder / "nn_sampling_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(individual[-1]))
        writer.writeheader()
        writer.writerows(individual)
    (folder / "nn_sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for index, (name, colour) in enumerate(zip(names, colours)):
        selected = [row for row in individual if row["group"] == name]
        axes[0].errorbar(index + np.linspace(-0.15, 0.15, 6),
                         [row["log_Z"] for row in selected],
                         yerr=[row["log_Z_uncert"] for row in selected], fmt="o", color=colour)
        axes[1].scatter([row["evaluations"] / 1e6 for row in selected],
                        [abs(row["log_Z"]) for row in selected], label=name, color=colour)
        widths = [
            summary[name]["median_widths"][f"x{i}"] / (0.05 if i == 5 else np.sqrt(1 / 12))
            for i in (1, 16, 17, 5)
        ]
        axes[2].semilogy(range(4), widths, "o-", color=colour)
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(
        xticks=range(5),
        xticklabels=["Isotropic", "Exact\nHessian", "GP 128", "NN\nstored", "NN\nprobes"],
        ylabel="log evidence (truth ≈ 0)",
    )
    axes[1].set(xlabel="True likelihood calls, including probes (millions)",
                ylabel="Absolute log-evidence error")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Median posterior SD / true SD")
    fig.suptitle("18D ridge, seeds 0–5: 150 roots, 90 slices, float64 cube\n"
                 "NN arms stop prematurely; their smaller call counts do not indicate efficiency")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"nn_sampling_comparison.{extension}", dpi=160)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for row_index, name in enumerate(names[3:]):
        for seed in range(6):
            points, weights = snapshots[name, seed]
            for column, coordinate in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, coordinate])
                axes[row_index, column].plot(
                    points[order, coordinate], np.cumsum(weights[order]),
                    lw=1, alpha=0.8, label=f"Seed {seed}",
                )
        for column, coordinate in enumerate((1, 16, 17, 5)):
            # Failed NN runs can put even x5 near a prior boundary. Show the
            # entire cube so those misplaced narrow marginals are not clipped.
            grid = np.linspace(0, 1, 400)
            axes[row_index, column].plot(
                grid, norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid, "k--", lw=1.5,
            )
            axes[row_index, column].set(title=f"{name}: x{coordinate}", ylim=(0, 1),
                                        xlim=(grid.min(), grid.max()))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("NN sampling posterior CDFs; dashed curve is the exact marginal")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"nn_sampling_marginals.{extension}", dpi=160)


if __name__ == "__main__":
    main()
