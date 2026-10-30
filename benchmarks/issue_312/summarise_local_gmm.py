"""Validate all local/global GMM runs and plot matched evidence and marginals."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, wasserstein_distance


def likelihood(points: np.ndarray) -> np.ndarray:
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
    return peak - 0.5 * (residual / 1e-5) ** 2 - 0.5 * np.sum(
        ((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1,
    )


def main():
    folder = Path(__file__).resolve().parent
    names = ("Global warm", "Local warm", "Global fresh", "Local fresh")
    individual, runs, all_fits = [], [], []
    snapshots = {}
    metrics = ("log_Z", "log_Z_uncert", "evaluations", "peak_gap",
               "std_x1", "std_x16", "std_x17", "std_x5",
               "w1_x1", "w1_x16", "w1_x17", "w1_x5")
    with (folder / "gp_hessian_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["group"] in ("Isotropic", "Analytic Hessian", "GP 128"):
                individual.append({
                    "group": row["group"], "seed": int(row["seed"]),
                    **{key: float(row[key]) for key in metrics},
                    "archive_sha256": row["archive_sha256"],
                })
    with (folder / "reflection_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["group"] == "Reflection + 9 slices" and int(row["seed"]) < 6:
                individual.append({
                    "group": row["group"], "seed": int(row["seed"]),
                    **{key: float(row[key]) for key in metrics},
                    "archive_sha256": row["archive_sha256"],
                    "gradient_evaluations": int(row["gradient_evaluations"]),
                })
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    quantiles = (np.arange(8192) + 0.5) / 8192
    maximum_likelihood_error = 0.0
    for fresh in (False, True):
        for selection in ("global", "local"):
            group = f"{selection.title()} {'fresh' if fresh else 'warm'}"
            for suffix in ("a", "b"):
                root = Path(f"/tmp/jaxns-gmm-{selection}100{'-fresh' if fresh else ''}-{suffix}")
                rows = [json.loads(line) for line in
                        (root / "results.jsonl").read_text().splitlines()]
                assert len(rows) == 3
                for row in rows:
                    assert row["roots"] == 150 and row["slices"] == 90 and row["phantoms"] == 0
                    assert row["fit_iterations"] == 100 and row["components"] == 10
                    assert row["warm_start"] == (not fresh) and row["correct_mh"]
                    assert row["iso_prob"] == 0.01 and row["regularisation"] == 1e-6
                    assert row["refresh_efolds"] == 1 and row["training_evaluations"] == 0
                    assert row["allocation_target"] == "uniform" and row["epochs"] == 1
                    assert row["dlogZ"] == np.log1p(1e-3) and row["unit_dtype"] == "float64"
                    assert row["depth_reached"] and row["termination_reason"] == 0
                    for name, checksum in row["source_sha256"].items():
                        source = (
                            folder.parents[1] / name if name.startswith("src/") else folder / name
                        )
                        candidates = [source.read_bytes()]
                        if name == "run_local_gmm.py":
                            candidates.append(gzip.decompress(
                                (folder / "raw/run_local_gmm.py.warm_only.gz").read_bytes(),
                            ))
                        assert checksum in {
                            hashlib.sha256(value).hexdigest() for value in candidates
                        }
                    directory = root / f"seed-{row['seed']}"
                    fits = [json.loads(line) for line in
                            (directory / "fits.jsonl").read_text().splitlines()]
                    assert len(fits) == row["geometry_updates"]
                    log_x = np.array([fit["log_X"] for fit in fits])
                    assert log_x[0] == 0 and np.all(np.abs(np.diff(log_x)) >= 1)
                    for fit in fits:
                        path = directory / f"fit-{fit['update']:03d}.npz"
                        with np.load(path) as saved:
                            points, values = saved["points"], saved["values"]
                            assert len(points) == fit["population"]
                            if fit["contour"] is not None:
                                assert np.all(values > fit["contour"])
                            np.testing.assert_allclose(
                                values, likelihood(points), rtol=1e-12, atol=5e-5,
                            )
                            covariance = saved["covariances"]
                            assert covariance.shape == (10, 18, 18)
                            assert np.all(np.isfinite(np.linalg.cholesky(covariance)))
                            masses = np.exp(saved["log_masses"])
                            np.testing.assert_allclose(masses.sum(), 1.0, atol=1e-12)
                            np.testing.assert_allclose(masses, fit["component_masses"])
                        all_fits.append({
                            **fit, "group": group, "directory": str(directory),
                            "components_above_one_percent": int(np.sum(masses >= 0.01)),
                            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        })
                    path = directory / "samples.npz"
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["archive_sha256"]
                    with np.load(path) as saved:
                        points, values = saved["samples"], saved["log_likelihood"]
                        assert points.shape == (row["samples"], 18) and points.dtype == np.float64
                        assert saved["unit_samples"].dtype == np.float64
                        np.testing.assert_array_equal(points, saved["unit_samples"])
                        assert np.all((points >= 0) & (points <= 1))
                        assert np.all(values > saved["log_L_constraints"])
                        reference = likelihood(points)
                        np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
                        maximum_likelihood_error = max(
                            maximum_likelihood_error, float(np.max(abs(values - reference))),
                        )
                        np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
                        weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                        mean = np.sum(points * weights[:, None], axis=0)
                        std = np.sqrt(np.sum((points - mean) ** 2 * weights[:, None], axis=0))
                        distances = {}
                        for i in (1, 16, 17, 5):
                            np.testing.assert_allclose(std[i], row[f"std_x{i}"], atol=1e-12)
                            target = norm.ppf(quantiles, 0.5, 0.05) if i == 5 else quantiles
                            distances[f"w1_x{i}"] = float(wasserstein_distance(
                                points[:, i], target, u_weights=weights,
                            ))
                        snapshots[group, row["seed"]] = (points, weights)
                    runs.append({**row, "group": group, "directory": str(directory)})
                    individual.append({
                        "group": group, "seed": row["seed"],
                        **{key: row[key] for key in metrics if not key.startswith("w1_")},
                        **distances, "archive_sha256": row["archive_sha256"],
                        "seconds": row["seconds"], "fit_seconds": row["fit_seconds"],
                        "final_components_above_one_percent": (
                            all_fits[-1]["components_above_one_percent"]
                        ),
                    })
    summary = {}
    for name in ("Isotropic", "Analytic Hessian", "GP 128", "Reflection + 9 slices", *names):
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
            "median_w1": {f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                          for i in (1, 16, 17, 5)},
        }
        if name in names:
            summary[name].update({
                "final_components_above_one_percent": [
                    row["final_components_above_one_percent"] for row in selected
                ],
                "median_seconds": float(np.median([row["seconds"] for row in selected])),
                "median_fit_seconds": float(np.median([row["fit_seconds"] for row in selected])),
            })
    summary["validation"] = {"max_likelihood_error": maximum_likelihood_error,
                             "gmm_runs": len(runs), "gmm_fits": len(all_fits)}
    fields = list(individual[-1]) + ["gradient_evaluations"]
    with (folder / "local_gmm_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(individual)
    (folder / "local_gmm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("runs", runs), ("fits", all_fits)):
        (folder / f"raw/local-gmm-{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
        )
    print(json.dumps(summary, indent=2))

    colours = ("#888888", "#2b8c69", "#377eb8", "#d46a30")
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
        axes[2].plot(range(4), widths, "o-", color=colour, label=name)
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(xticks=range(4), xticklabels=[name.replace(" ", "\n") for name in names],
                ylabel="log evidence (truth ≈ 0)")
    axes[1].set(xlabel="True likelihood calls (millions)", ylabel="Absolute log-evidence error")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Median posterior SD / true SD", ylim=(0, 1.05))
    fig.suptitle("18D ridge: 10-component GMM, 100 EM iterations, seeds 0–5\n"
                 "150 roots, 90 slices, float64 cube; refit after each e-fold")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"local_gmm_comparison.{extension}", dpi=160)

    fig, axes = plt.subplots(4, 4, figsize=(13, 11), constrained_layout=True)
    for row_index, name in enumerate(names):
        for seed in range(6):
            points, weights = snapshots[name, seed]
            for column, coordinate in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, coordinate])
                axes[row_index, column].plot(
                    points[order, coordinate], np.cumsum(weights[order]), lw=1, alpha=0.8,
                    label=f"Seed {seed}",
                )
        for column, coordinate in enumerate((1, 16, 17, 5)):
            grid = np.linspace(0, 1, 400)
            axes[row_index, column].plot(
                grid, norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid, "k--", lw=1.5,
            )
            axes[row_index, column].set(title=f"{name}: x{coordinate}", xlim=(0, 1), ylim=(0, 1))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("GMM posterior CDFs; dashed curve is the exact marginal")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"local_gmm_marginals.{extension}", dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, colour in zip(names, colours):
        for seed in range(6):
            rows = [row for row in all_fits if row["group"] == name and row["seed"] == seed]
            axis = axes[int("fresh" in name)]
            axis.plot([-row["log_X"] for row in rows],
                      [row["components_above_one_percent"] for row in rows],
                      color=colour, alpha=0.6, label=name if seed == 0 else None)
    for axis, label in zip(axes, ("Warm starts", "Fresh starts")):
        axis.set(title=label, xlabel="Compression −log X", ylabel="Components with mass ≥ 1%",
                 ylim=(0, 11))
        axis.legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"local_gmm_components.{extension}", dpi=160)


if __name__ == "__main__":
    main()
