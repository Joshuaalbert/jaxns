"""Validate saved Hessian experiments and compare accuracy, work and coverage."""

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
    raw = folder / "raw"
    groups = [
        ("Isotropic", [f"precision-f64-{i}" for i in (0, 10, 20)], 30, 0.3, None),
        ("Hessian, unscaled", [f"curved-hessian-{i}" for i in (0, 10, 20)], 30, 0.3, False),
        ("Hessian, contour scaled", [f"curved-hessian-scaled-{i}" for i in (0, 10, 20)],
         30, 0.3, True),
        ("Straight, unscaled", ["straight-hessian"], 10, 0.0, False),
        ("Straight, contour scaled", ["straight-hessian-scaled"], 10, 0.0, True),
    ]
    summary, individual, samples = {}, [], {}
    fields = (
        "samples", "evaluations", "log_Z", "log_Z_uncert", "peak_gap",
        "std_x1", "std_x16", "std_x17", "std_x5", "posterior_mean_log_X_error",
    )
    quantiles = (np.arange(8192) + 0.5) / 8192
    for label, stems, expected_count, curvature, scaled in groups:
        rows = []
        for stem in stems:
            for line in (raw / f"{stem}.jsonl").read_text().splitlines():
                row = json.loads(line)
                archive_dir = "precision-f64" if scaled is None else stem
                row["archive"] = f"/tmp/issue312-{archive_dir}/seed-{row['seed']}.npz"
                rows.append(row)
        rows.sort(key=lambda row: row["seed"])
        assert [row["seed"] for row in rows] == list(range(expected_count))
        maximum_error = 0.0
        for row in rows:
            assert row["roots"] == 150 and row["slices"] == 90 and row["width"] is None
            assert row["unit_dtypes"] == ["float64"]
            assert row["physical_dtype"] == row["log_likelihood_dtype"] == "float64"
            assert row["allocation_target"] == "uniform" and row["target_log_Z_uncert"] is None
            assert row["goal_reached"] and row["epochs"] == 1 and row["termination_reason"] == 0
            assert row["curvature"] == curvature and row["sigma_stiff"] == 1e-5
            if scaled is None:
                script_source = gzip.decompress((raw / "precision_run_ridge.py.gz").read_bytes())
            elif scaled:
                assert row["hessian_contour_scale"] and row["precision_floor"] == 12.0
                script_source = (folder / "run_ridge.py").read_bytes()
                sampler_source = (folder / "hessian_sampler.py").read_bytes()
            else:
                assert row["precision_floor"] == 12.0
                script_source = gzip.decompress((raw / "run_ridge.unscaled.py.gz").read_bytes())
                sampler_source = gzip.decompress(
                    (raw / "hessian_sampler.unscaled.py.gz").read_bytes(),
                )
            assert hashlib.sha256(script_source).hexdigest() == row["script_sha256"]
            if scaled is not None:
                assert hashlib.sha256(sampler_source).hexdigest() == row["hessian_sampler_sha256"]
            archive = Path(row["archive"])
            with np.load(archive) as saved:
                x = saved["samples"]
                np.testing.assert_array_equal(x, saved["unit_samples"])
                assert x.dtype == np.float64 and len(x) == row["samples"]
                assert np.all((x >= 0) & (x <= 1))
                weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                mean = np.sum(weights[:, None] * x, axis=0)
                std = np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0))
                distances = {}
                for coordinate in (1, 16, 17, 5):
                    np.testing.assert_allclose(
                        std[coordinate], row[f"std_x{coordinate}"], atol=1e-12,
                    )
                    reference = (
                        norm.ppf(quantiles, loc=0.5, scale=0.05)
                        if coordinate == 5 else quantiles
                    )
                    distances[f"w1_x{coordinate}"] = float(wasserstein_distance(
                        x[:, coordinate], reference, u_weights=weights,
                    ))
                peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
                residual = x[:, 0] - (0.5 + curvature * np.sin(2 * np.pi * x[:, 1]))
                exact = peak - 0.5 * (residual / 1e-5) ** 2
                exact -= 0.5 * np.sum(((x[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
                maximum_error = max(
                    maximum_error, float(np.max(np.abs(exact - saved["log_likelihood"]))),
                )
                np.testing.assert_allclose(exact, saved["log_likelihood"], rtol=1e-12, atol=5e-5)
                np.testing.assert_allclose(peak - saved["log_likelihood"].max(), row["peak_gap"])
                if scaled and curvature == 0.3 and row["seed"] < 6:
                    samples[row["seed"]] = (x, weights)
            individual.append({
                "group": label, "seed": row["seed"],
                **{field: row[field] for field in fields}, **distances,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            })
        selected = [row for row in individual if row["group"] == label]
        log_z = np.array([row["log_Z"] for row in rows])
        summary[label] = {
            "seeds": expected_count,
            "mean_log_Z": float(log_z.mean()),
            "median_log_Z": float(np.median(log_z)),
            "log_Z_range": [float(log_z.min()), float(log_z.max())],
            "rmse": float(np.sqrt(np.mean(log_z ** 2))),
            "mean_evaluations": float(np.mean([row["evaluations"] for row in rows])),
            "median_peak_gap": float(np.median([row["peak_gap"] for row in rows])),
            "mean_quoted_uncertainty": float(np.mean([row["log_Z_uncert"] for row in rows])),
            "within_two_quoted_sd": int(sum(
                abs(row["log_Z"]) <= 2 * row["log_Z_uncert"] for row in rows
            )),
            "median_widths": {
                f"x{i}": float(np.median([row[f"std_x{i}"] for row in rows]))
                for i in (1, 16, 17, 5)
            },
            "median_wasserstein": {
                f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                for i in (1, 16, 17, 5)
            },
            "max_likelihood_reference_error": maximum_error,
        }
    with (folder / "hessian_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(individual[0]))
        writer.writeheader()
        writer.writerows(individual)
    (folder / "hessian_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colours = ["#3c6eb4", "#999999", "#d16a23"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for index, ((label, _, _, _, _), colour) in enumerate(zip(groups[:3], colours)):
        selected = [row for row in individual if row["group"] == label]
        jitter = np.linspace(-0.15, 0.15, len(selected))
        axes[0].scatter(index + jitter, [row["log_Z"] for row in selected],
                        color=colour, alpha=0.75)
        axes[1].scatter([row["evaluations"] / 1e6 for row in selected],
                        [abs(row["log_Z"]) for row in selected],
                        color=colour, alpha=0.75, label=label)
        if index != 1:
            for coordinate_index, coordinate in enumerate((1, 16, 17, 5)):
                correct_width = 0.05 if coordinate == 5 else 1 / np.sqrt(12)
                axes[2].scatter(coordinate_index + (index - 1) * 0.14 + jitter * 0.5,
                                [row[f"std_x{coordinate}"] / correct_width for row in selected],
                                color=colour, alpha=0.65, s=18)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xticks=[0, 1, 2], xticklabels=["Isotropic", "Hessian\nunscaled", "Hessian\nscaled"],
                ylabel="ln Z error (nats)", title="Evidence across 30 seeds")
    axes[0].set_yscale("symlog", linthresh=2)
    axes[1].set(xlabel="Likelihood evaluations (millions)", ylabel="Absolute ln Z error",
                yscale="log", title="Cost and accuracy")
    axes[1].legend(fontsize=8)
    axes[2].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Posterior SD / correct SD", title="Isotropic vs contour-scaled Hessian")
    fig.suptitle("18D curved ridge: 150 roots, 90 transitions, float64 cube, default depth epoch")
    fig.savefig(folder / "hessian_comparison.png", dpi=160)
    fig.savefig(folder / "hessian_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for ax, coordinate in zip(axes.flat, (1, 16, 17, 5)):
        edges = np.linspace(0, 1, 61)
        for seed, (points, weights) in sorted(samples.items()):
            counts, _ = np.histogram(
                points[:, coordinate], bins=edges, weights=weights, density=True,
            )
            ax.stairs(counts, edges, alpha=0.7, label=f"seed {seed}")
        if coordinate == 5:
            grid = np.linspace(0.25, 0.75, 300)
            ax.plot(grid, norm.pdf(grid, 0.5, 0.05), "k--", label="Correct density")
            ax.set_xlim(0.25, 0.75)
        else:
            ax.axhline(1, color="black", linestyle="--", label="Correct density")
        ax.set(xlabel=f"x{coordinate}", ylabel="Posterior density")
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("Contour-scaled Hessian: first six seeds, plotted separately")
    fig.savefig(folder / "hessian_marginals.png", dpi=160)
    fig.savefig(folder / "hessian_marginals.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
