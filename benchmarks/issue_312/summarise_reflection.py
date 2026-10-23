"""Validate all new reflection archives and plot matched ridge comparisons."""

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
    baseline_rows = list(csv.DictReader((folder / "hessian_summary.csv").open()))
    rows = []
    for row in baseline_rows:
        if row["group"] in ("Isotropic", "Hessian, contour scaled"):
            rows.append({
                **{key: float(value) for key, value in row.items()
                   if key not in ("group", "archive_sha256")},
                "group": row["group"], "seed": int(row["seed"]),
                "gradient_evaluations": 0, "archive_sha256": row["archive_sha256"],
            })
    # The first runs predate a lint-only unused-variable rename. Preserve and
    # verify the archived executed source, including after a squash merge.
    original = gzip.decompress((folder / "raw/reflection_sampler.initial.py.gz").read_bytes())
    sampler_hashes = {
        hashlib.sha256(original).hexdigest(),
        hashlib.sha256((folder / "reflection_sampler.py").read_bytes()).hexdigest(),
    }
    snapshots = {}
    maximum_error = 0.0
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    quantiles = (np.arange(8192) + 0.5) / 8192
    for kind, letters, slices in (("hug", "ab", 10), ("short", "abcde", 1)):
        name = f"Reflection + {slices * 9} slices"
        for letter in letters:
            path = folder / f"raw/reflection-{kind}-{letter}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                assert row["source_sha256"]["reflection_sampler.py"] in sampler_hashes
                for source in ("run_reflection.py", "run_ridge.py"):
                    assert row["source_sha256"][source] == hashlib.sha256(
                        (folder / source).read_bytes(),
                    ).hexdigest()
                assert row["cycles"] == 9 and row["bounces"] == 10
                assert row["slices_per_cycle"] == slices and row["step_size"] == 0.003
                assert row["roots"] == 150 and row["allocation_target"] == "uniform"
                assert row["phantoms"] == 0 and row["dlogZ"] == float(np.log1p(1e-3))
                assert row["epochs"] == 1 and row["depth_reached"]
                assert row["termination_reason"] == 0
                assert row["gradient_evaluations"] == (row["samples"] - 150) * 90
                assert row["evaluations"] == (
                    row["value_only_evaluations"] + row["gradient_evaluations"]
                )
                archive = Path(f"/tmp/jaxns-reflection-{kind}-{letter}/seed-{row['seed']}.npz")
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
                with np.load(archive) as saved:
                    points = saved["samples"]
                    values = saved["log_likelihood"]
                    assert points.dtype == np.float64 and points.shape == (row["samples"], 18)
                    np.testing.assert_array_equal(points, saved["unit_samples"])
                    assert np.all((points >= 0) & (points <= 1))
                    assert np.all(values > saved["log_L_constraints"])
                    residual = points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1])
                    reference = peak - 0.5 * (
                        (residual / 1e-5) ** 2
                        + np.sum(((points[:, 2:16] - 0.5) / 0.05) ** 2, axis=1)
                    )
                    np.testing.assert_allclose(values, reference, rtol=1e-12, atol=5e-5)
                    maximum_error = max(maximum_error, float(np.max(abs(values - reference))))
                    np.testing.assert_allclose(peak - values.max(), row["peak_gap"])
                    weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                    mean = np.sum(weights[:, None] * points, axis=0)
                    std = np.sqrt(np.sum(weights[:, None] * (points - mean) ** 2, axis=0))
                    for coordinate in (1, 16, 17, 5):
                        np.testing.assert_allclose(std[coordinate], row[f"std_x{coordinate}"])
                        reference = (norm.ppf(quantiles, loc=0.5, scale=0.05)
                                     if coordinate == 5 else quantiles)
                        row[f"w1_x{coordinate}"] = float(wasserstein_distance(
                            points[:, coordinate], reference, u_weights=weights,
                        ))
                    snapshots[name, row["seed"]] = (points, weights)
                rows.append({**row, "group": name})

    summary = {}
    names = (
        "Isotropic", "Hessian, contour scaled", "Reflection + 9 slices", "Reflection + 90 slices",
    )
    for name in names:
        selected = sorted(
            [row for row in rows if row["group"] == name], key=lambda row: row["seed"],
        )
        count = 6 if name == "Reflection + 90 slices" else 30
        assert [row["seed"] for row in selected] == list(range(count))
        z = np.array([row["log_Z"] for row in selected])
        summary[name] = {
            "seeds": count, "mean_log_Z": float(z.mean()), "rmse": float(np.sqrt(np.mean(z ** 2))),
            "range_log_Z": [float(z.min()), float(z.max())],
            "mean_evaluations": float(np.mean([row["evaluations"] for row in selected])),
            "mean_gradients": float(np.mean([row["gradient_evaluations"] for row in selected])),
            "mean_quoted_uncertainty": float(np.mean([row["log_Z_uncert"] for row in selected])),
            "median_peak_gap": float(np.median([row["peak_gap"] for row in selected])),
            "within_two_quoted_sd": sum(
                abs(row["log_Z"]) <= 2 * row["log_Z_uncert"] for row in selected
            ),
            "median_widths": {f"x{i}": float(np.median([row[f"std_x{i}"] for row in selected]))
                              for i in (1, 16, 17, 5)},
            "median_wasserstein": {f"x{i}": float(np.median([row[f"w1_x{i}"] for row in selected]))
                                   for i in (1, 16, 17, 5)},
        }
    summary["validation"] = {"new_archives": 36, "max_likelihood_error": maximum_error}
    (folder / "reflection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ("group", "seed", "log_Z", "log_Z_uncert", "evaluations", "gradient_evaluations",
              "peak_gap", "std_x1", "std_x16", "std_x17", "std_x5", "w1_x1", "w1_x16",
              "w1_x17", "w1_x5", "archive_sha256")
    with (folder / "reflection_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))

    colours = ["#888888", "#2b8c69", "#377eb8"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for index, (name, colour) in enumerate(zip(names[:3], colours)):
        selected = [row for row in rows if row["group"] == name]
        axes[0].errorbar(
            index + np.linspace(-0.18, 0.18, 30), [row["log_Z"] for row in selected],
            yerr=[row["log_Z_uncert"] for row in selected], fmt=".", color=colour,
        )
        axes[1].scatter([row["evaluations"] / 1e6 for row in selected],
                        [abs(row["log_Z"]) for row in selected], color=colour, s=18, label=name)
        widths = summary[name]["median_widths"]
        axes[2].plot(
            range(4), [widths[f"x{i}"] / (0.05 if i == 5 else np.sqrt(1 / 12))
                       for i in (1, 16, 17, 5)], "o-", color=colour,
        )
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(
        xticks=range(3), xticklabels=["Isotropic", "Scaled\nHessian", "Reflection\n+ 9 slices"],
        ylabel="log evidence (truth ≈ 0)",
    )
    axes[1].set(xlabel="Model calls, including gradient calls (millions)",
                ylabel="Absolute log-evidence error")
    axes[1].legend(fontsize=7)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Median posterior SD / true SD")
    fig.suptitle(
        "18D ridge, 30 matched seeds: reflections improve evidence; "
        "flat directions still mix poorly",
    )
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"reflection_comparison.{extension}", dpi=160)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for row_index, name in enumerate(("Reflection + 90 slices", "Reflection + 9 slices")):
        for seed in range(6 if row_index == 0 else 30):
            points, weights = snapshots[name, seed]
            for column, coordinate in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, coordinate])
                axes[row_index, column].plot(points[order, coordinate], np.cumsum(weights[order]),
                                             color="#377eb8", alpha=0.35, lw=0.8)
        for column, coordinate in enumerate((1, 16, 17, 5)):
            grid = np.linspace(0.3, 0.7, 200) if coordinate == 5 else np.linspace(0, 1, 200)
            axes[row_index, column].plot(
                grid, norm.cdf(grid, 0.5, 0.05) if coordinate == 5 else grid, "k--", lw=1.5,
            )
            axes[row_index, column].set(title=f"{name}: x{coordinate}", ylim=(0, 1),
                                        xlim=(grid.min(), grid.max()))
    fig.suptitle("Posterior CDFs; dashed line is the exact marginal")
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"reflection_marginals.{extension}", dpi=160)


if __name__ == "__main__":
    main()
