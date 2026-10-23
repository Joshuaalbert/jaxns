"""Validate the precision control and summarize observed Dynesty mechanisms."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


def main():
    folder = Path(__file__).resolve().parent
    raw = folder / "raw"
    precision = {}
    comparison_fields = (
        "samples", "evaluations", "log_Z", "log_Z_uncert", "peak_gap",
        "std_x1", "std_x16", "std_x17", "std_x5", "posterior_mean_log_X_error",
    )
    historical = {
        row["seed"]: row
        for path in raw.glob("curved-isotropic-*.jsonl")
        for line in path.read_text().splitlines()
        for row in [json.loads(line)]
    }
    csv_rows = []
    for label, dtype in (("f32", "float32"), ("f64", "float64")):
        rows = [
            json.loads(line)
            for path in sorted(raw.glob(f"precision-{label}-*.jsonl"))
            for line in path.read_text().splitlines()
        ]
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != list(range(30)):
            raise ValueError(f"Expected exactly seeds 0–29 for {dtype}.")
        largest_error = 0.0
        largest_posterior_error = 0.0
        for row in rows:
            if not (
                row["unit_dtypes"] == [dtype]
                and row["physical_dtype"] == row["log_likelihood_dtype"] == "float64"
                and row["evidence_dtype"] == "float64"
                and row["mode"] == "isotropic" and row["roots"] == 150
                and row["slices"] == 90 and row["width"] is None
                and row["allocation_target"] == "uniform"
                and row["target_log_Z_uncert"] is None
                and row["epochs"] == 1 and row["goal_reached"]
                and row["termination_reason"] == 0 and row["geometry_updates"] == 0
                and row["curvature"] == 0.3 and row["sigma_stiff"] == 1e-5
            ):
                raise ValueError(
                    f"Unexpected configuration or return status: {dtype}, {row['seed']}"
                )
            archive = Path(f"/tmp/issue312-precision-{label}/seed-{row['seed']}.npz")
            with np.load(archive) as saved:
                x = saved["samples"]
                u = saved["unit_samples"]
                log_l = saved["log_likelihood"]
                if str(u.dtype) != dtype or len(x) != row["samples"]:
                    raise ValueError("Saved cube precision or sample count differs from metadata.")
                np.testing.assert_array_equal(x, u)
                if np.any((u < 0) | (u > 1)):
                    raise ValueError("Saved cube coordinates are out of bounds.")
                weights = np.exp(saved["log_weights"] - logsumexp(saved["log_weights"]))
                mean = np.sum(weights[:, None] * x, axis=0)
                std = np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0))
                for coordinate in (1, 5, 16, 17):
                    np.testing.assert_allclose(
                        std[coordinate], row[f"std_x{coordinate}"], atol=1e-12,
                    )
                peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(.05 * np.sqrt(2 * np.pi))
                centre = .5 + .3 * np.sin(2 * np.pi * x[:, 1])
                exact = peak - .5 * ((x[:, 0] - centre) / 1e-5) ** 2
                exact -= .5 * np.sum(((x[:, 2:16] - .5) / .05) ** 2, axis=1)
                error = np.abs(exact - log_l)
                largest_error = max(largest_error, float(error.max()))
                largest_posterior_error = max(
                    largest_posterior_error, float(error[log_l > peak - 50].max()),
                )
                # TFP's scalar Normal(0.5, 0.05) constants are float32 in both
                # arms. Their tiny normalization difference is unrelated to
                # the cube-coordinate intervention; bound it independently.
                np.testing.assert_allclose(exact, log_l, rtol=1e-12, atol=5e-5)
                np.testing.assert_allclose(peak - log_l.max(), row["peak_gap"], atol=1e-12)
            csv_rows.append({
                "dtype": dtype, "seed": row["seed"],
                **{field: row[field] for field in comparison_fields},
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            })
        values = np.asarray([row["log_Z"] for row in rows])
        precision[dtype] = {
            "mean_log_Z": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mean_calls": float(np.mean([row["evaluations"] for row in rows])),
            "mean_quoted_uncertainty": float(np.mean([row["log_Z_uncert"] for row in rows])),
            "median_peak_gap": float(np.median([row["peak_gap"] for row in rows])),
            "within_two_quoted_sd": sum(
                abs(row["log_Z"]) <= 2 * row["log_Z_uncert"] for row in rows
            ),
            "median_widths": {
                f"x{coordinate}": float(np.median([row[f"std_x{coordinate}"] for row in rows]))
                for coordinate in (1, 5, 16, 17)
            },
            "max_likelihood_reference_error": largest_error,
            "max_interior_likelihood_reference_error": largest_posterior_error,
            "matches_historical_f32": sum(
                all(row[field] == historical[row["seed"]][field] for field in comparison_fields)
                for row in rows
            ) if label == "f32" else None,
            "rows": rows,
        }

    # Seed identities are paired, but changing the random-number dtype changes
    # draws from the initial population onward: these are not shared trajectories.
    bootstrap = np.random.default_rng(31264).integers(0, 30, size=(10000, 30))
    z32 = np.array([row["log_Z"] for row in precision["float32"]["rows"]])
    z64 = np.array([row["log_Z"] for row in precision["float64"]["rows"]])
    changes = np.sqrt(np.mean(z64[bootstrap] ** 2, axis=1))
    changes -= np.sqrt(np.mean(z32[bootstrap] ** 2, axis=1))
    precision["rmse_change_f64_minus_f32_95_percentile"] = np.quantile(
        changes, [.025, .975],
    ).tolist()
    with (folder / "precision_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (folder / "precision_summary.json").write_text(json.dumps(precision, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for index, (dtype, colour) in enumerate((("float32", "#3366aa"), ("float64", "#bd6425"))):
        rows = precision[dtype]["rows"]
        jitter = np.linspace(-.13, .13, 30)
        axes[0].scatter(index + jitter, [row["log_Z"] for row in rows], color=colour, alpha=.8)
        for j, coordinate in enumerate((1, 16, 17, 5)):
            reference = .05 if coordinate == 5 else 1 / np.sqrt(12)
            values = [row[f"std_x{coordinate}"] / reference for row in rows]
            axes[1].scatter(j + (index - .5) * .24 + jitter * .4, values,
                            color=colour, alpha=.65, s=17, label=dtype if j == 0 else None)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xticks=[0, 1], xticklabels=["float32 cube", "float64 cube"],
                ylabel="ln Z error (nats)", title="30 seeds per precision")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set(xticks=range(4), xticklabels=["x1", "x16", "x17", "x5"],
                ylabel="Posterior SD / correct SD", title="Posterior width recovery")
    axes[1].legend()
    fig.suptitle("JAXNS: 150 roots, 90 isotropic slices, one default depth epoch")
    fig.savefig(folder / "precision_comparison.png", dpi=160)
    fig.savefig(folder / "precision_comparison.pdf")
    plt.close(fig)

    runtime = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for seed in range(6):
        path = raw / f"dynesty-runtime-150-{seed}.json.gz"
        trace = json.loads(gzip.decompress(path.read_bytes()))
        if not trace["matches_archived_samples_and_weights"]:
            raise ValueError("Instrumented replay did not match its original archive.")
        chains = trace["chains"]
        bounds = trace["bounds"]
        count = 90 * len(chains)
        counts, occurrences = np.unique([b["nells"] for b in bounds], return_counts=True)
        bulk = [c for c in chains if peak - c["log_L_constraint"] < 30]
        row = {
            "seed": seed, "config": trace["config"], "chains": len(chains),
            "bound_updates": len(bounds), "forced_updates": sum(b["forced"] for b in bounds),
            "ellipsoid_counts": dict(zip(map(str, counts), map(int, occurrences))),
            "bulk_chain_ellipsoid_counts": sorted({c["nells"] for c in bulk}),
            "first_slice_iteration": chains[0]["iteration"],
            "median_refresh_efolds": float(
                np.median(np.diff([b["iteration"] for b in bounds])) * np.log(151 / 150)
            ),
            "doubling_switch_iterations": [
                c["iteration"] for c in chains if c["expansion_warnings"]
            ],
            **{f"{key}_per_slice": sum(c[key] for c in chains) / count for key in (
                "actual_calls", "native_calls", "expansion_tests", "contraction_tests",
                "doubling_acceptance_tests", "doubling_steps",
            )},
        }
        runtime.append(row)
        axes[0].step([b["iteration"] * np.log(151 / 150) for b in bounds],
                     [b["nells"] for b in bounds], where="post", alpha=.65, label=f"seed {seed}")
        compression = np.array([c["iteration"] * np.log(151 / 150) for c in chains])
        costs = np.array([c["actual_calls"] / 90 for c in chains])
        centres, averages = [], []
        for left in range(int(np.floor(compression.min())), int(np.ceil(compression.max()))):
            selected = (compression >= left) & (compression < left + 1)
            if selected.any():
                centres.append(left + .5)
                averages.append(costs[selected].mean())
        axes[1].plot(centres, averages, alpha=.8)
    axes[0].set(ylabel="Fitted ellipsoids", yticks=[1, 2, 3, 4],
                title="Dynesty 3.0.0: exact replays of the 150-live-point controls")
    axes[0].legend(ncol=3)
    axes[1].set(xlabel="Inferred compression −ln X", ylabel="Actual likelihood calls / slice")
    fig.savefig(folder / "dynesty_runtime.png", dpi=160)
    fig.savefig(folder / "dynesty_runtime.pdf")
    plt.close(fig)
    (folder / "dynesty_runtime_summary.json").write_text(
        json.dumps(runtime, indent=2) + "\n"
    )
    for dtype in ("float32", "float64"):
        print(dtype, json.dumps({
            key: value for key, value in precision[dtype].items() if key != "rows"
        }, indent=2))
    print("RMSE change 95% interval", precision["rmse_change_f64_minus_f32_95_percentile"])
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
