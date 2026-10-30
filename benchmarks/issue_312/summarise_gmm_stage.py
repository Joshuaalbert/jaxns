"""Summarise paired GMM/isotropic continuations from identical evidence states."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


def main():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(__file__).parent
    rows = []
    fits = []
    groups = {name: [] for name in ("evidence", "isotropic", "gmm")}
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for seed in range(3):
        baseline = None
        checksum = None
        for mode, linestyle in (("isotropic", "--"), ("gmm", "-")):
            prefix = f"staged-gmm10-180-{seed}.{mode}"
            path = directory / "raw" / f"{prefix}.jsonl"
            lines = path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError(f"Expected one final record in {path}.")
            record = json.loads(lines[0])
            if not (
                record["seed"] == seed and record["mode"] == mode
                and record["slices"] == 180 and record["components"] == 10
                and record["fit_iterations"] == 10 and record["iso_prob"] == 0.01
                and record["regularisation"] == 1e-6
                and record["allocation_target"] == "posterior_improving"
                and not record["covariance_whitening"] and record["goal_reached"]
                and record["termination_reason"] == 0
                and record["final"]["ess"] >= record["target_ess"]
                and record["baseline"]["log_Z_uncert"] <= 0.1
            ):
                raise ValueError(f"Unexpected configuration or unfinished goal in {path}.")
            if baseline is None:
                baseline = record["baseline"]
                checksum = record["baseline_checksum"]
                groups["evidence"].append(baseline)
                rows.append({"seed": seed, "stage": "evidence", **{
                    key: baseline[key] for key in (
                        "log_Z", "log_Z_uncert", "ess", "evaluations", "peak_gap",
                        "std_x1", "std_x16", "std_x17", "std_x5",
                        "w1_x1", "w1_x16", "w1_x17",
                    )
                }, "additional_evaluations": 0, "ess_ratio": 1.0,
                    "isotropic_direction_fraction": 1.0})
            elif checksum != record["baseline_checksum"] or baseline != record["baseline"]:
                raise ValueError(f"Seed {seed} arms do not share the same baseline.")
            np.testing.assert_allclose(record["target_ess"], 2 * baseline["ess"], rtol=1e-12)
            final = record["final"]
            groups[mode].append(final)
            isotropic_fraction = 1.0
            if mode == "gmm":
                if record["num_directions"] <= 0:
                    raise ValueError(f"No GMM-enabled direction attempts for seed {seed}.")
                isotropic_fraction = record["num_isotropic"] / record["num_directions"]
            rows.append({"seed": seed, "stage": mode, **{
                key: final[key] for key in (
                    "log_Z", "log_Z_uncert", "ess", "evaluations", "peak_gap",
                    "std_x1", "std_x16", "std_x17", "std_x5",
                    "w1_x1", "w1_x16", "w1_x17",
                )
            }, "additional_evaluations": record["additional_evaluations"],
                "ess_ratio": final["ess"] / baseline["ess"],
                "isotropic_direction_fraction": isotropic_fraction})
            progress_path = directory / "raw" / f"{prefix}.progress.jsonl"
            progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
            progress = sorted(
                {row["epoch"]: row for row in progress}.values(),
                key=lambda row: row["epoch"],
            )
            if mode == "gmm":
                fit = json.loads((directory / "raw" / f"{prefix}.fit.json").read_text())
                if not (
                    len(fit["valid"]) == 10 and all(fit["valid"])
                    and fit["num_updates"] == 1
                    and all(row["fit_updates"] == 1 for row in progress)
                ):
                    raise ValueError(f"Seed {seed} must use one frozen ten-component fit.")
                centres = np.asarray(fit["centres"])
                # These are conditional probabilities of choosing a fitted
                # component, not observed component counts or mixture masses.
                fits.append({
                    "seed": seed,
                    "min_centre_x1": float(centres[:, 1].min()),
                    "max_centre_x1": float(centres[:, 1].max()),
                    "largest_mixture_mass": max(fit["mixture_masses"]),
                    "largest_selection_probability_by_peak_gap": {
                        gap: max(probabilities)
                        for gap, probabilities in
                        fit["selection_probabilities_by_peak_gap"].items()
                    },
                    "isotropic_direction_fraction": isotropic_fraction,
                })
            progress = [
                {**baseline, "additional_evaluations": 0},
                *[row for row in progress if row["additional_evaluations"] > 0],
            ]
            calls = np.asarray([row["additional_evaluations"] for row in progress]) / 1e6
            values = (
                [row["log_Z"] for row in progress],
                [row["ess"] / baseline["ess"] for row in progress],
                [row["std_x16"] * np.sqrt(12) for row in progress],
                [row["std_x17"] * np.sqrt(12) for row in progress],
            )
            for axis, value in zip(axes, values):
                axis.plot(calls, value, linestyle, color=f"C{seed}",
                          label=f"{mode}, seed {seed}", linewidth=1.5)
                axis.plot(calls[-1], value[-1], "o", color=f"C{seed}", markersize=3)
    summaries = []
    for stage, records in groups.items():
        errors = np.asarray([row["log_Z"] for row in records])
        summaries.append({
            "stage": stage,
            "seeds": len(records),
            "mean_log_Z": float(errors.mean()),
            "rmse_log_Z": float(np.sqrt(np.mean(errors ** 2))),
            "within_two_sigma": sum(abs(row["log_Z"]) <= 2 * row["log_Z_uncert"]
                                    for row in records),
            "mean_log_Z_uncert": float(np.mean([row["log_Z_uncert"] for row in records])),
            "mean_evaluations": float(np.mean([row["evaluations"] for row in records])),
            **{
                f"median_{key}": float(np.median([row[key] for row in records]))
                for key in ("std_x1", "std_x16", "std_x17", "std_x5", "peak_gap",
                            "w1_x1", "w1_x16", "w1_x17")
            },
        })
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("log Z (true value: 0)")
    axes[1].axhline(2, color="black", linewidth=0.8)
    axes[1].set_ylabel("Kish ESS / evidence-stage ESS")
    axes[1].legend(loc="upper left", fontsize=8)
    for axis, coordinate in zip(axes[2:], (16, 17)):
        axis.axhline(1, color="black", linewidth=0.8)
        axis.set_ylabel(f"x{coordinate} width / uniform-prior width")
    for axis in axes:
        axis.set_xlabel("Additional likelihood evaluations (millions)")
        axis.grid(alpha=0.15)
    figure.suptitle("10-component GMM after uncertainty 0.1: continue to twice the Kish ESS")
    figure.savefig(directory / "staged_gmm_progress.png", dpi=180)
    figure.savefig(directory / "staged_gmm_progress.pdf")
    plt.close(figure)
    with (directory / "staged_gmm_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output = {"groups": summaries, "runs": rows, "fits": fits}
    (directory / "staged_gmm_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
