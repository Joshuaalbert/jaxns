"""Compare 10 and 100 EM iterations from identical evidence-stage checkpoints."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


def main():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(__file__).parent
    # Reuse the completed controls. A fresh evidence run would change the
    # fitting population and obscure the effect of the EM iteration count.
    arms = (
        ("isotropic", "staged-gmm10", "isotropic", 10, ":"),
        ("gmm-10", "staged-gmm10", "gmm", 10, "--"),
        ("gmm-100", "staged-gmm100", "gmm", 100, "-"),
    )
    groups = {name: [] for name in ("evidence", "isotropic", "gmm-10", "gmm-100")}
    rows, fits = [], []
    metrics = (
        "log_Z", "log_Z_uncert", "ess", "evaluations", "peak_gap",
        "std_x1", "std_x16", "std_x17", "std_x5", "w1_x1", "w1_x16", "w1_x17",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for seed in range(3):
        baseline, checksum = None, None
        for label, prefix, mode, iterations, linestyle in arms:
            stem = f"{prefix}-180-{seed}.{mode}"
            path = directory / "raw" / f"{stem}.jsonl"
            lines = path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError(f"Expected one completed record in {path}.")
            record = json.loads(lines[0])
            if not (
                record["seed"] == seed and record["mode"] == mode
                and record["slices"] == 180 and record["components"] == 10
                and record["fit_iterations"] == iterations
                and record["iso_prob"] == 0.01 and record["regularisation"] == 1e-6
                and record["allocation_target"] == "posterior_improving"
                and not record["covariance_whitening"] and record["goal_reached"]
                and record["termination_reason"] == 0
                and record["baseline"]["log_Z_uncert"] <= 0.1
                and record["final"]["ess"] >= record["target_ess"]
            ):
                raise ValueError(f"Unexpected configuration or unfinished goal in {path}.")
            if baseline is None:
                baseline, checksum = record["baseline"], record["baseline_checksum"]
                groups["evidence"].append(baseline)
                rows.append({
                    "seed": seed, "stage": "evidence",
                    **{key: baseline[key] for key in metrics},
                    "additional_evaluations": 0, "ess_ratio": 1.0,
                    "stage_epochs": 0, "fit_seconds": 0.0,
                    "isotropic_direction_fraction": 1.0,
                })
            elif baseline != record["baseline"] or checksum != record["baseline_checksum"]:
                raise ValueError(f"Seed {seed} arms do not share exactly the same baseline.")
            np.testing.assert_allclose(record["target_ess"], 2 * baseline["ess"], rtol=1e-12)
            final = record["final"]
            if record["additional_evaluations"] != final["evaluations"] - baseline["evaluations"]:
                raise ValueError(f"Inconsistent evaluation count in {path}.")
            groups[label].append(final)
            progress_path = directory / "raw" / f"{stem}.progress.jsonl"
            progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
            # A resumed run can report its starting boundary again. Deduplicate
            # that boundary without treating it as an additional allocation epoch.
            progress = sorted({item["epoch"]: item for item in progress}.values(),
                              key=lambda item: item["epoch"])
            np.testing.assert_allclose(progress[-1]["ess"], final["ess"], rtol=1e-12)
            isotropic_fraction = 1.0
            if mode == "gmm":
                fit = json.loads((directory / "raw" / f"{stem}.fit.json").read_text())
                if not (
                    len(fit["valid"]) == 10 and fit["num_updates"] == 1
                    and sum(fit["valid"]) == record["fit_valid_components"]
                    and all(item["fit_updates"] == 1 for item in progress)
                    and record["num_directions"] > 0
                ):
                    raise ValueError(f"Expected one frozen ten-component fit in {path}.")
                isotropic_fraction = record["num_isotropic"] / record["num_directions"]
                centres = np.asarray(fit["centres"])
                fits.append({
                    "seed": seed, "stage": label, "valid_components": sum(fit["valid"]),
                    "fit_seconds": record["fit_seconds"],
                    "min_centre_x1": float(centres[:, 1].min()),
                    "max_centre_x1": float(centres[:, 1].max()),
                    "largest_selection_probability_by_peak_gap": {
                        gap: max(probabilities) for gap, probabilities in
                        fit["selection_probabilities_by_peak_gap"].items()
                    },
                    "isotropic_direction_fraction": isotropic_fraction,
                })
            rows.append({
                "seed": seed, "stage": label,
                **{key: final[key] for key in metrics},
                "additional_evaluations": record["additional_evaluations"],
                "ess_ratio": final["ess"] / baseline["ess"],
                "stage_epochs": record["stage_epochs"], "fit_seconds": record["fit_seconds"],
                "isotropic_direction_fraction": isotropic_fraction,
            })
            trajectory = [
                {**baseline, "additional_evaluations": 0},
                *[item for item in progress if item["additional_evaluations"] > 0],
            ]
            calls = np.asarray([item["additional_evaluations"] for item in trajectory]) / 1e6
            values = (
                [item["log_Z"] for item in trajectory],
                [item["ess"] / baseline["ess"] for item in trajectory],
                [item["peak_gap"] for item in trajectory],
                *[[item[f"std_x{dimension}"] * np.sqrt(12) for item in trajectory]
                  for dimension in (1, 16, 17)],
            )
            for axis, value in zip(axes.flat, values):
                axis.plot(calls, value, linestyle, color=f"C{seed}", linewidth=1.5,
                          label=f"{label}, seed {seed}")
                axis.plot(calls[-1], value[-1], "o", color=f"C{seed}", markersize=3)

    summaries = []
    for stage, records in groups.items():
        errors = np.asarray([item["log_Z"] for item in records])
        stage_rows = [item for item in rows if item["stage"] == stage]
        summaries.append({
            "stage": stage, "seeds": len(records),
            "mean_log_Z": float(errors.mean()),
            "rmse_log_Z": float(np.sqrt(np.mean(errors ** 2))),
            "within_two_sigma": sum(abs(item["log_Z"]) <= 2 * item["log_Z_uncert"]
                                    for item in records),
            "mean_log_Z_uncert": float(np.mean([item["log_Z_uncert"] for item in records])),
            "mean_evaluations": float(np.mean([item["evaluations"] for item in records])),
            "mean_additional_evaluations": float(np.mean([
                item["additional_evaluations"] for item in stage_rows
            ])),
            **{
                f"median_{key}": float(np.median([item[key] for item in records]))
                for key in ("std_x1", "std_x16", "std_x17", "std_x5", "peak_gap",
                            "w1_x1", "w1_x16", "w1_x17")
            },
        })
    labels = (
        "log Z (true value: 0)", "Kish ESS / baseline ESS", "Best log L below the peak",
        "x1 width / uniform-prior width", "x16 width / uniform-prior width",
        "x17 width / uniform-prior width",
    )
    for axis, ylabel, target in zip(axes.flat, labels, (0, 2, 0, 1, 1, 1)):
        axis.axhline(target, color="black", linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Additional likelihood evaluations (millions)")
        axis.grid(alpha=0.15)
    axes[0, 1].legend(fontsize=7, loc="lower right")
    figure.suptitle("10 versus 100 EM iterations: same evidence states, then double Kish ESS")
    figure.savefig(directory / "gmm_iterations_progress.png", dpi=180)
    figure.savefig(directory / "gmm_iterations_progress.pdf")
    plt.close(figure)
    with (directory / "gmm_iterations_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output = {"groups": summaries, "runs": rows, "fits": fits}
    (directory / "gmm_iterations_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
