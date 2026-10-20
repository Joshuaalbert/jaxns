"""Summarise evidence-improving allocation at the requested uncertainty goal."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


def main():
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(__file__).resolve().parent
    rows = []
    summaries = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for mode, linestyle in (("isotropic", "--"), ("covariance", "-")):
        records = []
        for seed in range(3):
            path = directory / "raw" / f"evidence-{mode}-{seed}.jsonl"
            lines = path.read_text().splitlines()
            if len(lines) != 1:
                raise ValueError(f"Expected one completed seed in {path}.")
            record = json.loads(lines[0])
            if not (
                record["seed"] == seed and record["mode"] == mode
                and record["allocation_target"] == "evidence_improving"
                and record["target_log_Z_uncert"] == 0.1
                and record["goal_reached"] and record["termination_reason"] == 0
                and record["log_Z_uncert"] <= 0.1
            ):
                raise ValueError(f"Run did not reach the intended goal: {path}.")
            records.append(record)
            rows.append({
                key: record[key] for key in (
                    "mode", "seed", "epochs", "samples", "evaluations", "log_Z",
                    "log_Z_uncert", "peak_gap", "std_x1", "std_x16", "std_x17",
                    "std_x5", "geometry_updates",
                )
            })
            progress = [
                json.loads(line)
                for line in path.with_suffix(".progress.jsonl").read_text().splitlines()
            ]
            # A checkpoint resume may repeat its last coherent boundary. The
            # latest record for each epoch retains the complete scientific path.
            progress = list({
                row["epoch"]: row for row in progress
            }.values())
            progress.sort(key=lambda row: row["epoch"])
            calls = np.asarray([row["evaluations"] for row in progress]) / 1e6
            label = f"{mode}, seed {seed}"
            for axis, key in zip(axes, ("log_Z", "log_Z_uncert")):
                axis.plot(
                    calls, [row[key] for row in progress], linestyle,
                    color=f"C{seed}", label=label, linewidth=1.5,
                )
        errors = np.asarray([row["log_Z"] for row in records])
        summaries.append({
            "mode": mode,
            "seeds": len(records),
            "mean_log_Z": float(errors.mean()),
            "rmse_log_Z": float(np.sqrt(np.mean(errors ** 2))),
            "mean_quoted_uncertainty": float(np.mean([
                row["log_Z_uncert"] for row in records
            ])),
            "within_two_quoted_sd": sum(
                abs(row["log_Z"]) <= 2 * row["log_Z_uncert"] for row in records
            ),
            "mean_evaluations": float(np.mean([
                row["evaluations"] for row in records
            ])),
            **{
                f"median_{key}": float(np.median([row[key] for row in records]))
                for key in ("std_x1", "std_x16", "std_x17", "std_x5")
            },
        })
    axes[0].axhline(0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_ylabel("Classic log Z (truth = 0)")
    axes[1].axhline(0.1, color="black", linewidth=1, alpha=0.6)
    axes[1].set_ylabel("Quoted log Z uncertainty")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Cumulative logical likelihood evaluations (millions)")
        axis.set_xscale("log")
        axis.grid(alpha=0.15)
    axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle("Evidence-improving allocation: 18-D curved ridge")
    figure.savefig(directory / "evidence_progress.png", dpi=180)
    figure.savefig(directory / "evidence_progress.pdf")
    plt.close(figure)
    with (directory / "evidence_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output = {"groups": summaries, "runs": rows}
    (directory / "evidence_summary.json").write_text(
        json.dumps(output, indent=2) + "\n",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
