"""Validate and compare classic/phantom evidence on the identical allocation runs."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    allocations = ("uniform", "evidence_improving", "posterior_improving")
    rows, summaries, intervals = [], [], []
    bootstrap = np.random.default_rng(312045).integers(0, 30, size=(10000, 30))
    for kernel in ("hessian", "reflection"):
        for allocation in allocations:
            cell = []
            for letter in "ab":
                path = folder / f"raw/allocation-phantoms-{kernel}-{allocation}-{letter}.jsonl"
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    assert row["kernel"] == kernel and row["allocation"] == allocation
                    assert row["classic_bit_identical"]
                    assert row["phantoms"] == (89 if kernel == "hessian" else 8)
                    assert row["mc_draws"] == 4096 and row["mc_batch_size"] == 64
                    assert row["C_min"] == 20
                    assert row["mc_backend"] == "numpy_PCG64_scipy_sparse_production_event_plan"
                    for key in ("source_sha256", "replay_source_sha256", "mc_source_sha256"):
                        for name, digest in row[key].items():
                            source = (folder.parents[1] / name
                                      if name.startswith("src/") else folder / name)
                            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
                    for key, digest_key in (("phantom_archive", "phantom_archive_sha256"),
                                            ("draws_archive", "draws_archive_sha256")):
                        assert hashlib.sha256(Path(row[key]).read_bytes()).hexdigest() == row[
                            digest_key
                        ]
                    with np.load(row["phantom_archive"]) as saved:
                        n = int(saved["num_samples"])
                        valid = saved["valid_phantom"]
                        constraints = saved["log_L_constraints"]
                        assert n == row["samples"]
                        assert saved["log_L_phantom"].shape[1] == row["phantoms"]
                        np.testing.assert_array_equal(valid[:n], np.isfinite(constraints[:n]))
                        assert not valid[n:].any()
                        assert np.all(saved["log_L_phantom"][valid] > constraints[valid, None])
                        assert int((~valid[:n]).sum()) == row["roots"]
                    with np.load(row["draws_archive"]) as saved:
                        for mode in ("classic_mc", "phantom"):
                            draws = saved[f"{mode}_log_Z_samples"]
                            assert draws.shape == (4096,) and np.all(np.isfinite(draws))
                            np.testing.assert_equal(draws.mean(), row[f"{mode}_log_Z"])
                            np.testing.assert_equal(draws.std(), row[f"{mode}_log_Z_uncert"])
                            np.testing.assert_equal(draws.std(ddof=1) / 64, row[f"{mode}_mcse"])
                    cell.append(row)
            for budget in (2000000, 4000000):
                selected = sorted((row for row in cell if row["budget"] == budget),
                                  key=lambda row: row["seed"])
                assert [row["seed"] for row in selected] == list(range(30))
                errors = {}
                for mode in ("classic", "classic_mc", "phantom"):
                    prefix = "" if mode == "classic" else f"{mode}_"
                    values = np.array([row[f"{prefix}log_Z"] for row in selected])
                    sigma = np.array([row[f"{prefix}log_Z_uncert"] for row in selected])
                    errors[mode] = values
                    summaries.append({
                        "kernel": kernel, "allocation": allocation, "budget": budget,
                        "estimator": mode, "n": 30,
                        "rmse": float(np.sqrt(np.mean(values ** 2))),
                        "bias": float(values.mean()), "mean_quoted_sd": float(sigma.mean()),
                        "within_one_sd": int(np.sum(abs(values) <= sigma)),
                        "within_two_sd": int(np.sum(abs(values) <= 2 * sigma)),
                        "max_mcse": (
                            0.0 if mode == "classic" else
                            max(row[f"{prefix}mcse"] for row in selected)
                        ),
                        "mean_model_calls": float(np.mean([
                            row["evaluations"] for row in selected
                        ])),
                    })
                for reference in ("classic", "classic_mc"):
                    difference = (
                        np.sqrt(np.mean(errors["phantom"][bootstrap] ** 2, axis=1))
                        - np.sqrt(np.mean(errors[reference][bootstrap] ** 2, axis=1))
                    )
                    intervals.append({
                        "kernel": kernel, "allocation": allocation, "budget": budget,
                        "difference": f"phantom_minus_{reference}",
                        "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
                    })
            rows.extend(cell)
    assert len(rows) == 360
    # Keep the original paired-seed comparison between allocation strategies,
    # now using phantom evidence for every arm.
    for kernel in ("hessian", "reflection"):
        for budget in (2000000, 4000000):
            errors = {}
            for allocation in allocations:
                selected = sorted(
                    (row for row in rows if row["kernel"] == kernel
                     and row["budget"] == budget and row["allocation"] == allocation),
                    key=lambda row: row["seed"],
                )
                errors[allocation] = np.array([row["phantom_log_Z"] for row in selected])
            for allocation in allocations[1:]:
                difference = (
                    np.sqrt(np.mean(errors[allocation][bootstrap] ** 2, axis=1))
                    - np.sqrt(np.mean(errors["uniform"][bootstrap] ** 2, axis=1))
                )
                intervals.append({
                    "kernel": kernel, "budget": budget,
                    "difference": f"phantom_{allocation}_minus_uniform",
                    "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
                })
    summary = {
        "summaries": summaries, "paired_bootstrap": intervals,
        "maximum_mcse": max(row[f"{mode}_mcse"] for row in rows
                            for mode in ("classic_mc", "phantom")),
        "minimum_gate_evidence_fraction": min(
            row["phantom_gate_evidence_fraction"] for row in rows
        ),
        "summed_conditioning_wall_seconds_including_compilation": sum(
            row["classic_mc_seconds"] + row["phantom_seconds"] for row in rows
        ),
    }
    (folder / "allocation_phantom_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (folder / "allocation_phantom_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ("#667085", "#4b73bc", "#cf6c35")
    for i, kernel in enumerate(("hessian", "reflection")):
        for j, budget in enumerate((2000000, 4000000)):
            ax = axes[i, j]
            for m, mode in enumerate(("classic", "classic_mc", "phantom")):
                group = [row for row in summaries if row["kernel"] == kernel
                         and row["budget"] == budget and row["estimator"] == mode]
                xpos = np.arange(3) + (m - 1) * 0.24
                ax.bar(xpos, [row["rmse"] for row in group], width=0.23,
                       color=colors[m], label=mode.replace("_", " "))
                ax.scatter(xpos, [row["mean_quoted_sd"] for row in group],
                           color="black", marker="_", s=80, zorder=3)
            ax.set_xticks(np.arange(3), ["uniform", "evidence", "posterior"])
            ax.set_title(f"{kernel.capitalize()}, {budget / 1e6:.0f}M model calls")
            ax.set_ylabel("log Z RMSE (nats)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Identical classic runs; bars: RMSE, black ticks: mean quoted SD; 30 seeds/cell")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"allocation_phantom_comparison.{suffix}", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
