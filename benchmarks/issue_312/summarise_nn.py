"""Plot and verify the three incremental NN fit audits against the GP audit."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    gp = {row["update"]: row for row in map(json.loads,
          (folder / "gp_audit/seed-0/fits.jsonl").read_text().splitlines())}
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for name, label, colour in (
        ("stored", "NN: stored only", "#377eb8"),
        ("augmented", "NN: added value probes", "#d46a30"),
        ("relative", "NN: probes, relative loss", "#9b59b6"),
    ):
        rows = [json.loads(line) for line in
                (folder / f"raw/nn-{name}.jsonl").read_text().splitlines()]
        assert [row["update"] for row in rows] == [0, 1, 2, 9, 10, 19, 20]
        for row in rows:
            for source, checksum in row["source_sha256"].items():
                assert hashlib.sha256((folder / source).read_bytes()).hexdigest() == checksum
            assert row["loss"] == min(row["restart_losses"])
            assert row["parameters"] == 1526 and row["hidden_features"] == 32
            assert row["new_true_evaluations"] == (0 if name == "stored" else 1024)
        x = [-gp[row["update"]]["log_X"] for row in rows]
        axes[0].semilogy(x, [row["local"]["stiff_eigenvalue_ratio"] for row in rows],
                         "o-", label=label, color=colour)
        axes[1].plot(x, [np.degrees(np.arccos(np.clip(row["local"]["stiff_alignment"], 0, 1)))
                         for row in rows], "o-", color=colour)
        axes[2].semilogy(x, [row["population_audit_rmse"] for row in rows], "o-", color=colour)
        summary[name] = {
            "fits": len(rows),
            "training_value_queries": sum(row["new_true_evaluations"] for row in rows),
            "converged_selected_fits": sum(row["optimizer_success"] for row in rows),
            "warm_starts_selected": sum(row["selected_start"] == 1 for row in rows),
            "last_fit": rows[-1],
        }
    selected = [gp[index] for index in (0, 1, 2, 9, 10, 19, 20)]
    x = [-row["log_X"] for row in selected]
    axes[0].semilogy(x, [row["stiff_eigenvalue_ratio"] for row in selected],
                     "o--", color="#888888", label="GP: stored only")
    axes[1].plot(x, [np.degrees(np.arccos(np.clip(row["stiff_alignment"], 0, 1)))
                     for row in selected], "o--", color="#888888")
    axes[2].semilogy(x, [row["held_out_rmse"] for row in selected], "o--", color="#888888")
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set(ylabel="Predicted / true stiff Hessian eigenvalue")
    axes[1].set(ylabel="Median stiff-axis error (degrees)", ylim=(-2, 92))
    axes[2].set(ylabel="Held-out population log-likelihood RMSE (nats)")
    for axis in axes:
        axis.set_xlabel("Compression −log X of archived population")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Incremental NN fit audit: off-contour values improve local curvature, not global coverage",
    )
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"nn_fit_quality.{extension}", dpi=160)
    (folder / "nn_fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "last_fit"}
                      for name, row in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
