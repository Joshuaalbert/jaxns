"""Compare the two Gaussian mean choices under evidence-improving allocation.

The means define different models. Paired intervals compare phantom prefixes
with classic shrinkage within each model, where every prefix shares a tree.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Check fixed settings and render all prefixes for both Gaussian cases."""
    root = Path(__file__).resolve().parent
    old = root.parent / "classic-seed-mean23-20260920"
    summaries = []
    protocols = []
    for source in (old, root):
        summary = json.loads((source / "SUMMARY.json").read_text())
        protocol_bytes = (source / "PROTOCOL.json").read_bytes()
        assert summary["protocol"] == hashlib.sha256(
            protocol_bytes,
        ).hexdigest()
        assert summary["bootstrap_resamples"] == 100000
        assert summary["bootstrap_seed"] == 20260911
        summaries.append(summary["problems"])
        protocols.append(json.loads(protocol_bytes))

    # The new mean changes both references. Everything governing sampling,
    # stopping and phantom reduction must match the evidence-improving control.
    for field in (
        "C_min", "cluster_weights", "collect_phantom_samples", "cpu_only",
        "depth_dlogZ", "dimension", "direction", "discovery", "draws",
        "goal", "no_step_out", "num_slices", "phantom_coordinates_in_state",
        "phantom_seeds", "prefix_sizes", "retained_phantoms", "root_degree",
        "sampler_base", "seeds", "shell_size", "shrinkage_classes",
        "shrinkage_key", "uncertainty_targets", "unlimited_samples",
    ):
        assert protocols[0][field] == protocols[1][field], field
    for protocol in protocols:
        assert protocol["continuation"]["allocation_target"] == (
            "evidence_improving"
        )
        assert protocol["continuation"]["delta_K"] == 300
    assert protocols[0]["model"]["shared_mean"] == [2, -3] + [0] * 8
    assert protocols[1]["model"]["shared_mean"] == [2, -2] + [0] * 8
    for field in (
        "covariance_diagonal", "covariance_off_diagonal", "prior", "cg_shear",
    ):
        assert protocols[0]["model"][field] == protocols[1]["model"][field]

    labels = (r"$2e_1-3e_2$", r"$2e_1-2e_2$")
    plain_labels = ("(2,-3,0,...)", "(2,-2,0,...)")
    colors = ("#aa5735", "#176f91")
    table = [
        "# Shared-mean comparison with evidence-improving allocation", "",
        (
            "Each model uses 30 seeds and the same classic uncertainty "
            "threshold of 0.05. The different means define different models "
            "with independently computed reference evidences. This is a "
            "descriptive comparison of model difficulty, not a same-model "
            "sampler or allocation ablation."
        ), "",
        (
            "Every prefix within a seed uses its single completed tree. "
            "Pointwise 95% intervals use 100,000 paired whole-seed bootstrap "
            "resamples and compare prefix RMSE with classic RMSE within "
            "that same cohort. The 90-phantom endpoint is predefined; all "
            "other prefixes are retained without selecting a best prefix."
        ), "",
    ]
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.7))
    prefixes = np.arange(10)
    for row, case in enumerate(("g10", "cg10")):
        for index, (label, plain, color, summary) in enumerate(
            zip(labels, plain_labels, colors, summaries, strict=True)
        ):
            data = summary[case]["classic_seeds"]
            rmse = np.asarray(data["rmse"])
            intervals = np.asarray(data["paired_rmse_minus_classic_ci95"])
            table.extend([
                f"## {case.upper()}, mean {plain}", "",
                (
                    "Likelihood calls: "
                    f"{data['mean_likelihood_evaluations'] / 1e6:.3f} "
                    f"± {data['sd_likelihood_evaluations'] / 1e6:.3f} "
                    "million (mean ± across-seed SD); "
                    f"mean goals: {data['mean_goals']:.2f}."
                ), "",
                (
                    "| Prefix | Bias | Error SD | RMSE ± SE | Reported SD | "
                    "95% coverage | RMSE − classic | Paired 95% interval |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for prefix in range(10):
                low, high = intervals[prefix]
                table.append(
                    f"| {prefix}D | {data['bias'][prefix]:+.5f} | "
                    f"{data['error_sd'][prefix]:.5f} | "
                    f"{rmse[prefix]:.5f} ± "
                    f"{data['rmse_bootstrap_se'][prefix]:.5f} | "
                    f"{data['mean_reported_sd'][prefix]:.5f} | "
                    f"{100 * data['coverage95'][prefix]:.1f}% | "
                    f"{rmse[prefix] - rmse[0]:+.5f} | "
                    f"[{low:+.5f}, {high:+.5f}] |"
                )
            improved = [
                f"{p}D" for p in range(1, 10) if intervals[p, 1] < 0
            ]
            degraded = [
                f"{p}D" for p in range(1, 10) if intervals[p, 0] > 0
            ]
            table.extend([
                "", "Resolved RMSE reductions: "
                + (", ".join(improved) if improved else "none") + ".",
                "Resolved RMSE increases: "
                + (", ".join(degraded) if degraded else "none") + ".", "",
            ])
            axes[row, 0].errorbar(
                prefixes, rmse, yerr=data["rmse_bootstrap_se"], fmt="o-",
                color=color, capsize=3, label=f"{label}: RMSE ± SE",
            )
            axes[row, 0].plot(
                prefixes, data["mean_reported_sd"], "--", color=color,
                label=f"{label}: reported SD",
            )
            # Percentile intervals need not be symmetric about the estimate.
            x = prefixes[1:] + (-0.09 if index == 0 else 0.09)
            axes[row, 1].vlines(
                x, intervals[1:, 0], intervals[1:, 1], color=color,
            )
            axes[row, 1].plot(
                x, rmse[1:] - rmse[0], "o", color=color, label=label,
            )
            axes[row, 2].plot(
                prefixes, 100 * np.asarray(data["coverage95"]), "o-",
                color=color, label=label,
            )
        axes[row, 0].set(
            title=f"{case.upper()}: accuracy and reported uncertainty",
            ylabel="Log-evidence error scale", ylim=(0, None),
        )
        axes[row, 0].legend(frameon=False, fontsize=8)
        axes[row, 1].axhline(0, color="0.4", linestyle=":")
        axes[row, 1].set(
            title=f"{case.upper()}: paired 95% intervals",
            ylabel="RMSE − classic RMSE",
        )
        axes[row, 1].legend(frameon=False, fontsize=9)
        axes[row, 2].axhline(
            95, color="0.4", linestyle=":", label="Nominal 95%",
        )
        axes[row, 2].set(
            title=f"{case.upper()}: interval coverage",
            ylabel="Coverage (%)", ylim=(0, 103),
        )
        axes[row, 2].legend(frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.set_xticks(prefixes, [f"{p}D" for p in prefixes])
        axis.set_xlabel("Retained phantom prefix (0D: classic)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Evidence-improving allocation; 30 trees per model and mean\n"
        "Different models; same classic uncertainty threshold (0.05)"
    )
    figure.tight_layout()
    figure.savefig(root / "LOCAL_MEAN_COMPARISON.pdf")
    figure.savefig(root / "LOCAL_MEAN_COMPARISON.png", dpi=180)
    plt.close(figure)
    (root / "LOCAL_MEAN_COMPARISON.md").write_text("\n".join(table))
    print("Mean change and fixed evidence-improving settings verified.")


if __name__ == "__main__":
    main()
