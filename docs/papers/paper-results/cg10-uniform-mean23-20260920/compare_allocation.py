"""Compare CG10 phantom conditioning under the two allocation policies.

Run after importing the audited uniform cohort. Both cohorts stop at the same
classic uncertainty; their likelihood costs can differ. Intervals compare
prefixes within a cohort, where each seed supplies exactly one shared tree.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Check the allocation contrast and save all-prefix tables and plots."""
    root = Path(__file__).resolve().parent
    old = root.parent / "classic-seed-mean23-20260920"
    summaries = []
    protocols = []
    for source, cohort in ((old, "classic_seeds"), (root, "uniform")):
        summary = json.loads((source / "SUMMARY.json").read_text())
        protocol_bytes = (source / "PROTOCOL.json").read_bytes()
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        assert summary["protocol"] == protocol_hash
        assert summary["bootstrap_resamples"] == 100000
        assert summary["bootstrap_seed"] == 20260911
        summaries.append(summary["problems"]["cg10"][cohort])
        protocols.append(json.loads(protocol_bytes))

    # Only continuation allocation changes scientifically. Verify the model,
    # sampling, stopping and reduction controls before comparing the reports.
    for field in (
        "C_min", "cluster_weights", "collect_phantom_samples", "cpu_only",
        "depth_dlogZ", "dimension", "direction", "draws", "goal",
        "no_step_out", "num_slices", "phantom_coordinates_in_state",
        "phantom_seeds", "prefix_sizes", "retained_phantoms", "root_degree",
        "sampler_base", "seeds", "shell_size", "shrinkage_classes",
        "shrinkage_key", "uncertainty_targets", "unlimited_samples",
    ):
        assert protocols[0][field] == protocols[1][field], field
    for field in (
        "shared_mean", "covariance_diagonal", "covariance_off_diagonal",
        "prior", "cg_shear",
    ):
        assert protocols[0]["model"][field] == protocols[1]["model"][field]
    assert protocols[0]["continuation"]["allocation_target"] == (
        "evidence_improving"
    )
    assert protocols[1]["continuation"]["allocation_target"] == "uniform"
    assert all(p["continuation"]["delta_K"] == 300 for p in protocols)

    for seed in range(30):
        relative = Path("0.05") / "cg10" / f"seed-{seed:02d}" / "CORE.json"
        original = json.loads((old / relative).read_text())["goal_progress"]
        uniform = json.loads((root / relative).read_text())["goal_progress"]
        # Timing and capacity are operational details. The initial random
        # trajectory and scientific summaries must match before policies fork.
        for field in (
            "iteration", "allocation_iteration", "depth_iteration",
            "root_out_degree", "classic_log_Z", "classic_log_Z_uncert",
            "classic_kish_ess", "likelihood_evaluations", "classic_samples",
        ):
            assert original[0][field] == uniform[0][field], (seed, field)
        for goal in uniform:
            assert goal["root_out_degree"] == (
                300 * goal["allocation_iteration"]
            ), (seed, goal["iteration"])

    labels = ("Evidence-improving", "Uniform")
    colors = ("#aa5735", "#176f91")
    table = [
        "# CG10 allocation comparison", "",
        (
            "Both cohorts use 30 seeds, mean `(2,-3,0,...,0)`, and the same "
            "classic uncertainty stopping threshold of 0.05. All prefixes "
            "within a seed use its single completed tree and 2,048 paired "
            "shrinkage draws."
        ),
        "",
        (
            "Intervals below use 100,000 paired whole-seed bootstrap "
            "resamples. They are pointwise intervals for prefix RMSE minus "
            "classic RMSE within the same allocation policy. These are "
            "equal-threshold comparisons; likelihood budgets differ. "
            "No prefix is selected."
        ), "",
    ]
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    prefixes = np.arange(10)
    for index, (label, color, data) in enumerate(
        zip(labels, colors, summaries, strict=True)
    ):
        rmse = np.asarray(data["rmse"])
        intervals = np.asarray(data["paired_rmse_minus_classic_ci95"])
        table.extend([
            f"## {label}", "",
            (
                "Likelihood calls: "
                f"{data['mean_likelihood_evaluations'] / 1e6:.3f} "
                f"± {data['sd_likelihood_evaluations'] / 1e6:.3f} million "
                "(mean ± across-seed SD). "
                f"Mean completed goals: {data['mean_goals']:.2f}."
            ), "",
            (
                "| Prefix | Bias | Error SD | RMSE ± SE | Mean reported SD | "
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
        improved = [f"{p}D" for p in range(1, 10) if intervals[p, 1] < 0]
        degraded = [f"{p}D" for p in range(1, 10) if intervals[p, 0] > 0]
        table.extend([
            "", "Resolved RMSE reductions: "
            + (", ".join(improved) if improved else "none") + ".",
            "Resolved RMSE increases: "
            + (", ".join(degraded) if degraded else "none") + ".", "",
        ])
        axes[0].errorbar(
            prefixes, rmse, yerr=data["rmse_bootstrap_se"], fmt="o-",
            color=color, capsize=3, label=f"{label}: RMSE ± SE",
        )
        axes[0].plot(
            prefixes, data["mean_reported_sd"], "--", color=color,
            label=f"{label}: reported SD",
        )
        # Draw percentile endpoints directly: they need not be symmetric
        # around the observed RMSE difference.
        x = prefixes[1:] + (-0.09 if index == 0 else 0.09)
        axes[1].vlines(x, intervals[1:, 0], intervals[1:, 1], color=color)
        axes[1].plot(x, rmse[1:] - rmse[0], "o", color=color, label=label)
        axes[2].plot(
            prefixes, 100 * np.asarray(data["coverage95"]), "o-",
            color=color, label=label,
        )
    axes[0].set(title="Accuracy and reported uncertainty",
                ylabel="Log-evidence error scale", ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="0.4", linestyle=":")
    axes[1].set(title="Paired 95% intervals", ylabel="RMSE − classic RMSE")
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
    axes[2].set(
        title="Interval coverage", ylabel="Coverage (%)", ylim=(0, 103),
    )
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xticks(prefixes, [f"{p}D" for p in prefixes])
        axis.set_xlabel("Retained phantom prefix (0D: classic)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        r"CG10: $\mu_G=2e_1-3e_2$; 30 trees per allocation policy"
        "\nSame classic uncertainty threshold (0.05); mean likelihood calls: "
        f"{summaries[0]['mean_likelihood_evaluations'] / 1e6:.1f}M "
        "evidence-improving, "
        f"{summaries[1]['mean_likelihood_evaluations'] / 1e6:.1f}M uniform"
    )
    figure.tight_layout()
    figure.savefig(root / "LOCAL_ALLOCATION_COMPARISON.pdf")
    figure.savefig(root / "LOCAL_ALLOCATION_COMPARISON.png", dpi=180)
    plt.close(figure)
    (root / "LOCAL_ALLOCATION_COMPARISON.md").write_text("\n".join(table))
    print("Controls, 30 discovery passes and uniform targets agree.")


if __name__ == "__main__":
    main()
