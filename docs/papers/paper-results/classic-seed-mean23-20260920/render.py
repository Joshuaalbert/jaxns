"""Render the new Gaussian cohorts with the unchanged archived SS10 cohort.

Run with ``conda run -n jaxns_py python <this file>`` after copying the audited
remote report into this directory. The layout follows the paper's original
classic-seed-evidence-20260912/render.py, preserving its table precision and
plot scales. Each source summary is checked against its own frozen protocol.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Write all prefix tables and the combined evidence-accuracy figure."""
    root = Path(__file__).resolve().parent
    summaries = []
    for source in (root, root / "archived-ss10"):
        summary = json.loads((source / "SUMMARY.json").read_text())
        protocol_bytes = (source / "PROTOCOL.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        assert summary["protocol"] == protocol_hash
        assert summary["bootstrap_resamples"] == 100000
        assert summary["bootstrap_seed"] == 20260911
        assert protocol["prefix_sizes"] == list(range(0, 100, 10))
        summaries.append(summary)

    # The mean change defines new Gaussian experiments. Only SS10 is reused
    # from the old report, so its historical Gaussian data cannot enter here.
    data = {
        "g10": summaries[0]["problems"]["g10"]["classic_seeds"],
        "cg10": summaries[0]["problems"]["cg10"]["classic_seeds"],
        "ss10": summaries[1]["problems"]["ss10"]["classic_seeds"],
    }
    images = root.parents[1] / "images"
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    tables = []
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), sharex=True)
    for column, (case, metrics) in enumerate(data.items()):
        caption = (
            rf"\caption{{{case.upper()}, "
            r"classic-only constrained-chain seeds, $d_0=300$, "
            r"classic expected uncertainty target $0.05$, $30$ trees. "
            r"Error SD is across-tree SD; RMSE SE uses $100{,}000$ paired "
            r"bootstrap resamples. Reported SD is the mean "
            r"shrinkage uncertainty. The final column is the pointwise "
            r"$95\%$ interval for RMSE minus "
            r"classic RMSE. Mean likelihood calls: $("
            f"{metrics['mean_likelihood_evaluations'] / 1e6:.3f}"
            r"\pm"
            f"{metrics['sd_likelihood_evaluations'] / 1e6:.3f}"
            r")\times10^6$ (mean $\pm$ across-tree SD); mean goals: $"
            f"{metrics['mean_goals']:.2f}"
            r"$. Costs and classic posterior quantities are identical for "
            r"every prefix.}"
        )
        tables.extend([
            r"\begin{table*}[t]", r"\centering", r"\scriptsize", caption,
            rf"\label{{tab:phantom_evidence_{case}_0p05}}",
            r"\begin{tabular}{lrrrrrl}", r"\toprule",
            (
                r"Prefix & Bias & Error SD & RMSE $\pm$ SE & Reported SD & "
                r"95\% coverage & $\Delta$RMSE 95\% interval \\"
            ),
            r"\midrule",
        ])
        for prefix in range(10):
            label = "$0D$ (classic)" if prefix == 0 else f"${prefix}D$"
            low, high = metrics["paired_rmse_minus_classic_ci95"][prefix]
            contrast = "---" if prefix == 0 else f"$[{low:+.5f},{high:+.5f}]$"
            tables.append(
                f"{label} & {metrics['bias'][prefix]:+.5f} & "
                f"{metrics['error_sd'][prefix]:.5f} & "
                f"{metrics['rmse'][prefix]:.5f} $\\pm$ "
                f"{metrics['rmse_bootstrap_se'][prefix]:.5f} & "
                f"{metrics['mean_reported_sd'][prefix]:.5f} & "
                f"{100 * metrics['coverage95'][prefix]:.1f}\\% & "
                f"{contrast} " + r"\\"
            )
        tables.extend([
            r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
        ])
        axis = axes[0, column]
        axis.errorbar(
            range(10), metrics["rmse"], yerr=metrics["rmse_bootstrap_se"],
            fmt="o-", color="#176f91", capsize=3, label="Empirical RMSE",
        )
        axis.plot(
            range(10), metrics["mean_reported_sd"], "--", color="#d26928",
            label="Mean reported SD",
        )
        axis.set_title(case.upper())
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        if column == 0:
            axis.set_ylabel("Log-evidence error scale")
            axis.legend(frameon=False, fontsize=9)
        axis = axes[1, column]
        axis.plot(
            range(10), 100 * np.asarray(metrics["coverage95"]), "o-",
            color="#176f91",
        )
        axis.axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
        axis.set_ylim(-3, 103)
        axis.set_xticks(range(10), [f"{prefix}D" for prefix in range(10)])
        axis.set_xlabel("Retained phantom prefix (0D: classic)")
        axis.grid(axis="y", alpha=0.2)
        if column == 0:
            axis.set_ylabel("95% interval coverage (%)")
            axis.legend(frameon=False, fontsize=9, loc="lower left")
    (root / "PAPER_TABLES.tex").write_text("\n".join(tables))
    figure.suptitle(
        "Classic-only seeds; evidence-improving allocation; "
        "30 trees per problem"
    )
    figure.tight_layout()
    figure.savefig(images / "evidence_accuracy.pdf")
    figure.savefig(images / "evidence_accuracy.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
