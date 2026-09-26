"""Plot the historical SS10 mode-mass comparison retained by the author.

The accompanying CSV transcribes the rounded cohort statistics already in the
manuscript before this revision. It contains no new experimental measurements.
The source manuscript snapshot is /tmp/jaxns-paper-before-results-20260922.tex;
results-mode-recovery.tex retains the settings and budget-matching limitations.
Missing bootstrap SEs are left absent, never interpreted as zero uncertainty.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    """Keep the continuation trajectory separate from independent cohorts."""
    root = Path(__file__).resolve().parent
    with (root / "ss10_mode_recovery_effort.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ladder = [row for row in rows if row["cohort"] == "historical_phantom"]
    if len(ladder) != 3 or len(rows) != 6:
        raise ValueError("Expected the three-stage ladder and three comparators.")

    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axis = plt.subplots(figsize=(7.4, 4.1), layout="constrained")
    calls_million = [float(row["mean_likelihood_calls"]) / 1e6 for row in ladder]
    rmse = [float(row["mass_rmse"]) for row in ladder]
    axis.errorbar(
        calls_million, rmse,
        yerr=[float(row["mass_rmse_bootstrap_se"]) for row in ladder],
        fmt="o-", capsize=3, color="#0072B2",
        label="JAXNS: historical phantom-seed ladder",
    )
    for calls, error, row in zip(calls_million, rmse, ladder):
        axis.annotate(
            row["uncertainty_target"], (calls, error),
            xytext=(7, 13), textcoords="offset points", color="#0072B2",
        )

    styles = {
        "classic": ("s", "#009E73", "JAXNS: classic seeds, target 0.05"),
        "dynesty": ("^", "#D55E00", "dynesty: matched baseline work"),
        "polychord": ("D", "#CC79A7", "PolyChord: approximate baseline match"),
    }
    for row in rows:
        if row["cohort"] == "historical_phantom":
            continue
        marker, color, label = styles[row["cohort"]]
        axis.plot(
            float(row["mean_likelihood_calls"]) / 1e6,
            float(row["mass_rmse"]), marker=marker, color=color,
            linestyle="none", markersize=7, label=label,
        )
    axis.axhline(.05, color="0.45", linestyle=":", label="Recovery RMSE threshold")
    axis.set(
        xscale="log", xlim=(23, 470), ylim=(0, .54),
        xlabel="Cumulative mean likelihood calls (millions)",
        ylabel="Spike-mass RMSE",
        title="SS10 mode recovery: 30 runs per cohort",
    )
    axis.set_xticks([30, 50, 100, 200, 400], ["30", "50", "100", "200", "400"])
    axis.minorticks_off()
    axis.grid(alpha=.15)
    axis.legend(loc="lower left", bbox_to_anchor=(0, .12), frameon=False, fontsize=9)
    output = root.parents[1] / "images" / "ss10_mode_recovery_effort"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
