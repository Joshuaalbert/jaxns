"""Plot the measured phantom-evidence error versus likelihood work."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("/largedata/albert/jaxns-AC240-20260908/report")
summary = json.loads((out / "summary.json").read_text())
colors = {
    "R240": "#676767",
    "A240": "#0072B2",
    "C240": "#009E73",
    "Cprime240": "#CC79A7",
    "AC240": "#D55E00",
}
labels = {
    "R240": "R",
    "A240": "A",
    "C240": "C",
    "Cprime240": "C′",
    "AC240": "A+C",
}
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
for ax, (case, name) in zip(
    axes.flat,
    [
        ("basic_mvn", "G8"),
        ("weak_curved_mvn8", "CG8"),
        ("spike_slab", "SS8"),
        ("curved_spike_slab8", "CSS8"),
    ],
):
    for variant, cases in summary.items():
        r = cases[case]
        ax.errorbar(
            r["mean_calls"] / 1e6,
            r["rmse"][9],
            yerr=r["rmse_se"][9],
            fmt="o",
            markersize=7,
            color=colors[variant],
            capsize=4,
            label=labels[variant],
        )
    ax.set(
        title=name,
        xlabel="Mean likelihood evaluations (millions)",
        ylabel="Ph72 log-evidence RMSE ± bootstrap SE",
    )
    ax.grid(alpha=0.18)
    ax.margins(x=0.15, y=0.15)
    ax.set_ylim(bottom=0)
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="outside upper center", ncol=5, frameon=False
)
fig.savefig(out / "accuracy_cost.pdf")
fig.savefig(out / "accuracy_cost.png", dpi=160)
print("Saved accuracy_cost.pdf and accuracy_cost.png")
