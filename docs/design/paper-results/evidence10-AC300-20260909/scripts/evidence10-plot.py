"""Plot evidence accuracy, SS10 recovery, and current problem analogues."""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
matplotlib.use("Agg")
out = args.root / "report"
figures = out / "figures"
figures.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"}
)
blue, orange = "#176f91", "#d26928"
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
for ax, case in zip(axes, ("g10", "cg10", "ss10"), strict=True):
    limits = (-4, 8) if case == "ss10" else (-4, 8)
    grid = np.linspace(*limits, 700)
    xx, yy = np.meshgrid(grid, grid)
    x = np.stack([xx, yy], axis=-1)  # [y,x,D=2]
    if case == "ss10":
        log_l = logsumexp(
            np.stack(
                [
                    multivariate_normal.logpdf(x, mean=[6, 6], cov=0.08 * np.eye(2)),
                    multivariate_normal.logpdf(x, mean=[2.5, 2.5], cov=0.8 * np.eye(2)),
                ]
            ),
            axis=0,
        )
        ax.add_patch(
            Rectangle(
                (-4, -4),
                12,
                12,
                fill=False,
                edgecolor="#bf3c32",
                linestyle="--",
                linewidth=2,
                zorder=5,
                clip_on=False,
            )
        )
    else:
        z = x.copy()
        if case == "cg10":
            z[..., 1] -= 0.4 * ((z[..., 0] - 3.0) ** 2 - 1.0)
        log_l = multivariate_normal.logpdf(z, mean=[3, 0], cov=[[1, 0.99], [0.99, 1]])
        ax.add_patch(
            Circle((0, 0), 1, fill=False, color="#bf3c32", linestyle="--", linewidth=1.6, zorder=5)
        )
    relative = log_l - log_l.max()
    artist = ax.contourf(
        xx, yy, relative, levels=np.linspace(-16, 0, 33), cmap="viridis", extend="min"
    )
    ax.contour(
        xx, yy, relative, levels=[-12, -8, -4, -1], colors="white", linewidths=0.5, alpha=0.75
    )
    ax.set(
        title=f"{case.upper()} analogue",
        xlabel="$x_1$",
        ylabel="$x_2$",
        xlim=limits,
        ylim=limits,
        aspect="equal",
    )
fig.colorbar(artist, ax=axes, shrink=0.75, label="Relative log likelihood")
fig.savefig(figures / "evidence_problems.pdf")
fig.savefig(figures / "evidence_problems.png", dpi=180)
plt.close(fig)

if not (out / "SUMMARY.json").exists():
    raise SystemExit(0)
summary = json.loads((out / "SUMMARY.json").read_text())
if all(c in summary.get("0.05", {}) for c in ("g10", "cg10", "ss10")):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for ax, case in zip(axes, ("g10", "cg10", "ss10"), strict=True):
        r = summary["0.05"][case]
        ax.errorbar(
            np.arange(10),
            r["rmse"],
            yerr=r["rmse_se"],
            color=blue,
            marker="o",
            markersize=4,
            capsize=2,
            label="Empirical RMSE ± SE",
        )
        ax.plot(
            np.arange(10),
            r["mean_reported_sd"],
            "--",
            color=orange,
            label="Mean reported shrinkage SD",
        )
        ax.set(
            title=case.upper(),
            xlabel="Phantom prefix / D",
            ylabel="Log-evidence error",
            xticks=[0, 3, 6, 9],
            ylim=(0, None),
        )
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    fig.savefig(figures / "evidence_accuracy.pdf")
    fig.savefig(figures / "evidence_accuracy.png", dpi=180)
    plt.close(fig)
stages = [s for s in ("0.05", "0.02", "0.01", "0.005") if "ss10" in summary.get(s, {})]
if stages:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    mass = np.array([summary[s]["ss10"]["posterior"]["mass_by_seed"] for s in stages])
    offsets = np.linspace(-0.16, 0.16, 30)
    for seed in range(30):
        axes[0].plot(
            np.arange(len(stages)) + offsets[seed],
            mass[:, seed],
            color=blue,
            alpha=0.65,
            linewidth=0.6,
            marker=".",
            markersize=5,
        )
    axes[0].axhline(0.5000077444, color=orange, linestyle="--", label="Reference mass")
    axes[0].set(
        ylabel="Classic spike posterior mass",
        ylim=(-0.025, 1.025),
        xticks=np.arange(len(stages)),
        xticklabels=stages,
        xlabel="Classic uncertainty target",
        title="Matched seeds",
    )
    axes[0].legend(fontsize=8)
    for index, color, label in ((0, blue, "Classic"), (-1, orange, "Ph90")):
        rmse = [summary[s]["ss10"]["rmse"][index] for s in stages]
        se = [summary[s]["ss10"]["rmse_se"][index] for s in stages]
        sd = [summary[s]["ss10"]["mean_reported_sd"][index] for s in stages]
        position = np.arange(len(stages)) + (-0.035 if index == 0 else 0.035)
        axes[1].errorbar(
            position,
            rmse,
            yerr=se,
            color=color,
            marker="o",
            capsize=2,
            label=label + " RMSE",
        )
        axes[1].plot(position, sd, "--", marker="x", color=color, label=label + " mean SD")
    axes[1].set(
        ylabel="Log-evidence error",
        ylim=(0, None),
        xticks=np.arange(len(stages)),
        xticklabels=stages,
        xlabel="Classic uncertainty target",
        title="Accuracy and reported uncertainty",
    )
    axes[1].legend(fontsize=7, ncols=2)
    calls = [summary[s]["ss10"]["mean_calls"] / 1e6 for s in stages]
    axes[2].plot(np.arange(len(stages)), calls, marker="o", color=blue)
    axes[2].set(
        ylabel="Mean cumulative likelihood calls / million",
        xticks=np.arange(len(stages)),
        xticklabels=stages,
        xlabel="Classic uncertainty target",
        title="Sampling effort",
        ylim=(0, None),
    )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_xlim(-0.35, len(stages) - 0.65)
    fig.savefig(figures / "ss10_recovery_ladder.pdf")
    fig.savefig(figures / "ss10_recovery_ladder.png", dpi=180)
    plt.close(fig)
