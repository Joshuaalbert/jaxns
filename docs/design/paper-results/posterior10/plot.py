"""Plot weighted classic posterior marginals against independent references."""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--cases", nargs="+",
                    choices=("eggbox10", "rosenbrock10", "rastrigin10", "ss10"),
                    default=("eggbox10", "rosenbrock10", "rastrigin10"))
args = parser.parse_args()
out = args.root
matplotlib.use("Agg")
figures = out / "figures"
figures.mkdir(exist_ok=True)
references = json.loads((out / "references/REFERENCE.json").read_text())
blue, orange = "#176f91", "#d26928"
summary = json.loads((out / "SUMMARY.json").read_text()) if (out / "SUMMARY.json").exists() else {}
for case, title in (
    ("eggbox10", "Eggbox 10D"),
    ("rosenbrock10", "Rosenbrock 10D"),
    ("rastrigin10", "Rastrigin 10D"),
    ("ss10", "Spike-slab 10D"),
):
    if case not in args.cases:
        continue
    cell = out / case / "seed-00"
    core = json.loads((cell / "CORE.json").read_text())
    data = np.load(cell / "classic_posterior.npz")
    x = data["x"]  # [N,D], classic samples only
    w = np.exp(data["log_dp"] - logsumexp(data["log_dp"]))  # [N]
    if case == "eggbox10":
        reference = np.load(out / "references/eggbox10.npz")
        grid = reference["x"]
        pdf = np.broadcast_to(reference["pdf"], (10, len(grid)))
        ref_means = np.full(10, references["eggbox10_reference"]["mean"])
        ref_std = np.full(10, references["eggbox10_reference"]["standard_deviation"])
        logz = references["eggbox10_reference"]["log_Z"]
        ranges = [(0.0, 10 * np.pi)] * 10
        bins = 180
    elif case == "rosenbrock10":
        reference = np.load(out / "references/rosenbrock10.npz")
        grid, pdf = reference["x"], reference["pdf"]
        ref_means = np.array(references["rosenbrock10"][-1]["mean"])
        ref_std = np.array(references["rosenbrock10"][-1]["standard_deviation"])
        logz = references["rosenbrock10"][-1]["log_Z"]
        ranges = []
        for dim in range(10):
            cdf = np.cumsum(pdf[dim] * reference["weights"])
            low, high = np.interp([0.0001, 0.9999], cdf, grid)
            order = np.argsort(x[:, dim])
            p_low, p_high = np.interp(
                [0.0001, 0.9999], np.cumsum(w[order]), x[order, dim]
            )
            low, high = min(low, p_low), max(high, p_high)
            margin = 0.03 * (high - low)
            ranges.append((max(-5.0, low - margin), min(5.0, high + margin)))
        bins = 85
    elif case == "ss10":
        reference = np.load(out / "references/ss10.npz")
        ss_reference = json.loads((out / "references/SS10_REFERENCE.json").read_text())
        grid, pdf = reference["x"], reference["pdf"]
        ref_means = np.array(ss_reference["posterior_mean"])
        ref_std = np.array(ss_reference["posterior_standard_deviation"])
        logz = ss_reference["log_Z"]
        ranges = [(-1.5, 7.5)] * 2 + [(-3.5, 3.5)] * 8
        bins = 100
    else:
        grid = np.linspace(-5.12, 5.12, 16385)
        marginal = (
            np.exp(-10 - grid**2 + 10 * np.cos(2 * np.pi * grid))
            / core["reference"]["one_dimensional_mass"]
        )
        pdf = np.broadcast_to(marginal, (10, len(grid)))
        ref_means = np.zeros(10)
        ref_std = np.full(10, core["reference"]["standard_deviation"])
        logz = core["reference"]["log_Z"]
        ranges = [(-3.5, 3.5)] * 10
        bins = 140
    fig, axes = plt.subplots(10, 10, figsize=(23, 23), squeeze=False)
    for row in range(10):
        for col in range(10):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if row == col:
                counts, edges = np.histogram(
                    x[:, row], bins=bins, range=ranges[row], weights=w
                )
                density = counts / np.diff(edges)
                ax.stairs(density, edges, color=blue, linewidth=1.1)
                ax.plot(grid, pdf[row], color=orange, linewidth=1.05)
                ax.set_ylim(bottom=0.0)
                ax.set_yticks([])
            else:
                h, ex, ey = np.histogram2d(
                    x[:, col],
                    x[:, row],
                    bins=bins,
                    range=[ranges[col], ranges[row]],
                    weights=w,
                )
                smooth = gaussian_filter(h, sigma=0.6, mode="constant")
                # Enclosed levels of the displayed weighted histogram, with
                # 0.6-bin smoothing only for the contour lines.
                ordered = np.sort(smooth.ravel())[::-1]
                cumulative = np.cumsum(ordered) / np.sum(ordered)
                levels = np.unique(np.interp([0.95, 0.68], cumulative, ordered))
                cx, cy = 0.5 * (ex[:-1] + ex[1:]), 0.5 * (ey[:-1] + ey[1:])
                ax.pcolormesh(
                    ex, ey, h.T, cmap="Blues", rasterized=True, shading="flat"
                )
                if len(levels) > 0 and levels[0] > 0:
                    ax.contour(
                        cx, cy, smooth.T, levels=levels, colors=[blue], linewidths=0.5
                    )
                ax.set_ylim(ranges[row])
                ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.set_xlim(ranges[col])
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.tick_params(labelsize=16, length=3, pad=3)
            if row == 9:
                ax.set_xlabel(f"$x_{{{col + 1}}}$", fontsize=22)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(f"$x_{{{row + 1}}}$", fontsize=22)
            elif row != col:
                ax.set_yticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"{title} | seed 0 | classic Kish ESS {core['classic_kish_ess']:,.0f}",
        fontsize=28,
        x=0.48,
        y=0.974,
    )
    allocation_label = (
        "A+C: evidence-improving allocation after goal 1\n"
        if case == "ss10" else
        "10% evidence + 90% posterior allocation after goal 1\n"
    )
    fig.text(
        0.59,
        0.87,
        "d0 = 300; all phantom states eligible as seeds\n"
        + allocation_label
        + f"Goal iterations: {core['iteration']}; "
        f"{core['likelihood_evaluations'] / 1e6:.3f} million likelihood calls\n"
        "Classic posterior weights only",
        fontsize=20,
        linespacing=1.6,
        va="top",
    )
    if core["iteration"] == 1 and case != "ss10":
        fig.text(
            0.59,
            0.76,
            "ESS target met in the first uniform goal;\nmixed allocation was not reached.",
            fontsize=20,
            color="#a34a22",
            va="top",
        )
    if case == "ss10":
        fig.text(
            0.59, 0.76,
            f"Stopped at classic sigma(log Z) = {core['classic_log_Z_uncert']:.5f}\n"
            "Requested target: sigma(log Z) < 0.05",
            fontsize=20, va="top",
        )
    fig.legend(
        handles=[
            Line2D([], [], color=blue, label="Classic weighted posterior"),
            Line2D([], [], color=orange, label="Deterministic marginal reference"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.945, 0.955),
        frameon=False,
        fontsize=20,
    )
    fig.subplots_adjust(
        left=0.048, right=0.985, bottom=0.04, top=0.95, hspace=0.075, wspace=0.075
    )
    fig.savefig(figures / f"{case}_corner.pdf", dpi=140)
    fig.savefig(figures / f"{case}_corner.png", dpi=100)
    plt.close(fig)
    # A separate compact marginal figure supports inspection at screen size.
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
    for dim, ax in enumerate(axes.flat):
        counts, edges = np.histogram(x[:, dim], bins=bins, range=ranges[dim], weights=w)
        ax.stairs(counts / np.diff(edges), edges, color=blue)
        ax.plot(grid, pdf[dim], color=orange)
        ax.set_xlim(ranges[dim])
        ax.set_xlabel(f"$x_{{{dim + 1}}}$")
        ax.set_ylim(bottom=0)
    fig.suptitle(
        f"{title}: classic weighted marginals (blue), deterministic reference (orange)"
    )
    fig.tight_layout()
    fig.savefig(figures / f"{case}_marginals.png", dpi=150)
    plt.close(fig)
    mean = np.asarray(core["posterior_mean"])
    std = np.asarray(core["posterior_standard_deviation"])
    summary[case] = dict(
        core,
        reference_log_Z=logz,
        log_Z_error=core["classic_log_Z"] - logz,
        reference_posterior_mean=ref_means.tolist(),
        reference_posterior_standard_deviation=ref_std.tolist(),
        rms_standardised_mean_error=float(
            np.sqrt(np.mean(((mean - ref_means) / ref_std) ** 2))
        ),
        rms_relative_sd_error=float(np.sqrt(np.mean((std / ref_std - 1) ** 2))),
        plot_ranges=ranges,
        plot_bins=bins,
    )
    print(
        case,
        "logZ error",
        summary[case]["log_Z_error"],
        "mean RMS/SD",
        summary[case]["rms_standardised_mean_error"],
        flush=True,
    )
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
