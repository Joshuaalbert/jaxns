"""Plot measured SS10 cost scaling and shared accuracy after paired goals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
record = json.loads(args.input.read_text())
assert record["complete"] and record["summary"]["completed_pairs"] == 30
pairs = record["pairs"]
n = np.asarray([row["input_classics"] for row in pairs]) / 1e6
seed = np.asarray([row["seed"] for row in pairs])
old_time = np.asarray([row["old_seconds"] for row in pairs])
new_time = np.asarray([row["new_seconds"] for row in pairs])
old_memory = np.asarray([row["old_warm_rss_bytes"] for row in pairs]) / 2**30
new_memory = np.asarray([row["new_warm_rss_bytes"] for row in pairs]) / 2**30
error = np.asarray([row["log_Z_error"] for row in pairs])
sigma = np.asarray([row["log_Z_uncert"] for row in pairs])
spike = np.asarray([row["spike_mass"] for row in pairs])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for axis, old, new, ylabel in (
    (axes[0, 0], old_time, new_time, "Warm goal time (seconds)"),
    (axes[0, 1], old_memory, new_memory, "Warm peak RSS (GiB)"),
):
    for x, a, b in zip(n, old, new, strict=True):
        axis.plot([x, x], [a, b], color="0.8", linewidth=.8, zorder=1)
    axis.scatter(n, old, label="Old A+C", color="#2864a0", marker="s", s=25, zorder=2)
    axis.scatter(n, new, label="Efficient A+C", color="#dd7527", s=25, zorder=3)
    axis.set(xlabel="Input classic samples (millions)", ylabel=ylabel)
    axis.legend(frameon=False)
axes[1, 0].errorbar(seed, error, yerr=1.96 * sigma, fmt="o", markersize=4,
                    color="#2864a0", capsize=2, label="Both implementations")
axes[1, 0].axhline(0., color="black", linestyle="--", linewidth=1, label="Truth")
axes[1, 0].set(xlabel="Seed", ylabel="Classic log Z error (nominal 95% intervals)")
axes[1, 0].legend(frameon=False)
axes[1, 1].scatter(seed, spike, color="#2864a0", s=25, label="Both implementations")
axes[1, 1].axhline(.5000077444258325, color="black", linestyle="--", linewidth=1,
                   label="True spike mass")
axes[1, 1].set(xlabel="Seed", ylabel="Classic posterior spike mass", ylim=(-.03, 1.03))
axes[1, 1].legend(frameon=False)
for axis in axes.flat:
    axis.grid(alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("SS10: one additional goal iteration from each completed 0.02 tree\n"
             "30 paired seeds; 3 warm repetitions; one pinned CPU per pair", fontsize=12)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output.with_suffix(".png"), dpi=180)
fig.savefig(args.output.with_suffix(".pdf"))
plt.close(fig)
