"""Render the verified SS10 baseline while matched seeding runs are pending."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Keep the evidence-accuracy plot tied to its unchanged source cohort."""
    root = Path(__file__).resolve().parent
    source = root.parent / "classic-seed-mean23-20260920" / "archived-ss10"
    summary = json.loads((source / "SUMMARY.json").read_text())
    protocol_bytes = (source / "PROTOCOL.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    if summary["protocol"] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError("SS10 summary does not match its archived protocol.")
    if protocol["prefix_sizes"] != list(range(0, 100, 10)):
        raise ValueError("The SS10 baseline must contain all ten prefixes.")

    metrics = summary["problems"]["ss10"]["classic_seeds"]
    errors = np.asarray(metrics["per_seed_errors"])
    if errors.shape != (30, 10) or not np.all(np.isfinite(errors)):
        raise ValueError("The SS10 baseline requires thirty complete paired trees.")
    np.testing.assert_allclose(
        np.sqrt(np.mean(errors**2, axis=0)), metrics["rmse"],
    )

    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
    prefix = np.arange(10)
    axes[0].errorbar(
        prefix, metrics["rmse"], yerr=metrics["rmse_bootstrap_se"],
        fmt="o-", capsize=3, color="#0072B2", label="Empirical RMSE",
    )
    axes[0].plot(
        prefix, metrics["mean_reported_sd"], "s--",
        color="#D55E00", label="Mean reported SD",
    )
    axes[0].axhline(.05, color="0.4", linestyle=":", label="Stopping target")
    axes[0].set(yscale="log", ylabel="Log-evidence error / uncertainty")
    axes[0].legend(loc="center right", frameon=False)
    axes[1].plot(
        prefix, 100 * np.asarray(metrics["coverage95"]), "o-", color="#0072B2",
    )
    axes[1].axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
    axes[1].set(ylabel="Central 95% interval coverage (%)", ylim=(-3, 103))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set(xlabel="Phantom prefix / D", xticks=prefix)
        axis.grid(alpha=.15)
    figure.suptitle("SS10: classic seeds, evidence-improving allocation, 30 runs")
    output = root.parents[1] / "images" / "ss10_prefix_accuracy_classic"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
