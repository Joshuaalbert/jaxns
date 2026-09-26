"""Render the archived (2,-3) Gaussian sweeps while full retention is reduced.

The archive's validate_local_summary.py checks all raw draws, stopping records,
and paired bootstrap statistics. This renderer preserves the frozen archive
and produces separate paper artifacts for its existing 0D through 9D sweep.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Render both Gaussian tables and separate accuracy/coverage figures."""
    root = Path(__file__).resolve().parent
    source = root.parent / "classic-seed-mean23-20260920"
    summary = json.loads((source / "SUMMARY.json").read_text())
    protocol_bytes = (source / "PROTOCOL.json").read_bytes()
    if summary["protocol"] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError("Archived Gaussian summary has a different protocol.")
    if json.loads(protocol_bytes)["prefix_sizes"] != list(range(0, 100, 10)):
        raise ValueError("The archived Gaussian sweep must end at 9D.")
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    archived_tables = (source / "PAPER_TABLES.tex").read_text().split(
        r"\begin{table*}[t]",
    )[1:]
    for case, archived_table in zip(("g10", "cg10"), archived_tables[:2], strict=True):
        metrics = summary["problems"][case]["classic_seeds"]
        errors = np.asarray(metrics["per_seed_errors"])
        if errors.shape != (30, 10):
            raise ValueError(f"Incomplete cohort: {case}")
        np.testing.assert_allclose(
            np.sqrt(np.mean(errors**2, axis=0)), metrics["rmse"],
        )
        # Remove the two diagnostic columns from the archived wide table.
        # Their unrounded values remain in SUMMARY.json and the original table.
        rows = [r"\begin{table*}[t]"]
        for line in archived_table.splitlines():
            if "&" in line:
                cells = line.split(" & ")
                line = " & ".join([cells[0]] + cells[3:])
            line = line.replace("{lrrrrrl}", "{lrrrl}")
            line = line.replace(
                f"\\caption{{{case.upper()}, ",
                f"\\caption{{{case.upper()}, " + r"$\mu_{\rm G}=2e_1-3e_2$, ",
            ).replace("Error SD is across-tree SD; ", "")
            line = line.replace(
                "Costs and classic posterior quantities are identical for every prefix.",
                "Costs and classic posterior quantities are identical for every prefix. "
                "The full $99=sD-1$ endpoint is pending and is not shown here.",
            )
            rows.append(line)
            if line == r"\scriptsize":
                rows.append(r"\setlength{\tabcolsep}{4pt}")
        (root / f"{case.upper()}_MEAN23_TABLE.tex").write_text("\n".join(rows))

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
        axes[0].errorbar(
            range(10), metrics["rmse"], yerr=metrics["rmse_bootstrap_se"],
            fmt="o-", capsize=3, color="#0072B2", label="Empirical RMSE",
        )
        axes[0].plot(
            range(10), metrics["mean_reported_sd"], "s--", color="#D55E00",
            label="Mean reported SD",
        )
        axes[0].axhline(.05, color="0.4", linestyle=":", label="Stopping target")
        axes[0].set(ylabel="Log-evidence error / uncertainty", ylim=(0, None))
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].plot(
            range(10), 100 * np.asarray(metrics["coverage95"]), "o-",
            color="#0072B2",
        )
        axes[1].axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
        axes[1].set(ylabel="Central 95% interval coverage (%)", ylim=(-3, 103))
        axes[1].legend(frameon=False, fontsize=9)
        for axis in axes:
            axis.set(xlabel="Phantom prefix / D (full 99 pending)", xticks=range(10))
            axis.grid(alpha=.15)
        fig.suptitle(
            f"{case.upper()}, mean (2, -3): classic seeds, evidence allocation, 30 runs",
        )
        output = root.parents[1] / "images" / f"{case}_prefix_accuracy_mean23"
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
