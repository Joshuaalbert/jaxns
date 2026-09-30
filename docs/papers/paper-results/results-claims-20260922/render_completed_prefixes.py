"""Verify downloaded shrinkage draws and render complete baseline prefix sweeps.

Sampling and shrinkage reductions run on dorrie. This script checks the compact
records against their published summary and renders the local paper artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Require every seed and retained prefix before publishing a cohort."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=root / "completed-02")
    args = parser.parse_args()
    summary = json.loads((args.records / "SUMMARY.json").read_text())
    protocol_hash = hashlib.sha256(
        (root / "PROTOCOL-full-retention.json").read_bytes(),
    ).hexdigest()
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("Downloaded summary does not match the full protocol.")
    if (summary["bootstrap_resamples"], summary["bootstrap_seed"]) != (
        100_000, 20260911,
    ):
        raise ValueError("The paired bootstrap protocol changed.")

    # Reuse the same whole-seed resamples for every prefix. Independent
    # resampling would discard the pairing and misstate RMSE-difference errors.
    indices = np.random.default_rng(20260911).integers(0, 30, (100_000, 30))
    verification = {}
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    for arm, case, table_label in (
        ("g10_e", "G10", "tab:phantom_evidence_g10"),
        ("ss10_e", "SS10", "tab:phantom_evidence_ss10_0p05"),
    ):
        metrics = summary["arms"][arm]
        prefix_sizes = list(range(0, 100, 10)) + [99]
        if not metrics["complete"] or metrics["seeds"] != list(range(30)):
            raise ValueError(f"{arm} is not a complete 30-seed cohort.")
        if metrics["prefix_sizes"] != prefix_sizes:
            raise ValueError(f"{arm} is missing the full retained prefix.")
        errors = []
        reported_sd = []
        coverage = []
        draw_hashes = []
        for seed in range(30):
            seed_root = args.records / "arms" / arm / f"seed-{seed:02d}"
            analysis = json.loads((seed_root / "ANALYSIS.json").read_text())
            if not (
                analysis["complete"]
                and analysis["full_retention"]
                and analysis["mc_draws"] == 2048
                and analysis["prefix_sizes"] == prefix_sizes
                and analysis["seed"] == seed
                and analysis["protocol_sha256"] == protocol_hash
                and analysis["prefixes_share_exact_classic_tree_and_posterior"]
            ):
                raise ValueError(f"Incomplete or incompatible record: {seed_root}")
            draw_path = seed_root / "evidence_draws.npz"
            draw_hash = hashlib.sha256(draw_path.read_bytes()).hexdigest()
            if draw_hash != analysis["evidence_draws_sha256"]:
                raise ValueError(f"Draw checksum mismatch: {draw_path}")
            draw_hashes.append(draw_hash)
            with np.load(draw_path) as archive:
                draws = archive["log_Z"]
            if draws.shape != (2048, 11) or not np.all(np.isfinite(draws)):
                raise ValueError(f"Invalid shrinkage draws: {draw_path}")
            mean = draws.mean(axis=0)
            sd = draws.std(axis=0, ddof=1)
            interval = np.quantile(draws, [.025, .975], axis=0)
            reference = metrics["reference"]["log_Z"]
            np.testing.assert_allclose(mean, analysis["log_Z_mean"], atol=1e-12)
            np.testing.assert_allclose(sd, analysis["log_Z_uncert"], atol=1e-12)
            np.testing.assert_allclose(interval, analysis["interval95"], atol=1e-12)
            errors.append(mean - reference)
            reported_sd.append(sd)
            coverage.append((interval[0] <= reference) & (reference <= interval[1]))

        errors = np.asarray(errors)
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        # Process one prefix at a time to avoid materialising a 100000x30x11
        # array solely for verifying the remote bootstrap summary.
        boot_rmse = np.column_stack([
            np.sqrt(np.mean(errors[:, k][indices] ** 2, axis=1))
            for k in range(11)
        ])
        delta_interval = np.quantile(
            boot_rmse - boot_rmse[:, :1], [.025, .975], axis=0,
        ).T
        checks = {
            "per_seed_errors": errors,
            "rmse": rmse,
            "rmse_bootstrap_se": boot_rmse.std(axis=0, ddof=1),
            "bias": errors.mean(axis=0),
            "error_sd": errors.std(axis=0, ddof=1),
            "mean_reported_sd": np.mean(reported_sd, axis=0),
            "coverage95": np.mean(coverage, axis=0),
            "paired_rmse_minus_classic_ci95": delta_interval,
        }
        for key, actual in checks.items():
            np.testing.assert_allclose(actual, metrics[key], atol=1e-12)
        verification[arm] = {
            "seeds_verified": 30,
            "draws_per_seed": 2048,
            "prefix_sizes": prefix_sizes,
            "evidence_draws_sha256": draw_hashes,
            "full_minus_classic_rmse": float(rmse[-1] - rmse[0]),
            "full_minus_classic_ci95": delta_interval[-1].tolist(),
            "full_minus_9D_ci95": np.quantile(
                boot_rmse[:, -1] - boot_rmse[:, -2], [.025, .975],
            ).tolist(),
        }

        mean_caption = ""
        mean_title = ""
        if case == "G10":
            mean_caption = r"$\mu_{\rm G}=3e_1-3e_2$, "
            mean_title = ", mean (3, -3)"
        rows = [
            r"\begin{table*}[t]", r"\centering", r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            (
                rf"\caption{{{case}, {mean_caption}classic seeds and "
                r"evidence-improving allocation, $30$ trees and $2048$ paired "
                r"shrinkage draws per tree. All $99=sD-1$ phantoms are included "
                r"at the final endpoint. RMSE SE and pointwise $95\%$ intervals "
                r"for RMSE minus classic RMSE use $100{,}000$ paired whole-seed "
                r"bootstrap resamples. Reported SD is the mean within-tree "
                r"shrinkage uncertainty. Mean likelihood "
                rf"calls are $({metrics['mean_likelihood_evaluations'] / 1e6:.3f}"
                rf"\pm{metrics['sd_likelihood_evaluations'] / 1e6:.3f})\times10^6$ "
                r"(mean $\pm$ SD). Costs and classic posteriors are fixed across "
                r"prefixes.}"
            ),
            rf"\label{{{table_label}}}", r"\begin{tabular}{lrrrl}",
            r"\toprule",
            (
                r"Prefix & RMSE $\pm$ SE & Reported SD "
                r"& 95\% coverage & $\Delta$RMSE 95\% interval \\"
            ),
            r"\midrule",
        ]
        for k in range(11):
            label = rf"${k}D$"
            if k == 0:
                label = "Classic"
            elif k == 10:
                label = r"All ($sD-1$)"
            ci = delta_interval[k]
            difference = "---" if k == 0 else rf"$[{ci[0]:+.5f},{ci[1]:+.5f}]$"
            rows.append(
                rf"{label} & {rmse[k]:.5f} $\pm$ "
                f"{metrics['rmse_bootstrap_se'][k]:.5f} "
                f"& {metrics['mean_reported_sd'][k]:.5f} "
                rf"& {100 * metrics['coverage95'][k]:.1f}\% & {difference} \\",
            )
        rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        (root / f"{case}_BASELINE_TABLE.tex").write_text("\n".join(rows) + "\n")

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
        prefix = np.asarray(prefix_sizes) / 10
        axes[0].errorbar(
            prefix, rmse, yerr=metrics["rmse_bootstrap_se"], fmt="o-",
            capsize=3, color="#0072B2", label="Empirical RMSE",
        )
        axes[0].plot(
            prefix, metrics["mean_reported_sd"], "s--", color="#D55E00",
            label="Mean reported SD",
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
            axis.set(
                xlabel="Phantom prefix / D (all = 99 states)", xticks=prefix,
                xticklabels=[str(k) for k in range(10)] + ["All"],
            )
            axis.grid(alpha=.15)
        fig.suptitle(f"{case}{mean_title}: classic seeds, evidence allocation, 30 runs")
        output = root.parents[1] / "images" / f"{case.lower()}_prefix_accuracy_classic"
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(case, json.dumps(verification[arm], indent=2))
    (args.records / "WORKSTATION_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
