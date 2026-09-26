"""Verify dorrie's complete 10D records and render the paper's final sweeps.

This reads completed shrinkage draws; it never samples trees or reruns the
shrinkage calculation. Whole-seed bootstrap checks independently reproduce
the published remote summary before rendering any manuscript artifact.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Require all 30 seeds and all 99 phantoms for each completed comparison."""
    root = Path(__file__).resolve().parent
    records = root / "completed-07"
    summary = json.loads((records / "SUMMARY.json").read_text())
    protocol_hash = hashlib.sha256((root / "PROTOCOL.json").read_bytes()).hexdigest()
    assert summary["protocol_sha256"] == protocol_hash
    assert summary["bootstrap_resamples"] == 100_000
    assert summary["bootstrap_seed"] == 20260911
    indices = np.random.default_rng(20260911).integers(0, 30, (100_000, 30))
    bootstraps = {}
    cores = {}
    verification = {}
    for arm in summary["completed_arms"]:
        metrics = summary["arms"][arm]
        assert metrics["complete"] and metrics["seeds"] == list(range(30))
        assert metrics["prefix_sizes"] == list(range(0, 100, 10)) + [99]
        errors, sds, coverage, rows = [], [], [], []
        for seed in range(30):
            cell = records / "arms" / arm / f"seed-{seed:02d}"
            analysis_bytes = (cell / "ANALYSIS.json").read_bytes()
            analysis = json.loads(analysis_bytes)
            core = json.loads((cell / "CORE.json").read_text())
            assert analysis["complete"] and core["complete"]
            assert analysis["full_retention"] and analysis["mc_draws"] == 2048
            assert analysis["prefix_sizes"] == metrics["prefix_sizes"]
            assert analysis["seed"] == core["seed"] == seed
            assert analysis["state_sha256"] == core["state_sha256"]
            assert analysis["prefixes_share_exact_classic_tree_and_posterior"]
            if arm.startswith("ss10"):
                # SS10 was unchanged by the Gaussian mean correction. Require
                # its explicit reuse audit rather than accepting old protocols.
                reused = json.loads((cell / "REUSED.json").read_text())
                assert reused["compatible_full_reduction"]
                assert reused["scientific_settings_identical"]
                assert reused["model_and_reducer_source_identical"]
                assert hashlib.sha256(analysis_bytes).hexdigest() == reused[
                    "original_analysis_sha256"
                ]
            else:
                assert analysis["protocol_sha256"] == protocol_hash
                assert analysis["reference"]["mean"] == [2., -3.] + [0.] * 8
            draw_path = cell / "evidence_draws.npz"
            assert hashlib.sha256(draw_path.read_bytes()).hexdigest() == analysis[
                "evidence_draws_sha256"
            ]
            with np.load(draw_path) as archive:
                draws = archive["log_Z"]
            assert draws.shape == (2048, 11) and np.isfinite(draws).all()
            reference = metrics["reference"]["log_Z"]
            assert abs(analysis["reference"]["log_Z"] - reference) < 1e-12
            errors.append(draws.mean(axis=0) - reference)
            sds.append(draws.std(axis=0, ddof=1))
            low, high = np.quantile(draws, [.025, .975], axis=0)
            coverage.append((low <= reference) & (reference <= high))
            rows.append(core)
        errors = np.asarray(errors)
        np.testing.assert_allclose(errors, metrics["per_seed_errors"], atol=1e-12)
        np.testing.assert_allclose(sds, metrics["per_seed_reported_sd"], atol=1e-12)
        np.testing.assert_allclose(np.mean(coverage, axis=0), metrics["coverage95"])
        np.testing.assert_allclose(
            np.sqrt(np.mean(errors**2, axis=0)), metrics["rmse"],
        )
        # One prefix at a time avoids a large temporary resample tensor.
        boot = np.column_stack([
            np.sqrt(np.mean(errors[:, k][indices] ** 2, axis=1)) for k in range(11)
        ])
        np.testing.assert_allclose(boot.std(axis=0, ddof=1), metrics["rmse_bootstrap_se"])
        np.testing.assert_allclose(
            np.quantile(boot - boot[:, :1], [.025, .975], axis=0).T,
            metrics["paired_rmse_minus_classic_ci95"],
        )
        calls = np.array([r["likelihood_evaluations"] for r in rows])
        np.testing.assert_allclose(
            [calls.mean(), calls.std(ddof=1)],
            [metrics["mean_likelihood_evaluations"], metrics["sd_likelihood_evaluations"]],
        )
        bootstraps[arm], cores[arm] = boot, rows
        verification[arm] = {
            "seeds_verified": 30,
            "full_minus_9D_ci95": np.quantile(
                boot[:, -1] - boot[:, -2], [.025, .975],
            ).tolist(),
        }
    for contrast in summary["contrasts"].values():
        left, right = contrast["left"], contrast["right"]
        np.testing.assert_allclose(
            np.quantile(bootstraps[left] - bootstraps[right], [.025, .975], axis=0).T,
            contrast["paired_rmse_difference_ci95"],
        )
    evidence_calls = np.asarray(summary["arms"]["g10_e"]["calls"])
    uniform_calls = np.asarray(summary["arms"]["g10_uniform"]["calls"])
    cost_ratios = evidence_calls[indices].mean(axis=1) / uniform_calls[indices].mean(axis=1)
    np.testing.assert_allclose(
        np.quantile(cost_ratios, [.025, .975]),
        summary["contrasts"]["allocation"]["paired_cost_ratio_of_means_ci95"],
    )
    for arm in ("g10_e", "g10_uniform"):
        assert np.argmin(summary["arms"][arm]["rmse"]) == 10
    before, after = cores["cg10_e"], cores["cg10_p_followup"]
    for parent, child in zip(before, after, strict=True):
        assert child["followup"]["state_sha256"] == parent["state_sha256"]
        assert child["followup"]["initial_ess"] == parent["classic_kish_ess"]
        assert child["followup"]["target_ess"] == 2 * parent["classic_kish_ess"]
        assert child["classic_kish_ess"] >= child["followup"]["target_ess"]
    ess_before = np.array([r["classic_kish_ess"] for r in before])
    ess_after = np.array([r["classic_kish_ess"] for r in after])
    extra_calls = np.array([
        b["likelihood_evaluations"] - a["likelihood_evaluations"]
        for a, b in zip(before, after, strict=True)
    ])
    verification["posterior_continuation"] = {
        "initial_ess_mean_sd": [ess_before.mean(), ess_before.std(ddof=1)],
        "final_ess_mean_sd": [ess_after.mean(), ess_after.std(ddof=1)],
        "ess_ratio_mean_sd": [(ess_after / ess_before).mean(),
                              (ess_after / ess_before).std(ddof=1)],
        "extra_calls_mean_sd": [extra_calls.mean(), extra_calls.std(ddof=1)],
        "extra_fraction_of_mean_baseline_cost": extra_calls.mean()
        / summary["arms"]["cg10_e"]["mean_likelihood_evaluations"],
    }
    (records / "WORKSTATION_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n",
    )

    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    for case, baseline, comparison, comparison_label in (
        ("g10", "g10_e", "g10_uniform", "Uniform allocation"),
        ("cg10", "cg10_e", "cg10_phantom_e", "Classic + phantom seeds"),
    ):
        metrics = summary["arms"][baseline]
        rows = [
            r"\begin{table*}[t]", r"\centering", r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            (
                rf"\caption{{{case.upper()}, $\mu_{{\rm G}}=2e_1-3e_2$, classic seeds "
                r"and evidence-improving allocation: $30$ trees, $2048$ paired "
                r"shrinkage draws per tree, including all $99=sD-1$ phantoms. "
                r"RMSE SE and pointwise $95\%$ intervals for RMSE minus classic "
                r"RMSE use $100{,}000$ paired whole-seed bootstrap resamples. "
                r"Reported SD is mean shrinkage uncertainty. Likelihood calls "
                rf"are $({metrics['mean_likelihood_evaluations']/1e6:.3f}"
                rf"\pm{metrics['sd_likelihood_evaluations']/1e6:.3f})\times10^6$ "
                r"(mean $\pm$ SD), identical for every prefix.}"
            ),
            rf"\label{{tab:phantom_evidence_{case}_0p05}}",
            r"\begin{tabular}{lrrrl}", r"\toprule",
            (
                r"Prefix & RMSE $\pm$ SE & Reported SD & 95\% coverage "
                r"& $\Delta$RMSE 95\% interval \\"
            ),
            r"\midrule",
        ]
        for k in range(11):
            label = "Classic" if k == 0 else rf"${k}D$"
            if k == 10:
                label = r"All ($sD-1$)"
            lo, hi = metrics["paired_rmse_minus_classic_ci95"][k]
            ci = "---" if k == 0 else f"$[{lo:+.5f},{hi:+.5f}]$"
            rows.append(
                rf"{label} & {metrics['rmse'][k]:.5f} $\pm$ "
                rf"{metrics['rmse_bootstrap_se'][k]:.5f} "
                rf"& {metrics['mean_reported_sd'][k]:.5f} "
                rf"& {100*metrics['coverage95'][k]:.1f}\% & {ci} \\"
            )
        rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        (root / f"{case.upper()}_FULL_TABLE.tex").write_text("\n".join(rows) + "\n")
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
        baseline_label = "Evidence allocation" if case == "g10" else "Classic seeds"
        for arm, label, color, marker in (
            (baseline, baseline_label, "#0072B2", "o"),
            (comparison, comparison_label, "#CC79A7", "^"),
        ):
            m = summary["arms"][arm]
            x = np.asarray(m["prefix_sizes"]) / 10
            axes[0].errorbar(
                x, m["rmse"], yerr=m["rmse_bootstrap_se"], marker=marker,
                color=color, capsize=3, label=f"{label}: RMSE",
            )
            axes[0].plot(
                x, m["mean_reported_sd"], "--", marker=marker,
                color=color, label=f"{label}: SD",
            )
            axes[1].plot(
                x, 100 * np.asarray(m["coverage95"]), marker=marker,
                color=color, label=label,
            )
        axes[0].axhline(.05, color="0.4", linestyle=":", label="Stopping target")
        axes[0].set(ylabel="Log-evidence error / uncertainty", ylim=(0, None))
        axes[1].axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
        axes[1].set(ylabel="Central 95% interval coverage (%)", ylim=(-3, 103))
        for axis in axes:
            axis.set(xlabel="Phantom prefix / D (all = 99 states)", xticks=x,
                     xticklabels=[str(k) for k in range(10)] + ["All"])
            axis.grid(alpha=.15)
        # Put dense legends outside the data area to preserve the small RMSE
        # differences the paired experiment is designed to measure.
        axes[0].legend(frameon=False, fontsize=7, loc="upper center",
                       bbox_to_anchor=(.5, -.2), ncol=2)
        axes[1].legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle(f"{case.upper()}, mean (2, -3): 30 runs per policy")
        out = root.parents[1] / "images" / f"{case}_full_prefix_comparison_mean23"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(verification["posterior_continuation"], indent=2))
    print("Verified 210 complete records and all paired evidence contrasts.")


if __name__ == "__main__":
    main()
