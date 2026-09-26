"""Verify complete Gaussian dimension cohorts and render observed results only.

The compact records contain dorrie's completed shrinkage draws. Each dimension
enters the comparison only after all 30 sampling and reduction jobs finish.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Check all three complete cohorts before rendering the dimension table/plot."""
    root = Path(__file__).resolve().parent
    records = root / "completed-09"
    summary = json.loads((records / "SUMMARY.json").read_text())
    protocol_hash = hashlib.sha256((root / "PROTOCOL.json").read_bytes()).hexdigest()
    assert summary["protocol_sha256"] == protocol_hash
    assert (summary["bootstrap_resamples"], summary["bootstrap_seed"]) == (
        100_000, 20260911,
    )
    indices = np.random.default_rng(20260911).integers(0, 30, (100_000, 30))
    verification = {}
    for dimension in (10, 20, 40):
        arm = f"g{dimension}_e"
        metrics = summary["arms"][arm]
        prefixes = list(range(0, 10 * dimension, dimension)) + [10 * dimension - 1]
        assert metrics["complete"] and metrics["seeds"] == list(range(30))
        assert metrics["prefix_sizes"] == prefixes
        errors, deviations, coverage, calls, sources = [], [], [], [], set()
        for seed in range(30):
            cell = records / "arms" / arm / f"seed-{seed:02d}"
            core = json.loads((cell / "CORE.json").read_text())
            analysis = json.loads((cell / "ANALYSIS.json").read_text())
            assert core["complete"] and analysis["complete"]
            assert analysis["full_retention"] and analysis["mc_draws"] == 2048
            assert core["seed"] == analysis["seed"] == seed
            assert analysis["dimension"] == dimension
            assert analysis["prefix_sizes"] == prefixes
            assert analysis["protocol_sha256"] == protocol_hash
            assert analysis["reference"]["mean"] == [2., -3.] + [0.] * (dimension - 2)
            assert core["state_sha256"] == analysis["state_sha256"]
            assert core["classic_posterior_sha256"] == analysis["classic_posterior_sha256"]
            assert analysis["prefixes_share_exact_classic_tree_and_posterior"]
            assert core["classic_log_Z_uncert"] < .05
            assert core["likelihood_evaluations"] == analysis["likelihood_evaluations"]
            path = cell / "evidence_draws.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == analysis[
                "evidence_draws_sha256"
            ]
            with np.load(path) as archive:
                draws = archive["log_Z"]
            assert draws.shape == (2048, 11) and np.isfinite(draws).all()
            reference = metrics["reference"]["log_Z"]
            assert abs(analysis["reference"]["log_Z"] - reference) < 1e-12
            errors.append(draws.mean(axis=0) - reference)
            deviations.append(draws.std(axis=0, ddof=1))
            lo, hi = np.quantile(draws, [.025, .975], axis=0)
            coverage.append((lo <= reference) & (reference <= hi))
            calls.append(core["likelihood_evaluations"])
            sources.add(analysis["reduction_source_commit"])
        errors = np.asarray(errors)
        np.testing.assert_allclose(errors, metrics["per_seed_errors"], atol=1e-12)
        np.testing.assert_allclose(deviations, metrics["per_seed_reported_sd"])
        np.testing.assert_allclose(np.mean(coverage, axis=0), metrics["coverage95"])
        np.testing.assert_allclose(
            np.sqrt(np.mean(errors**2, axis=0)), metrics["rmse"],
        )
        boot_rmse = np.column_stack([
            np.sqrt(np.mean(errors[:, k][indices] ** 2, axis=1)) for k in range(11)
        ])
        np.testing.assert_allclose(
            boot_rmse.std(axis=0, ddof=1), metrics["rmse_bootstrap_se"],
        )
        delta_ci = np.quantile(boot_rmse - boot_rmse[:, :1], [.025, .975], axis=0).T
        np.testing.assert_allclose(delta_ci, metrics["paired_rmse_minus_classic_ci95"])
        np.testing.assert_allclose(
            [np.mean(calls), np.std(calls, ddof=1)],
            [metrics["mean_likelihood_evaluations"], metrics["sd_likelihood_evaluations"]],
        )
        verification[arm] = {
            "seeds_verified": 30,
            "prefix_sizes": prefixes,
            "reduction_sources": sorted(sources),
            "full_minus_classic_ci95": delta_ci[-1].tolist(),
            "full_minus_9D_ci95": np.quantile(
                boot_rmse[:, -1] - boot_rmse[:, -2], [.025, .975],
            ).tolist(),
        }

    rows = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\caption{Completed Gaussian dimension cohorts with classic seeds "
            r"and evidence-improving allocation, $30$ runs per dimension. "
            r"RMSE uncertainties are bootstrap SEs. Likelihood counts are "
            r"mean $\pm$ across-run SD. Full retention uses $99$ phantoms in "
            r"$10$ dimensions, $199$ in $20$, and $399$ in $40$.}"
        ),
        r"\label{tab:dimension_comparison}", r"\begin{tabular}{rrrr}",
        r"\toprule",
        (
            r"$D$ & Classic RMSE & Full RMSE "
            r"& Calls ($10^6$) \\"
        ),
        r"\midrule",
    ]
    for dimension in (10, 20, 40):
        m = summary["arms"][f"g{dimension}_e"]
        values = [
            rf"${m['rmse'][k]:.4f}\pm{m['rmse_bootstrap_se'][k]:.4f}$"
            for k in (0, 10)
        ]
        rows.append(
            f"{dimension} & " + " & ".join(values)
            + rf" & ${m['mean_likelihood_evaluations']/1e6:.2f}"
            + rf"\pm{m['sd_likelihood_evaluations']/1e6:.2f}$ \\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (root / "DIMENSION_TABLE.tex").write_text("\n".join(rows) + "\n")

    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    dimensions = [10, 20, 40]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
    for k, label, color, marker in (
        (0, "Classic", "#0072B2", "o"),
        (10, "All phantoms", "#009E73", "s"),
    ):
        axes[0].errorbar(
            dimensions,
            [summary["arms"][f"g{d}_e"]["rmse"][k] for d in dimensions],
            yerr=[summary["arms"][f"g{d}_e"]["rmse_bootstrap_se"][k] for d in dimensions],
            label=label, color=color, marker=marker, capsize=3,
        )
    axes[0].axhline(.05, color="0.4", linestyle=":", label="Stopping target")
    axes[0].set(ylabel="Log-evidence RMSE", ylim=(0, None))
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1].errorbar(
        dimensions,
        [summary["arms"][f"g{d}_e"]["mean_likelihood_evaluations"]/1e6 for d in dimensions],
        yerr=[summary["arms"][f"g{d}_e"]["sd_likelihood_evaluations"]/1e6 for d in dimensions],
        color="#0072B2", marker="o", capsize=3,
    )
    axes[1].set(ylabel="Likelihood evaluations (millions)", ylim=(0, None))
    for axis in axes:
        axis.set(xlabel="Dimension D", xticks=dimensions, xlim=(8, 42))
        axis.grid(alpha=.15)
    fig.suptitle("Completed Gaussian cohorts: 30 runs per dimension")
    output = root.parents[1] / "images" / "dimension_comparison_mean23"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    (records / "DIMENSION_WORKSTATION_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n",
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
