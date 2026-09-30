"""Check complete SS10 seeding cohorts and overlay their full-prefix sweeps."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Use the unchanged SS10 model independently of Gaussian mean revisions."""
    root = Path(__file__).resolve().parent
    records = root / "completed-03"
    summary = json.loads((records / "SUMMARY.json").read_text())
    indices = np.random.default_rng(20260911).integers(0, 30, (100_000, 30))
    evidence_bootstraps = []
    mass_bootstraps = []
    verification = {}
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
    mass_fig, mass_axis = plt.subplots(figsize=(8, 3.4), layout="constrained")
    for arm, label, color, marker in (
        ("ss10_e", "Classic seeds", "#0072B2", "o"),
        ("ss10_phantom_e", "Classic + phantom seeds", "#CC79A7", "^"),
    ):
        metrics = summary["arms"][arm]
        if not metrics["complete"] or metrics["seeds"] != list(range(30)):
            raise ValueError(f"Incomplete SS10 cohort: {arm}")
        errors = []
        deviations = []
        coverage = []
        masses = []
        for seed in range(30):
            cell = records / "arms" / arm / f"seed-{seed:02d}"
            analysis = json.loads((cell / "ANALYSIS.json").read_text())
            path = cell / "evidence_draws.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != analysis[
                "evidence_draws_sha256"
            ]:
                raise ValueError(f"Evidence draw checksum mismatch: {path}")
            if not (
                analysis["complete"] and analysis["full_retention"]
                and analysis["prefixes_share_exact_classic_tree_and_posterior"]
                and analysis["prefix_sizes"] == list(range(0, 100, 10)) + [99]
                and analysis["mc_draws"] == 2048 and analysis["seed"] == seed
                and analysis["reference"]["log_Z"] == metrics["reference"]["log_Z"]
            ):
                raise ValueError(f"Incompatible analysis: {cell}")
            with np.load(path) as archive:
                draws = archive["log_Z"]
            if draws.shape != (2048, 11) or not np.isfinite(draws).all():
                raise ValueError(f"Incomplete draws: {path}")
            errors.append(draws.mean(axis=0) - analysis["reference"]["log_Z"])
            deviations.append(draws.std(axis=0, ddof=1))
            low, high = np.quantile(draws, [.025, .975], axis=0)
            coverage.append(
                (low <= analysis["reference"]["log_Z"])
                & (analysis["reference"]["log_Z"] <= high),
            )
            masses.append(analysis["classic_spike_mass"])
        errors = np.asarray(errors)
        masses = np.asarray(masses)
        np.testing.assert_allclose(errors, metrics["per_seed_errors"], atol=1e-12)
        np.testing.assert_allclose(deviations, metrics["per_seed_reported_sd"])
        np.testing.assert_allclose(np.mean(coverage, axis=0), metrics["coverage95"])
        np.testing.assert_allclose(masses, metrics["spike"]["masses"])
        boot_rmse = np.column_stack([
            np.sqrt(np.mean(errors[:, k][indices] ** 2, axis=1))
            for k in range(11)
        ])
        np.testing.assert_allclose(
            boot_rmse.std(axis=0, ddof=1), metrics["rmse_bootstrap_se"],
        )
        evidence_bootstraps.append(boot_rmse)
        mass_errors = masses - metrics["spike"]["truth"]
        mass_bootstraps.append(np.sqrt(np.mean(mass_errors[indices] ** 2, axis=1)))
        verification[arm] = {
            "seeds_verified": 30,
            "full_prefix_rmse": metrics["rmse"][-1],
            "spike_rmse": float(np.sqrt(np.mean(mass_errors**2))),
        }
        prefix = np.asarray(metrics["prefix_sizes"]) / 10
        axes[0].errorbar(
            prefix, metrics["rmse"], yerr=metrics["rmse_bootstrap_se"],
            marker=marker, color=color, capsize=3, label=f"{label}: RMSE",
        )
        axes[0].plot(
            prefix, metrics["mean_reported_sd"], "--", marker=marker,
            color=color, label=f"{label}: reported SD",
        )
        axes[1].plot(
            prefix, 100 * np.asarray(metrics["coverage95"]), marker=marker,
            color=color, label=label,
        )
        mass_axis.plot(
            range(30), masses, marker, color=color, label=label, markersize=5,
        )
    # Seed labels pair different trees between policies; within each policy
    # every prefix still uses the exact same classic tree and posterior.
    difference = evidence_bootstraps[1] - evidence_bootstraps[0]
    delta_ci = np.quantile(difference, [.025, .975], axis=0).T
    np.testing.assert_allclose(
        delta_ci,
        summary["contrasts"]["ss10_seeding"]["paired_rmse_difference_ci95"],
    )
    verification["phantom_seeds_minus_classic_seeds"] = {
        "classic_evidence_rmse_ci95": delta_ci[0].tolist(),
        "full_evidence_rmse_ci95": delta_ci[-1].tolist(),
        "spike_rmse_ci95": np.quantile(
            mass_bootstraps[1] - mass_bootstraps[0], [.025, .975],
        ).tolist(),
    }
    axes[0].axhline(.05, color="0.4", linestyle=":", label="Stopping target")
    axes[0].set(yscale="log", ylabel="Log-evidence error / uncertainty")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    axes[1].axhline(95, color="0.4", linestyle=":", label="Nominal 95%")
    axes[1].set(ylabel="Central 95% interval coverage (%)", ylim=(-3, 103))
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    for axis in axes:
        axis.set(
            xlabel="Phantom prefix / D (all = 99 states)", xticks=prefix,
            xticklabels=[str(k) for k in range(10)] + ["All"],
        )
        axis.grid(alpha=.15)
    fig.suptitle("SS10: seed-population comparison, 30 runs per policy")
    mass_axis.axhline(
        .5000077444258325, color="0.4", linestyle=":", label="Reference mass",
    )
    mass_axis.set(xlabel="Seed", ylabel="Classic posterior spike mass", ylim=(-.03, 1.03))
    mass_axis.legend(frameon=False, fontsize=9, loc="upper right")
    mass_axis.grid(alpha=.15)
    for figure, name in (
        (fig, "ss10_prefix_accuracy_seeding"),
        (mass_fig, "ss10_seeding_mass"),
    ):
        output = root.parents[1] / "images" / name
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(figure)
    (records / "SS10_WORKSTATION_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n",
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
