"""Complete-cohort contrasts and figures; never selects settings from partial results."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def write_report(root: Path, full: dict) -> None:
    """Render the curated report from the already audited numerical summary."""
    import numpy as np

    summary = full["cells"]
    contrasts = full["contrasts"]
    runtimes = full["runtimes"]
    estimators = ("classic_moments", "classic_mc", "phantom")
    report = [
        "# Frozen reflection step tuning: full audited CPU cohort",
        "",
        "All 120 runs and 240 snapshots are included. Tuning to 70% path acceptance "
        "did not improve evidence RMSE at the prespecified 4M primary budget. "
        "Both controls and tuned arms are fresh ARM runs with matched prior populations.",
        "",
        "The contour-dependent schedule was learned offline from independent exact "
        "constrained pilot draws and frozen before evaluation. It depends only on the parent "
        "contour, with common fixed h=0.003 above gap 49; it is not online adaptation. "
        "B=10 reflections and C=9 cycles stayed fixed. Reducing h therefore shortened "
        "the nominal trajectory time B*h rather than compensating with more reflections. "
        "Gaussian velocity refresh, independent step jitter and rejection repeats are unchanged. "
        "No evaluation-driven retuning or trajectory-length matching was performed.",
        "",
        "## Evidence at the primary four-million-call budget",
        "",
        "| Allocation | Arm | Estimator | Bias | RMSE | RMSE bootstrap SE | "
        "Mean SD | 95% coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for allocation in ("uniform", "evidence_improving"):
        for arm in ("fixed", "tuned"):
            info = summary[f"{arm}_{allocation}_4000000"]
            for estimator in estimators:
                item = info["estimators"][estimator]
                report.append(
                    f"| {allocation} | {arm} | {estimator} | {item['bias']:.5f} | "
                    f"{item['rmse']:.5f} | {item['rmse_bootstrap_se']:.5f} | "
                    f"{item['mean_reported_sd']:.5f} | {item['coverage95_count']}/30 |"
                )
    report += [
        "",
        "## Prespecified paired RMSE contrasts",
        "",
        "| Allocation | Estimator | Tuned minus fixed | Paired bootstrap 95% interval |",
        "|---|---|---:|---|",
    ]
    for allocation in ("uniform", "evidence_improving"):
        for estimator in estimators:
            item = contrasts[f"tuned_minus_fixed_{allocation}_4000000_{estimator}"]
            lo, hi = item["paired_bootstrap95"]
            report.append(
                f"| {allocation} | {estimator} | {item['delta_rmse']:+.5f} | "
                f"[{lo:+.5f}, {hi:+.5f}] |"
            )
    report += [
        "", "## Secondary two-million-call snapshot", "",
        "| Allocation | Arm | Classic moments RMSE | Classic MC RMSE | Phantom MC RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for allocation in ("uniform", "evidence_improving"):
        for arm in ("fixed", "tuned"):
            estimates = summary[f"{arm}_{allocation}_2000000"]["estimators"]
            report.append(
                f"| {allocation} | {arm} | {estimates['classic_moments']['rmse']:.5f} | "
                f"{estimates['classic_mc']['rmse']:.5f} | {estimates['phantom']['rmse']:.5f} |"
            )
    report += [
        "", "## Posterior widths and actual movement at four million calls", "",
        "Medians over all 30 runs; movement uses the prespecified 5–15 parent-gap bin. "
        "The true flat-coordinate SD is 0.288675 and x5 SD is 0.05. "
        "Every chain uses 90 gradients; displacement per gradient is RMS/90.", "",
        "| Allocation | Arm | SD x1 | SD x16 | SD x17 | SD x5 | RMS move x1 | "
        "RMS move x16 | Path acceptance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for allocation in ("uniform", "evidence_improving"):
        for arm in ("fixed", "tuned"):
            item = summary[f"{arm}_{allocation}_4000000"]
            sd, move = item["median_marginal_sd"], item["movement"]["1"]
            report.append(
                f"| {allocation} | {arm} | {sd['x1']:.5f} | {sd['x16']:.5f} | "
                f"{sd['x17']:.5f} | {sd['x5']:.5f} | {move['rms_dx1']:.5f} | "
                f"{move['rms_dx16']:.5f} | {move['acceptance']:.3%} |"
            )
    report += [
        "", "## Matched work and execution cost at four million calls", "",
        "| Allocation | Arm | Mean calls | Call SD | Mean gradients | Mean completed epochs | "
        "Median core seconds | Median analysis seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for allocation in ("uniform", "evidence_improving"):
        for arm in ("fixed", "tuned"):
            item = summary[f"{arm}_{allocation}_4000000"]
            times = [r for r in runtimes if r["arm"] == arm and r["allocation"] == allocation]
            report.append(
                f"| {allocation} | {arm} | {item['mean_calls']:,.0f} | {item['sd_calls']:,.0f} | "
                f"{item['mean_gradients']:,.0f} | {item['mean_epochs']:.3f} | "
                f"{np.median([r['core_seconds'] for r in times]):.2f} | "
                f"{np.median([r['analysis_seconds'] for r in times]):.2f} |"
            )
    report += [
        "",
        "## Interpretation and limitations",
        "",
        "The frozen 70% target did not improve the primary 4M result in this cohort. "
        "Classic and phantom RMSE point estimates are higher under both allocations. "
        "Only the evidence-allocation classic contrast excludes zero in its pointwise 95% "
        "paired interval; these intervals are not adjusted for multiple comparisons. "
        "At 2M, tuned uniform has a lower classic point RMSE, so the primary endpoint matters. "
        "The complete two-budget comparison is retained above.", "",
        "The acceptance target was selected before evaluation. Stationarity and higher "
        "acceptance do not establish mixing. Compare absolute displacement, posterior "
        "CDF errors, marginal widths and evidence RMSE together; a collapsed seed cloud "
        "can have deceptively low endpoint correlation. Kish ESS here is weight ESS, "
        "not a correction for Markov-chain dependence.",
        "",
        "In the 5–15 gap bin, tuning reduces median absolute x1 movement by about 35% "
        "and inert-coordinate movement by about 40%, despite raising path acceptance to 70%. "
        "The held-out exact ensembles show the same shorter-movement effect. Posterior widths "
        "generally narrow and marginal CDF errors worsen; higher acceptance has not repaired "
        "population coverage. This supports a movement-versus-acceptance tradeoff for fixed B=10 "
        "and C=9, not a universal optimal acceptance target.", "",
        "At 4M, phantom conditioning increases RMSE relative to classic MC in all four cells, "
        "with positive pointwise paired intervals. Mean reported uncertainties shrink while "
        "coverage remains poor. The full contrasts and per-seed MCSE are in SUMMARY.json; "
        "the largest evidence-mean MCSE over both budgets is 0.00673 (classic) "
        "or 0.00390 (phantom).", "",
        "Classic moments, classic Monte Carlo and phantom Monte Carlo are distinct "
        "estimator conventions. Arms share seeds and initial populations but generate "
        "different trees; classic/phantom contrasts within an arm use the same tree. "
        "Phantoms receive no posterior weights or seed access. "
        "The original eight-cycle prefix and cluster-weight evidence law are unchanged. "
        "See SUMMARY.json for both budgets, all uncertainty diagnostics, per-seed "
        "errors, paired phantom-minus-classic contrasts and MC errors.",
        "",
        "## Work and provenance",
        "",
        "Each sampling process and reduction was pinned to one CPU with CPU-only JAX. "
        "Logical model-call budgets include each gradient evaluation once; gradient "
        "cost is also reported separately. Diagnostic replay, evidence reduction, "
        "normalization probes and calibration are overhead, not hidden in these budgets. "
        "Each independent seed pays its compilation cost; these timings do not measure "
        "production steady-state throughput.",
        "",
        f"Calibration used {full['calibration']['model_calls']:,} model calls, including "
        f"{full['calibration']['gradient_calls']:,} gradient calls, and took "
        f"{full['calibration']['seconds_including_compilation']:.2f} seconds. "
        f"Amortized over the 60 tuned runs that is {full['calibration']['model_calls'] / 60:,.0f} "
        "extra model calls/run (3.00% of a 4M nominal budget). The shared control comparison "
        "still uses identical evaluation budgets; calibration is disclosed separately.",
        "",
        "Raw state.pkl files, complete arrays, evidence inputs/draws, replay diagnostics, "
        "manifests/checksums and source archive remain under this output directory. "
        "All three independent coordinator audits (pilot, held-out and complete cohort) "
        "passed; their sources and receipts are retained, including independent agreement "
        "on every reported paired interval. The failed serializer attempt is preserved. "
        "No production code, manuscript or GitHub artifact was changed.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(report))
    path = root / "REPORT.md"
    path.with_suffix(".md.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument(
        "--report-only", action="store_true",
        help="Render REPORT.md from SUMMARY.json without repeating checks or figures.",
    )
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    if options.report_only:
        write_report(options.root, json.loads((options.root / "SUMMARY.json").read_text()))
        return
    import matplotlib
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import norm

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = options.root
    rows, final_arrays, hashes = [], {}, {}
    runtimes, audit_count = [], 0
    for arm in ("fixed", "tuned"):
        for allocation in ("uniform", "evidence_improving"):
            for seed in range(30):
                directory = root / "runs" / arm / allocation / f"seed-{seed:02d}"
                manifest = json.loads((directory / "MANIFEST.json").read_text())
                core = json.loads((directory / "COMPLETE.json").read_text())
                final = json.loads((directory / "ANALYSIS-COMPLETE.json").read_text())
                assert final["passed"] and final["evidence_complete"] and core["core_complete"]
                assert manifest["seed"] == seed and manifest["arm"] == arm
                assert manifest["allocation"] == allocation and not manifest["smoke"]
                assert manifest["budgets"] == [2000000, 4000000]
                assert manifest["initial_roots"] == manifest["replacement_width"] == 150
                assert manifest["delta_K"] == 150 and manifest["unit_dtype"] == "float64"
                assert manifest["phantoms"] == 8 and not manifest["phantom_seed_selection"]
                assert manifest["backend"] == "cpu" and len(manifest["cpu_affinity"]) == 1
                assert manifest["kernel"] == dict(
                    num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=0.003
                )
                assert (
                    hashlib.sha256((directory / "state.pkl").read_bytes()).hexdigest()
                    == core["state_sha256"]
                )
                hashes.setdefault("source", set()).add(
                    json.dumps(manifest["source_sha256"], sort_keys=True)
                )
                hashes.setdefault("schedule", set()).add(manifest["schedule_sha256"])
                runtimes.append(
                    {
                        "arm": arm,
                        "allocation": allocation,
                        "seed": seed,
                        "core_seconds": core["seconds"],
                        "analysis_seconds": final["seconds"],
                        "core_rss_gib": core["max_rss_kib"] / 1024**2,
                        "analysis_rss_gib": final["max_rss_kib"] / 1024**2,
                    }
                )
                for budget in (2000000, 4000000):
                    row = json.loads((directory / f"ANALYSIS-{budget}.json").read_text())
                    assert row["audit_passed"] and row["replay_exact"]
                    assert row["mc_draws"] == 4096 and row["C_min"] == 20
                    hashes.setdefault(seed, set()).add(row["initial_sha256"])
                    for name, field in (
                        (f"calls-{budget}.npz", "archive_sha256"),
                        (f"inputs-{budget}.npz", "phantom_archive_sha256"),
                        (f"evidence-{budget}.npz", "evidence_sha256"),
                    ):
                        assert (
                            hashlib.sha256((directory / name).read_bytes()).hexdigest()
                            == row[field]
                        )
                    with np.load(directory / f"evidence-{budget}.npz") as draws:
                        for estimator in ("classic_mc", "phantom"):
                            values = draws[f"{estimator}_log_Z_samples"]
                            assert values.shape == (4096,) and np.isfinite(values).all()
                            np.testing.assert_equal(values.mean(), row[f"{estimator}_log_Z"])
                            np.testing.assert_equal(
                                values.std(), row[f"{estimator}_log_Z_uncert"]
                            )
                            np.testing.assert_equal(
                                values.std(ddof=1) / 64, row[f"{estimator}_mcse"]
                            )
                            np.testing.assert_array_equal(
                                np.quantile(values, [0.025, 0.975]),
                                row[f"{estimator}_percentile95"],
                            )
                    # Independent final row-to-block lineage check, including padding.
                    with np.load(directory / f"inputs-{budget}.npz") as evidence_inputs:
                        count = int(evidence_inputs["num_samples"])
                        block_values = evidence_inputs["log_L_blocks"]
                        indices_by_value = np.searchsorted(
                            block_values, evidence_inputs["log_L_classic"][:count], side="left",
                        )
                        np.testing.assert_array_equal(
                            evidence_inputs["K_classic"][:count],
                            evidence_inputs["incoming_K"][indices_by_value],
                        )
                        assert np.all(evidence_inputs["K_classic"][count:] == 0)
                    rows.append(row)
                    audit_count += 1
                    if budget == 4000000:
                        with np.load(directory / f"calls-{budget}.npz") as raw:
                            final_arrays[arm, allocation, seed] = (
                                raw["samples"],
                                np.exp(raw["log_weights"] - logsumexp(raw["log_weights"])),
                            )
    assert audit_count == 240 and len(final_arrays) == 120
    assert all(len(value) == 1 for value in hashes.values())
    indices = np.random.default_rng(3122706).integers(0, 30, size=(100000, 30))
    summary, bootstrap = {}, {}
    estimators = ("classic_moments", "classic_mc", "phantom")
    for allocation in ("uniform", "evidence_improving"):
        for arm in ("fixed", "tuned"):
            for budget in (2000000, 4000000):
                group = sorted(
                    [
                        row
                        for row in rows
                        if row["arm"] == arm
                        and row["allocation"] == allocation
                        and row["budget"] == budget
                    ],
                    key=lambda row: row["seed"],
                )
                assert [row["seed"] for row in group] == list(range(30))
                key = f"{arm}_{allocation}_{budget}"
                info = {
                    "n": 30,
                    "estimators": {},
                    "mean_calls": float(np.mean([row["evaluations"] for row in group])),
                    "sd_calls": float(np.std([row["evaluations"] for row in group], ddof=1)),
                    "maximum_overshoot": max(row["evaluations"] - budget for row in group),
                    "mean_gradients": float(
                        np.mean([row["gradient_evaluations"] for row in group])
                    ),
                    "mean_epochs": float(np.mean([row["completed_epochs"] for row in group])),
                    "median_weight_ess": float(np.median([row["kish_ess"] for row in group])),
                    "median_marginal_sd": {
                        f"x{i}": float(np.median([row["std"][i] for row in group]))
                        for i in (1, 16, 17, 5)
                    },
                    "median_w1": {
                        f"x{i}": float(np.median([row[f"w1_x{i}"] for row in group]))
                        for i in (1, 16, 17, 5)
                    },
                    "median_cdf_error": {
                        f"x{i}": float(np.median([row[f"cdf_error_x{i}"] for row in group]))
                        for i in (1, 16, 17, 5)
                    },
                    "max_tail_fraction": max(
                        row["classic_tail_upper_fraction"] for row in group
                    ),
                    "movement": {},
                }
                for estimator in estimators:
                    prefix = "" if estimator == "classic_moments" else estimator + "_"
                    z = np.asarray([row[prefix + "log_Z"] for row in group])
                    sd = np.asarray([row[prefix + "log_Z_uncert"] for row in group])
                    boot = np.sqrt(np.mean(z[indices] ** 2, axis=1))
                    bootstrap[key, estimator] = boot
                    item = {
                        "bias": float(z.mean()),
                        "error_sd": float(z.std(ddof=1)),
                        "rmse": float(np.sqrt(np.mean(z * z))),
                        "rmse_bootstrap_se": float(boot.std(ddof=1)),
                        "rmse_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                        "mean_reported_sd": float(sd.mean()),
                        "coverage95_count": int(np.sum(abs(z) <= 1.959963984540054 * sd)),
                        "coverage95": float(np.mean(abs(z) <= 1.959963984540054 * sd)),
                        "standardized_error_rms": float(np.sqrt(np.mean((z / sd) ** 2))),
                        "standardized_error_variance": float(np.var(z / sd, ddof=1)),
                        "errors": z.tolist(),
                        "reported_sd": sd.tolist(),
                    }
                    if estimator != "classic_moments":
                        item["mean_mcse"] = float(
                            np.mean([row[prefix + "mcse"] for row in group])
                        )
                        item["maximum_mcse"] = max(row[prefix + "mcse"] for row in group)
                        item["mc_percentile_coverage95_count"] = sum(
                            row[prefix + "percentile95"][0]
                            <= 0
                            <= row[prefix + "percentile95"][1]
                            for row in group
                        )
                    info["estimators"][estimator] = item
                for i in range(6):
                    moves = [row["movement"][i] for row in group]
                    metrics = {
                        "median_pairs": float(np.median([move["pairs"] for move in moves]))
                    }
                    for field in (
                        "acceptance",
                        "mean_realised_h",
                        "rms_dx1",
                        "rms_dx16",
                        "rms_dx17",
                        "rms_dx5",
                        "corr_x1",
                        "corr_x16",
                        "corr_x17",
                        "corr_x5",
                        "rms_dx1_per_gradient",
                        "rms_dx16_per_gradient",
                    ):
                        values = [
                            move[field]
                            for move in moves
                            if field in move and move[field] is not None
                        ]
                        metrics[field] = float(np.median(values)) if values else None
                        metrics[field + "_runs"] = len(values)
                    info["movement"][str(i)] = metrics
                summary[key] = info
    contrasts = {}
    for allocation in ("uniform", "evidence_improving"):
        for budget in (2000000, 4000000):
            for estimator in estimators:
                fixed, tuned = f"fixed_{allocation}_{budget}", f"tuned_{allocation}_{budget}"
                difference = bootstrap[tuned, estimator] - bootstrap[fixed, estimator]
                contrasts[f"tuned_minus_fixed_{allocation}_{budget}_{estimator}"] = {
                    "delta_rmse": summary[tuned]["estimators"][estimator]["rmse"]
                    - summary[fixed]["estimators"][estimator]["rmse"],
                    "paired_bootstrap95": np.quantile(difference, [0.025, 0.975]).tolist(),
                    "paired_bootstrap_se": float(difference.std(ddof=1)),
                }
            for arm in ("fixed", "tuned"):
                key = f"{arm}_{allocation}_{budget}"
                difference = bootstrap[key, "phantom"] - bootstrap[key, "classic_mc"]
                contrasts[f"phantom_minus_classic_mc_{key}"] = {
                    "delta_rmse": summary[key]["estimators"]["phantom"]["rmse"]
                    - summary[key]["estimators"]["classic_mc"]["rmse"],
                    "paired_bootstrap95": np.quantile(difference, [0.025, 0.975]).tolist(),
                }
    full = {
        "cells": summary,
        "contrasts": contrasts,
        "runtimes": runtimes,
        "audit": {
            "runs": 120,
            "snapshots": 240,
            "all_seeds": list(range(30)),
            "matched_priors": True,
            "single_source": True,
            "single_schedule": True,
            "bootstrap_samples": 100000,
            "bootstrap_seed": 3122706,
            "max_likelihood_discrepancy": max(
                row["max_likelihood_discrepancy"] for row in rows
            ),
        },
        "calibration": json.loads((root / "calibration/SCHEDULE.json").read_text()),
    }
    (root / "SUMMARY.json").write_text(json.dumps(full, indent=2, allow_nan=False) + "\n")
    fields = [
        "arm",
        "allocation",
        "seed",
        "budget",
        "evaluations",
        "gradient_evaluations",
        "completed_epochs",
        "log_Z",
        "log_Z_uncert",
        "classic_mc_log_Z",
        "classic_mc_log_Z_uncert",
        "classic_mc_mcse",
        "phantom_log_Z",
        "phantom_log_Z_uncert",
        "phantom_mcse",
        "kish_ess",
        "w1_x1",
        "w1_x16",
        "w1_x17",
        "w1_x5",
        "cdf_error_x1",
        "cdf_error_x16",
        "cdf_error_x17",
        "cdf_error_x5",
        "classic_tail_upper_fraction",
        "initial_sha256",
    ]
    with (root / "ALL_RECORDS.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    colors = {"fixed": "#3a6b91", "tuned": "#b85e2b"}
    for r, allocation in enumerate(("uniform", "evidence_improving")):
        for arm in ("fixed", "tuned"):
            info = summary[f"{arm}_{allocation}_4000000"]
            for c, estimator in enumerate(estimators):
                values = [
                    summary[f"{arm}_{allocation}_{budget}"]["estimators"][estimator]["rmse"]
                    for budget in (2000000, 4000000)
                ]
                axes[r, c].plot([2, 4], values, "o-", color=colors[arm], label=arm)
                axes[r, c].set(
                    title=f"{allocation}: {estimator}",
                    xlabel="Logical model calls (millions)",
                    ylabel="Log-evidence RMSE",
                    xticks=[2, 4],
                )
                axes[r, c].legend()
    for extension in ("png", "pdf"):
        figure.savefig(root / f"evidence_comparison.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(4, 4, figsize=(13, 11), constrained_layout=True)
    for r, (allocation, arm) in enumerate(
        (a, b) for a in ("uniform", "evidence_improving") for b in ("fixed", "tuned")
    ):
        for seed in range(30):
            points, weights = final_arrays[arm, allocation, seed]
            for c, i in enumerate((1, 16, 17, 5)):
                order = np.argsort(points[:, i])
                axes[r, c].plot(
                    points[order, i],
                    np.cumsum(weights[order]),
                    color=colors[arm],
                    alpha=0.3,
                    linewidth=0.7,
                )
        for c, i in enumerate((1, 16, 17, 5)):
            grid = np.linspace(0.3, 0.7, 200) if i == 5 else np.linspace(0, 1, 200)
            axes[r, c].plot(grid, norm.cdf(grid, 0.5, 0.05) if i == 5 else grid, "k--")
            axes[r, c].set(
                title=f"{arm}/{allocation}: x{i}", ylim=(0, 1), xlim=(grid.min(), grid.max())
            )
    for extension in ("png", "pdf"):
        figure.savefig(root / f"posterior_cdfs.{extension}", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for r, allocation in enumerate(("uniform", "evidence_improving")):
        for arm in ("fixed", "tuned"):
            movement = summary[f"{arm}_{allocation}_4000000"]["movement"]
            for c, field in enumerate(("rms_dx1", "rms_dx16")):
                axes[r, c].plot(
                    range(6),
                    [movement[str(i)][field] for i in range(6)],
                    "o-",
                    color=colors[arm],
                    label=arm,
                )
                axes[r, c].axhline(np.sqrt(1 / 6), color="k", linestyle="--")
                axes[r, c].set(
                    title=f"{allocation}: {field}",
                    yscale="log",
                    xticks=range(6),
                    xticklabels=["0–5", "5–15", "15–40", "40–100", "100–1000", ">1000"],
                    xlabel="Parent peak-gap bin",
                    ylabel="Absolute RMS movement (cube units)",
                )
                axes[r, c].legend()
    for extension in ("png", "pdf"):
        figure.savefig(root / f"chain_movement.{extension}", dpi=160)
    plt.close(figure)
    write_report(root, full)
    for path in root.iterdir():
        if path.suffix in (".json", ".csv", ".md", ".pdf", ".png") and path.is_file():
            path.with_suffix(path.suffix + ".sha256").write_text(
                hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
            )
    print(json.dumps(full["audit"]))


if __name__ == "__main__":
    main()
