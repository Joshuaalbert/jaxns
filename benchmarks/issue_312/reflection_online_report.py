"""Audit, diagnose and report the released online reflection experiment.

This module consumes completed records only. It neither selects a policy nor
changes sampling, evidence reductions, or the prespecified bootstrap contrasts.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ARMS = ("fixed", "online", "frozen")
ALLOCATIONS = ("uniform", "evidence_improving")
ESTIMATORS = ("classic_moments", "classic_mc", "phantom")
COLORS = {"fixed": "#3a6b91", "online": "#b85e2b", "frozen": "#2f855a"}


def add_online_diagnostics(root: Path, full: dict, rows: list[dict],
                           freeze: dict, release: dict) -> None:
    """Bind raw counts, controller histories and actual overhead to the summary."""
    diagnostics = []
    for row in rows:
        arm, allocation, seed, budget = (row[k] for k in ("arm", "allocation", "seed", "budget"))
        directory = root / "evaluation" / arm / allocation / f"seed-{seed:02d}"
        for name, field in ((f"controller-{budget}.npz", "controller_ledger_sha256"),
                            (f"policies-{budget}.npz", "controller_policies_sha256")):
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == row[field]
        audit = json.loads((directory / "CONTROLLER-AUDIT.json").read_text())
        assert audit["passed"] and audit["chronological_replay_exact"]
        assert audit["snapshots"][str(budget)] == row["controller_audit"]
        with np.load(directory / f"calls-{budget}.npz") as raw:
            child = raw["seed_indices"] >= 0  # [num_samples]
            bounces = raw["bounce_counts"]
            gradients = raw["gradient_evaluations_per_sample"]
            values = raw["value_only_evaluations_per_sample"]
            calls = raw["evaluations_per_sample"]
            np.testing.assert_array_equal(gradients, 9 * bounces)
            np.testing.assert_array_equal(calls, values + gradients)
            assert not bounces[~child].any() and not raw["nominal_steps"][~child].any()
            assert np.all(calls[~child] == 1) and int((~child).sum()) == row["roots"]
            assert int(calls.sum()) == row["evaluations"]
            assert int(gradients.sum()) == row["gradient_evaluations"]
            assert int(values.sum()) == row["value_only_evaluations"]
            assert np.all((bounces[child] >= 1) & (bounces[child] <= 100))
            if arm == "fixed":
                assert np.all(bounces[child] == 10)
                np.testing.assert_array_equal(raw["nominal_steps"][child], 0.003)
        with np.load(directory / f"controller-{budget}.npz") as ledger:
            child = ledger["seed_indices"] >= 0
            requested_h = ledger["requested_step"][child]
            requested_t = ledger["requested_duration"][child]
            effective_h = ledger["step_size"][child]
            actual_t = ledger["nominal_duration"][child]
            np.testing.assert_allclose(actual_t, ledger["num_bounces"][child] * effective_h,
                                       rtol=2e-15, atol=0)
            # Raw requests precede clipping; isolate integer quantization from bounds.
            bounded_h = np.clip(requested_h, .001, .03)
            bounded_t = np.clip(requested_t, .003, .1)
            np.testing.assert_allclose(actual_t, bounded_t, rtol=2e-15, atol=0)
            children = int(child.sum())
            flags = {name: int(ledger[name][child].sum()) for name in (
                "bound_clipped", "duration_capped", "extrapolated",
                "extrapolation_clamped", "compression_clipped",
            )}
            choices = np.bincount(ledger["choice"][child], minlength=5).tolist()
            observed = {
                "children": children,
                "choice_counts": choices,
                "flag_counts": flags,
                "requested_h_range": [float(requested_h.min()), float(requested_h.max())],
                "requested_t_range": [float(requested_t.min()), float(requested_t.max())],
                "effective_h_range": [float(effective_h.min()), float(effective_h.max())],
                "nominal_t_range": [float(actual_t.min()), float(actual_t.max())],
                "mean_bounces": float(ledger["num_bounces"][child].mean()),
                "mean_effective_h": float(effective_h.mean()),
                "mean_nominal_t": float(actual_t.mean()),
                "maximum_ceil_h_reduction": float(np.max(1 - effective_h / bounded_h)),
            }
        history = [json.loads(line) for line in
                   (directory / "controller-history.jsonl").read_text().splitlines()]
        history = [item for item in history if item["stop"] <= row["samples"]]
        reasons = Counter(reason for item in history for eligible in item["eligibility"]
                          for reason in eligible["reasons"])
        observed.update(
            arm=arm, allocation=allocation, seed=seed, budget=budget,
            controller_version=row["controller_version"],
            real_parameter_change_batches=sum(item["real_parameter_change"] for item in history),
            fully_eligible_stratum_events=sum(not e["reasons"] for item in history
                                              for e in item["eligibility"]),
            suppression_reason_counts=dict(reasons),
            cumulative_information_exposure=sum(
                row["controller_audit"]["cumulative_information_exposure"]),
            deepest_observed_compression=row["controller_audit"]["deepest_observed_compression"],
            prepare_seconds=row["prepare_seconds"],
            feedback_seconds=row["feedback_seconds"],
            controller_seconds=row["controller_seconds"],
        )
        assert observed["real_parameter_change_batches"] == row["controller_real_updates"]
        assert flags == row["controller_audit"]["flag_counts"]
        assert choices == row["controller_audit"]["choice_counts"]
        diagnostics.append(observed)

    for runtime in full["runtimes"]:
        arm, allocation, seed = (runtime[k] for k in ("arm", "allocation", "seed"))
        directory = root / "evaluation" / arm / allocation / f"seed-{seed:02d}"
        progress = [json.loads(line) for line in (directory / "progress.jsonl").read_text()
                    .splitlines()]
        log = (root / "logs" / f"evaluation-{arm}-{allocation}-{seed}-core.log").read_text()
        compiles = re.findall(r"Finished XLA compilation of (.+?) in ([\d.e+-]+) sec", log)
        runtime.update(
            xla_compile_events=len(compiles),
            xla_compile_seconds_full_process=sum(float(seconds) for _, seconds in compiles),
            depth_cache_max=max(item["depth_compile_cache_size"] for item in progress),
            capacities=sorted({item["capacity"] for item in progress}),
        )
        final = next(row for row in rows if row["arm"] == arm
                     and row["allocation"] == allocation and row["seed"] == seed
                     and row["budget"] == 4000000)
        runtime.update({key: final[key] for key in (
            "replay_seconds_full_run", "replay_evaluations_full_run", "replay_gradients_full_run",
            "controller_audit_seconds_full_run", "prepare_seconds", "feedback_seconds",
            "controller_seconds",
        )})

    for allocation in ALLOCATIONS:
        for arm in ARMS:
            for budget in (2000000, 4000000):
                group = [d for d in diagnostics if d["arm"] == arm
                         and d["allocation"] == allocation and d["budget"] == budget]
                assert len(group) == 30
                children = sum(d["children"] for d in group)
                flags = {name: sum(d["flag_counts"][name] for d in group)
                         for name in group[0]["flag_counts"]}
                info = full["cells"][f"{arm}_{allocation}_{budget}"]
                info["controller"] = dict(
                    children=children,
                    choice_counts=np.sum([d["choice_counts"] for d in group], axis=0).tolist(),
                    flag_counts=flags, flag_fractions={k: v / children for k, v in flags.items()},
                    real_updates_range=[min(d["real_parameter_change_batches"] for d in group),
                                        max(d["real_parameter_change_batches"] for d in group)],
                    median_real_updates=float(np.median(
                        [d["real_parameter_change_batches"] for d in group])),
                    median_information_exposure=float(np.median(
                        [d["cumulative_information_exposure"] for d in group])),
                    median_deepest_compression=float(np.median(
                        [d["deepest_observed_compression"] for d in group])),
                    maximum_ceil_h_reduction=max(d["maximum_ceil_h_reduction"] for d in group),
                )
                for field in ("mean_bounces", "mean_effective_h", "mean_nominal_t"):
                    info["controller"]["median_" + field] = float(np.median(
                        [d[field] for d in group]))
                records = [row for row in rows if row["arm"] == arm
                           and row["allocation"] == allocation and row["budget"] == budget]
                pooled = []
                for stratum in range(64):
                    moves = [move for row in records
                             for move in row["controller_audit"]["compression_movement"]
                             if move["stratum"] == stratum]
                    if not moves:
                        continue
                    jumps = np.sum([move["squared_displacement_sums"] for move in moves], axis=0)
                    square_jumps = np.sum([move["squared_coordinate_jump_sums"]
                                           for move in moves], axis=0)
                    work = sum(move["total_calls"] for move in moves)
                    grads = sum(move["gradient_calls"] for move in moves)
                    pooled.append(dict(
                        stratum=stratum, contributing_runs=len(moves),
                        pairs=sum(move["pairs"] for move in moves), total_calls=work,
                        gradient_calls=grads, value_only_calls=work-grads,
                        squared_displacement_sums=jumps.tolist(),
                        squared_coordinate_jump_sums=square_jumps.tolist(),
                        esjd_per_total_call=(jumps / work).tolist(),
                        esjd_per_gradient=(jumps / grads).tolist(),
                        whole_chain_objective=float(jumps.sum() / work),
                    ))
                info["compression_movement"] = pooled

    training = json.loads((root / "TRAINING_SUMMARY.json").read_text())
    full["training"] = {
        "total_evaluations": training["total_evaluations"],
        "runs": [{k: v for k, v in run.items() if k != "controller_audit"}
                 for run in training["runs"]],
    }
    full["controller_records"] = diagnostics
    full["frozen_policies"] = {
        allocation: json.loads((root / "frozen" / f"{allocation}.json").read_text())
        for allocation in ALLOCATIONS
    }
    full["audit"].update(
        source_commit=freeze["source_commit"], protocol_sha256=freeze["protocol_sha256"],
        source_freeze_sha256=release["source_freeze_sha256"],
        validation_sha256=release["validation_sha256"],
        frozen_policy_sha256=release["frozen_policy_sha256"],
        evaluation_release_sha256=hashlib.sha256(
            (root / "EVALUATION_RELEASE.json").read_bytes()).hexdigest(),
        reference_log_Z=0., primary_budget=4000000,
        raw_counts_roots_and_controller_history_passed=True,
        reporting_source_sha256={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("summarise_reflection_tuning.py", "summarise_reflection_online.py",
                         "reflection_online_report.py")
        },
    )


def write_online_report(root: Path, full: dict) -> None:
    """Render all contrasts, without choosing an arm or estimator after inspection."""
    cells, contrasts = full["cells"], full["contrasts"]
    primary = {k: v for k, v in contrasts.items() if "_4000000_" in k
               and k.startswith(("online_minus_fixed_", "frozen_minus_fixed_"))}
    better = [k for k, v in primary.items() if v["paired_bootstrap95"][1] < 0]
    worse = [k for k, v in primary.items() if v["paired_bootstrap95"][0] > 0]
    frozen_evidence = contrasts["frozen_minus_fixed_evidence_improving_4000000_classic_mc"]
    frozen_rmse = cells["frozen_evidence_improving_4000000"]["estimators"]["classic_mc"]["rmse"]
    fixed_rmse = cells["fixed_evidence_improving_4000000"]["estimators"]["classic_mc"]["rmse"]
    frozen_lo, frozen_hi = frozen_evidence["paired_bootstrap95"]
    report = [
        "# Online reflection resolution and duration tuning", "",
        "Online learning did not resolve an evidence-RMSE improvement over the fixed "
        "control at 4M calls under either allocation. The independently trained frozen "
        f"policy with evidence allocation reduced classic-MC RMSE from {fixed_rmse:.5f} "
        f"to {frozen_rmse:.5f}: difference {frozen_evidence['delta_rmse']:+.5f}, paired "
        f"95% interval [{frozen_lo:+.5f}, {frozen_hi:+.5f}]. Its classic-moments and "
        "phantom-MC contrasts agree in direction and exclude zero; these are correlated "
        "estimators on the same cohort, not independent replications. No frozen-policy "
        "advantage was resolved under uniform allocation.", "",
        f"At the prespecified 4M-call endpoint, {len(better)} of the twelve primary "
        f"arm-minus-fixed contrasts have a negative paired 95% RMSE interval, and "
        f"{len(worse)} have a positive interval. The other intervals include zero. "
        "These are pointwise intervals, without multiplicity adjustment; inclusion "
        "of zero does not establish equivalence. All contrasts are reported below.", "",
        "The complete cohort contains 180 CPU runs: three arms, two "
        "allocations and the same 30 fresh seeds 60–89. All 360 snapshots are retained. "
        "Four million logical model calls is primary; two million is secondary. "
        "The analytic reference log evidence is zero. Arms are paired by seed and "
        "initial population, not by completed race tree.", "",
        "The fixed arm uses h=0.003, B=10 and nominal T=0.03. Online learns h and T "
        "from whole-chain squared displacement summed over all 18 cube coordinates "
        "divided by actual total model work. Nine cycles, one isotropic slice per "
        "cycle, independent U(0.8,1.2) jitter, Gaussian velocities, strict endpoint "
        "rejection and retained self-transitions remain unchanged. No replay or "
        "scouting model calls feed the controller. Eight phantoms condition evidence "
        "only: they never supply seeds, posterior weights or controller feedback.", "",
        "Online and frozen both use 90% central settings and four 2.5% perturbations. "
        "Frozen lookup uses actual parent likelihood and a table averaged equally "
        "in log parameters across four independent training runs per allocation; "
        "it never reconstructs compression from its evaluation population. The four "
        "training seeds are paired across allocations and separate from evaluation; "
        "they are independent within each allocation, not eight mutually independent "
        "training seeds. Online "
        "uses only the classic tree available before dispatch. The information clock "
        "sums log1p(1/K_parent) over new completed nonroot chains. Revisits supply "
        "new observations, while deepest frontier compression is reported separately.", "",
        "Desired h is bounded to [0.001,0.03] and requested T to [0.003,0.1]. "
        "B=ceil(T/h), with the prespecified near-integer rounding guard; effective "
        "h=T/B precedes jitter. Effective and jittered h can be below the desired "
        "lower bound. The old B≤256 cap is inactive because the new domain has B≤100. "
        "This domain was fixed before training and evaluation following an independent "
        "numerical range failure; no integrator or tolerance was changed.", "",
        "## Evidence estimates", "",
        "Error SD uses ddof=1. Gaussian coverage uses estimate ±1.959964 reported SD; "
        "MC coverage uses the empirical 2.5%–97.5% draw interval. Each MC estimator "
        "uses 4,096 shrinkage draws. Classic moments, classic MC and phantom MC are "
        "distinct estimators. Bootstrap SEs and intervals use the same 100,000 paired "
        "seed resamples throughout, RNG 3122903. Per-seed errors, reported SDs, MCSEs "
        "and standardized-error diagnostics are retained in SUMMARY.json and records.", "",
    ]
    for budget in (4000000, 2000000):
        report += [f"### {budget // 1000000}M calls", "",
                   "| Allocation | Arm | Estimator | Bias | Error SD | RMSE | SE | "
                   "Reported SD | Gaussian coverage | MC coverage | Mean MCSE |",
                   "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for allocation in ALLOCATIONS:
            for arm in ARMS:
                for estimator in ESTIMATORS:
                    item = cells[f"{arm}_{allocation}_{budget}"]["estimators"][estimator]
                    mc = (f"{item['mc_percentile_coverage95_count']}/30"
                          if estimator != "classic_moments" else "—")
                    mcse = f"{item['mean_mcse']:.5f}" if "mean_mcse" in item else "—"
                    report.append(
                        f"| {allocation} | {arm} | {estimator} | {item['bias']:+.5f} | "
                        f"{item['error_sd']:.5f} | {item['rmse']:.5f} | "
                        f"{item['rmse_bootstrap_se']:.5f} | {item['mean_reported_sd']:.5f} | "
                        f"{item['coverage95_count']}/30 | {mc} | {mcse} |")
        report += ["", "| Allocation | Contrast | Estimator | RMSE difference | Paired 95% CI |",
                   "|---|---|---|---:|---|"]
        for allocation in ALLOCATIONS:
            for left, right in (("online", "fixed"), ("frozen", "fixed"), ("online", "frozen")):
                for estimator in ESTIMATORS:
                    item = contrasts[f"{left}_minus_{right}_{allocation}_{budget}_{estimator}"]
                    lo, hi = item["paired_bootstrap95"]
                    report.append(f"| {allocation} | {left} − {right} | {estimator} | "
                                  f"{item['delta_rmse']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
        report += ["", "| Allocation | Arm | Phantom − classic MC RMSE | Paired 95% CI |",
                   "|---|---|---:|---|"]
        for allocation in ALLOCATIONS:
            for arm in ARMS:
                item = contrasts[f"phantom_minus_classic_mc_{arm}_{allocation}_{budget}"]
                lo, hi = item["paired_bootstrap95"]
                report.append(f"| {allocation} | {arm} | {item['delta_rmse']:+.5f} | "
                              f"[{lo:+.5f}, {hi:+.5f}] |")
        report.append("")

    report += ["## Posterior recovery at 4M calls", "",
               "Only classic posterior weights participate. SD, CDF error and W1 below are "
               "medians across all 30 runs. Flat-coordinate reference SD is 0.288675; "
               "x5 is approximately 0.05 with the unchanged model's TFP scale precision. "
               "CDF sup error checks both sides of every weighted empirical jump. The "
               "unchanged W1 reduction uses 8,192 target midpoint quantiles, a deterministic "
               "approximation to continuous-target W1. Kish ESS measures weight concentration, "
               "not Markov-chain mixing.", "",
               "| Allocation | Arm | Coordinate | SD | CDF sup error | W1 |",
               "|---|---|---|---:|---:|---:|"]
    for allocation in ALLOCATIONS:
        for arm in ARMS:
            item = cells[f"{arm}_{allocation}_4000000"]
            for coordinate in ("x1", "x16", "x17", "x5"):
                report.append(f"| {allocation} | {arm} | {coordinate} | "
                              f"{item['median_marginal_sd'][coordinate]:.5f} | "
                              f"{item['median_cdf_error'][coordinate]:.5f} | "
                              f"{item['median_w1'][coordinate]:.5f} |")
    report += ["", "## Controller learning and movement", "",
               "The following are per-run medians except flags, which pool all children. "
               "Real updates change h/T; consuming observations with a zero delta is not "
               "counted as learning. Information exposure and deepest compression have "
               "different meanings. Full chronological receipts preserve eligibility "
               "and suppression reasons, raw jump/work sums and every parameter history.", "",
               "| Allocation | Arm | Real updates (range) | Exposure | Deepest c | Mean B | "
               "Mean effective h | Mean nominal T | Extrapolated / clamped |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for allocation in ALLOCATIONS:
        for arm in ARMS:
            item = cells[f"{arm}_{allocation}_4000000"]["controller"]
            low, high = item["real_updates_range"]
            flags = item["flag_fractions"]
            report.append(
                f"| {allocation} | {arm} | {item['median_real_updates']:.1f} ({low}–{high}) | "
                f"{item['median_information_exposure']:.2f} | "
                f"{item['median_deepest_compression']:.2f} | "
                f"{item['median_mean_bounces']:.3f} | {item['median_mean_effective_h']:.6f} | "
                f"{item['median_mean_nominal_t']:.6f} | {flags['extrapolated']:.3%} / "
                f"{flags['extrapolation_clamped']:.3%} |")
    flags = Counter()
    for item in full["controller_records"]:
        if item["budget"] == 4000000:
            flags.update(item["flag_counts"])
    quantization = max(item["maximum_ceil_h_reduction"] for item in full["controller_records"])
    report += ["", f"Across final snapshots: bound clipping {flags['bound_clipped']}, "
               f"duration caps {flags['duration_capped']}, compression-range clips "
               f"{flags['compression_clipped']}. Maximum observed ceil reduction in effective "
               f"h versus bounded desired h was {quantization:.3%}. "
               "All occurrences are retained.", "",
               "Gap-bin movement and controller-compression-stratum movement are both retained. "
               "Whole-chain displacement includes the ordinary slice; endpoint rejection "
               "does not imply zero whole-chain movement. Compression aggregates retain all "
               "18 coordinate and squared-coordinate jump sums, actual total/gradient/value "
               "work, and per-run correlations in the underlying analyses. Ratios pool jump "
               "sums and denominators, not individual ratios. Figures show posterior CDFs, "
               "gap-bin movement, compression objectives and learned parameter curves.", "",
               "## Work and runtime", "",
               "Every gradient counts once in logical model work. Roots, including later "
               "allocations, have one value call and zero B/h/T/gradients. Complete batches "
               "cross each budget; overshoot is retained. Masked heterogeneous vectorized "
               "execution can do more hardware work than these logical counts. Core wall "
               "time includes its compilation and writes; separately logged full-process "
               "XLA compilation also includes setup outside that timer and is not subtracted "
               "from it. Prepare, feedback and update timers are within core work; replay "
               "and reduction are separately charged analysis overhead.", "",
               "| Allocation | Arm | Mean calls (SD) | Mean gradients | Max overshoot | "
               "Core s | Analysis s | Full-process XLA s | Prepare s | Feedback s | Update s |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for allocation in ALLOCATIONS:
        for arm in ARMS:
            cell = cells[f"{arm}_{allocation}_4000000"]
            times = [r for r in full["runtimes"] if r["arm"] == arm
                     and r["allocation"] == allocation]
            fields = ("core_seconds", "analysis_seconds", "xla_compile_seconds_full_process",
                      "prepare_seconds", "feedback_seconds", "controller_seconds")
            timings = " | ".join(f"{np.median([r[f] for r in times]):.3f}" for f in fields)
            report.append(f"| {allocation} | {arm} | {cell['mean_calls']:.0f} "
                          f"({cell['sd_calls']:.0f}) | {cell['mean_gradients']:.0f} | "
                          f"{cell['maximum_overshoot']} | {timings} |")
    training = full["training"]
    report += ["", f"Independent frozen-policy training cost {training['total_evaluations']:,} "
               "model calls across eight 4M runs, plus their full replay/reductions. "
               f"Amortized over the 60 frozen evaluation runs, this is "
               f"{training['total_evaluations']/60:,.0f} additional calls per run "
               f"({100*training['total_evaluations']/60/4000000:.2f}% of 4M). Online learning "
               "uses no extra model calls for feedback; it still incurs host bookkeeping "
               "and can change the work per chain. Neither training nor held-out ensembles "
               "are hidden inside the evaluation budgets.", "",
               "## Validation, interpretation and limits", "",
               "Pre-release focused/original tests passed, including fixed "
               "child/phantom/count/key parity, phantom-off/on continuation parity, fresh "
               "checkpoint restoration, new-root accounting, chronological updates, bounds "
               "and same-shape compilation stability. Both allocation smokes pass complete "
               "chain and phantom replay. Independent audits pass all eight training ledgers, "
               "both frozen-table reconstructions and 21 held-out cases (42 single-path and "
               "full-chain laws), at the original stationarity and reversal tolerances. "
               "The final combined suite passed all 45 tests, including the direct "
               "constant-correlation regression added after sampling and reduction.", "",
               "The original tiny-step domain failed reversal at B256, T0.00256, h1e-5, "
               "including accepted paths. All failed probes and raw arrays are preserved. "
               "The revised independent 4,480-path range audit passed original tolerances "
               "before training. The intermediate TFP/manual arithmetic comparison used "
               "different vector fields and is labeled separately. Passing these probes "
               "supports the tested range; it does not prove every floating-point path.", "",
               "Online is experimental history-dependent adaptive MCMC. Freezing parameters "
               "for one chain does not establish stationarity of the adaptive history. "
               "Frozen-versus-online differs in training histories and is diagnostic, "
               "not a pure causal test of adaptation bias. Better local displacement "
               "or endpoint acceptance alone does not establish population coverage. "
               "Interpret evidence error together with posterior widths and CDF error; "
               "smaller reported evidence uncertainty alone is not improved accuracy.", "",
               "The controller was exercised: online runs made 8–18 real updates under "
               "uniform allocation and 6–17 under evidence allocation. Therefore the "
               "unresolved online benefit is not explained by zero learning. In the "
               "5–15 gap bin, the frozen evidence arm has slightly less x1 movement "
               "and lower x1 displacement per gradient than fixed, despite its lower "
               "evidence RMSE. A single local movement bin does not explain that benefit. "
               "The final frozen configuration and its independent training history "
               "are promising under evidence allocation, but this cohort does not identify "
               "which policy region or mechanism caused the improvement.", "",
               "Even the frozen evidence arm remains under-dispersed: its median x1 SD "
               "is 0.12204 versus the 0.288675 reference. Phantom conditioning increases "
               "4M RMSE relative to classic MC in all six cells, with positive paired "
               "intervals, while reducing reported uncertainty. It therefore does not "
               "repair the incomplete population coverage in this experiment.", "",
               "The fixed-versus-learned comparisons also change the randomized "
               "parameter mixture: fixed has no perturbations, while online and frozen "
               "share the 90/10 mixture. They assess the complete configurations and "
               "cannot isolate the learned central schedule from that mixture. Only "
               "four independent training runs per allocation informed the frozen tables; "
               "training uncertainty and independent replication remain open.", "",
               "Source 47a9384 and its 144-file freeze remained unchanged throughout "
               "sampling and reductions. The complete 180-run/360-snapshot checks verify "
               "all hashes, paired initial populations, draws, root/gradient/value counts "
               "and controller history. Original raw states, ledgers, policies, draws and "
               "audit sources remain on dorrie. No seeds were discarded, no evaluation "
               "retuning occurred, and no manuscript or GitHub artifact was changed.", ""]
    retries = list((root / "evaluation").glob("*/*/seed-*/SERIALIZATION-RETRY.json"))
    if retries:
        report += ["One frozen/evidence seed-72 reduction initially stopped before chain "
                   "replay because two undefined squared-coordinate correlations could not "
                   "be encoded in strict JSON. Three identical seed coordinates had a "
                   "roundoff-positive SD but zero SD after squaring. A separately hashed "
                   "operational wrapper encoded only nonfinite correlation diagnostics "
                   "as null and rejected every other nonfinite field. The retry passed "
                   "full replay and both reductions; source and raw inputs were unchanged. "
                   "The original failing log, diagnosis and wrapper receipt are retained. "
                   "Successful-analysis timings above exclude this failed diagnostic "
                   "attempt. No sampling run was repeated or omitted.", ""]
    report += ["## Figures", "",
               "[Evidence RMSE](evidence_comparison.png) shows both budgets and all three "
               "estimators; shaded bands are paired-seed bootstrap RMSE intervals. "
               "[Posterior CDFs](posterior_cdfs.png) show every seed at 4M against the "
               "reference curves, without selecting examples.", "",
               "[Gap-bin movement](chain_movement.png) separates displacement, cost, "
               "midpoint count and realized duration. [Compression-stratum movement]"
               "(compression_movement.png) pools raw jump and total-work sums. "
               "[Controller policies](controller_policies.png) show all final online "
               "policies versus classic compression and the independently frozen tables "
               "versus actual likelihood; these are different lookup coordinates.", ""]
    (root / "REPORT.md").write_text("\n".join(report))


def plot_online_controller(root: Path, full: dict) -> None:
    """Plot frozen tables and complete online policy distributions, without selection."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for r, allocation in enumerate(ALLOCATIONS):
        allocation_label = "Uniform" if allocation == "uniform" else "Evidence"
        for seed in range(60, 90):
            directory = root / "evaluation/online" / allocation / f"seed-{seed:02d}"
            with np.load(directory / "policies-4000000.npz") as archive:
                parameters = np.exp(archive["log_parameters"][-1])  # [65, 2]
            for c in range(2):
                axes[r, c].plot(np.arange(65)*2, parameters[:, c],
                                color=COLORS["online"], alpha=.3, linewidth=.8)
        table = full["frozen_policies"][allocation]
        # The displayed coordinate is actual likelihood, as in the frozen lookup.
        contours = np.asarray(table["contours"])
        parameters = np.exp(np.asarray(table["log_parameters"]))
        lower, upper = float(contours.min()), float(contours.max())
        ticks = [lower, -np.sqrt(-lower * 1000), -1000., 0., upper]
        ticks = sorted(set(tick for tick in ticks if lower <= tick <= upper))
        labels = [f"{tick:.1e}" if abs(tick) >= 1000 else f"{tick:.1f}" if tick else "0"
                  for tick in ticks]
        for c, (label, baseline) in enumerate((("Desired h", .003), ("Nominal T", .03))):
            axes[r, c].axhline(baseline, color=COLORS["fixed"], linestyle="--")
            axes[r, c].set(title=f"{allocation_label}: online\n{label}",
                           xlabel="Classic compression c", ylabel=label, xlim=(0, 50))
            axes[r, c+2].plot(contours, parameters[:, c], color=COLORS["frozen"])
            axes[r, c+2].axhline(baseline, color=COLORS["fixed"], linestyle="--")
            axes[r, c+2].set_xscale("symlog", linthresh=10)
            axes[r, c+2].set(title=f"{allocation_label}: frozen\n{label}",
                             xlabel="Parent log likelihood (symlog)", ylabel=label,
                             xlim=(lower, upper))
            axes[r, c+2].xaxis.set_major_locator(FixedLocator(ticks))
            axes[r, c+2].set_xticklabels(labels, fontsize=8)
            axes[r, c+2].xaxis.set_minor_locator(NullLocator())
    for extension in ("png", "pdf"):
        figure.savefig(root / f"controller_policies.{extension}", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for r, allocation in enumerate(ALLOCATIONS):
        for arm in ARMS:
            moves = full["cells"][f"{arm}_{allocation}_4000000"]["compression_movement"]
            centres = [2*m["stratum"]+1 for m in moves]
            fields = ([m["whole_chain_objective"] for m in moves],
                      [m["esjd_per_total_call"][1] for m in moves],
                      [m["esjd_per_total_call"][16] for m in moves])
            for c, values in enumerate(fields):
                axes[r, c].plot(centres, values, color=COLORS[arm], label=arm)
        for c, label in enumerate(("All 18 coordinates", "x1", "x16")):
            axes[r, c].set(title=f"{allocation}: {label}", xlabel="Parent compression stratum",
                           ylabel="Squared displacement / total call", yscale="log")
            axes[r, c].legend()
    for extension in ("png", "pdf"):
        figure.savefig(root / f"compression_movement.{extension}", dpi=160)
    plt.close(figure)
