"""Human-readable, complete-cohort report for the frozen duration comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np


def write_duration_report(root: Path, full: dict) -> None:
    """Render every prespecified contrast without choosing a preferred estimator."""
    cells, contrasts = full['cells'], full['contrasts']
    arms = ('fixed', 'short', 'duration')
    allocations = ('uniform', 'evidence_improving')
    estimators = ('classic_moments', 'classic_mc', 'phantom')
    schedule = full['calibration']
    gaps = np.geomspace(schedule['gaps'][0], schedule['gaps'][-1], 10001)
    desired = np.exp(np.interp(np.log(gaps), np.log(schedule['gaps']),
                               np.log(schedule['steps'])))
    bounces = np.ceil(0.03 / desired)
    reduction = 1 - (0.03 / bounces) / desired
    primary = [value for key, value in contrasts.items()
               if key.startswith('duration_minus_') and '_4000000_' in key]
    assert len(primary) == 12
    assert all(lo <= 0 <= hi for lo, hi in [v['paired_bootstrap95'] for v in primary])
    conditioned = [value for key, value in contrasts.items()
                   if key.startswith('phantom_minus_classic_mc_') and key.endswith('_4000000')]
    assert len(conditioned) == 6 and all(v['paired_bootstrap95'][0] > 0 for v in conditioned)
    report = [
        '# Reflection resolution versus trajectory duration', '',
        'Restoring duration did not produce a resolved evidence-RMSE improvement at '
        'the prespecified 4M-call budget: all twelve primary duration-minus-control '
        'intervals include zero. This is not an equivalence result. Absolute movement '
        'in the 5–15 parent-gap bin returns close to the fixed arm, but uses more '
        'gradients. Flat marginals remain too narrow. Within each of the six cells, '
        'phantom conditioning increases RMSE relative to classic MC with a positive '
        'paired 95% interval, while quoting smaller uncertainties.', '',
        'All 180 fresh CPU runs (seeds 30–59 in six arm/allocation cells) and all 360 '
        'snapshots are included. The primary budget is four million logical model calls; '
        'two million is secondary. The analytic reference log evidence is zero. '
        'Results are paired by seed and initial population, not by completed race tree.', '',
        'The fixed arm uses B=10 and h=0.003. The short arm uses the unchanged offline '
        'contour schedule h_old with B=10. The duration arm uses B=ceil(0.03/h_old) and '
        'h=0.03/B, restoring nominal trajectory time 0.03. B/h depend only on the parent '
        'contour and stay fixed throughout the nine-cycle chain. Independent U(0.8,1.2) '
        'step jitter, Gaussian velocities, one ordinary slice per cycle and retained '
        'rejection repeats are unchanged. All arms use B=10 and h=0.003 above gap 49. '
        'The minimum-gap clamp is inherited. No new calibration or retuning occurred; '
        'the duration arm is not a new 70%-acceptance target.', '',
        f'Ceil quantization lowers h by up to {100 * reduction.max():.3f}% on a dense '
        '10,001-point log-gap grid. At gaps 0.1/5/15/40/49, B is 152/37/22/14/13; '
        'including the early region, the allowed B range is 10–152. The nominal '
        'duration is exactly 0.03 by construction, with jittered duration 0.024–0.036.', '',
        '## Evidence estimates', '',
        'Across-run error SD uses ddof=1. Reported SD is the mean internal uncertainty; '
        'coverage uses estimate ±1.959964 SD. Intervals are pointwise paired bootstrap '
        'percentile intervals from 100,000 resamples of the same 30 seeds '
        '(RNG 3122714). They are not multiplicity-adjusted. Each MC estimate uses '
        '4,096 shrinkage draws; classic moments is a distinct deterministic estimator.', '',
    ]
    for budget in (4000000, 2000000):
        report += [
            f'### {budget // 1000000}M calls '
            f'({"primary" if budget == 4000000 else "secondary"})', '',
            '| Allocation | Arm | Estimator | Bias | Error SD | RMSE | RMSE SE | '
            'Mean reported SD | 95% coverage | Mean MCSE |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for allocation in allocations:
            for arm in arms:
                for estimator in estimators:
                    item = cells[f'{arm}_{allocation}_{budget}']['estimators'][estimator]
                    mcse = f"{item['mean_mcse']:.5f}" if 'mean_mcse' in item else '—'
                    report.append(
                        f"| {allocation} | {arm} | {estimator} | {item['bias']:+.5f} | "
                        f"{item['error_sd']:.5f} | {item['rmse']:.5f} | "
                        f"{item['rmse_bootstrap_se']:.5f} | {item['mean_reported_sd']:.5f} | "
                        f"{item['coverage95_count']}/30 | {mcse} |"
                    )
        report += ['', '| Allocation | Contrast | Estimator | RMSE difference | '
                   'Paired bootstrap 95% interval |', '|---|---|---|---:|---|']
        for allocation in allocations:
            for left, right in (('duration', 'short'), ('duration', 'fixed'),
                                ('short', 'fixed')):
                for estimator in estimators:
                    item = contrasts[f'{left}_minus_{right}_{allocation}_{budget}_{estimator}']
                    lo, hi = item['paired_bootstrap95']
                    report.append(
                        f"| {allocation} | {left} − {right} | {estimator} | "
                        f"{item['delta_rmse']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |"
                    )
        report += ['', '| Allocation | Arm | Phantom MC minus classic MC RMSE | '
                   'Paired bootstrap 95% interval |', '|---|---|---:|---|']
        for allocation in allocations:
            for arm in arms:
                item = contrasts[f'phantom_minus_classic_mc_{arm}_{allocation}_{budget}']
                lo, hi = item['paired_bootstrap95']
                report.append(
                    f"| {allocation} | {arm} | {item['delta_rmse']:+.5f} | "
                    f"[{lo:+.5f}, {hi:+.5f}] |"
                )
        report.append('')
    report += [
        '## Posterior recovery and movement at four million calls', '',
        'Posterior quantities use classic weights exclusively. The true SD is '
        '1/sqrt(12)=0.288675 for flat coordinates x1/x16/x17, and approximately '
        '0.05 for x5 (the model retains its original TFP scale precision). '
        'Marginal SDs and weight ESS below are medians over all 30 runs. '
        'Kish weight ESS does not measure chain mixing. Per-run CDF errors and '
        'Wasserstein distances for all four coordinates are in ALL_RECORDS.csv; '
        'all posterior CDFs are plotted.', '',
        '| Allocation | Arm | SD x1 | SD x16 | SD x17 | SD x5 | Kish weight ESS |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for allocation in allocations:
        for arm in arms:
            item = cells[f'{arm}_{allocation}_4000000']
            sd = item['median_marginal_sd']
            report.append(
                f"| {allocation} | {arm} | {sd['x1']:.5f} | {sd['x16']:.5f} | "
                f"{sd['x17']:.5f} | {sd['x5']:.5f} | {item['median_weight_ess']:.1f} |"
            )
    report += [
        '', 'Movement below uses the prespecified 5–15 parent-gap bin. RMS, B, h and '
        'jittered trajectory time T are medians of per-run values. ESJD/gradient '
        'pools squared displacement sums and actual gradient counts across runs '
        'with at least 30 eligible pairs, including all rejected self-transitions. '
        'The numerator and denominator are retained, along with all six gap bins '
        'and endpoint correlations. No normalization by population variance is used.', '',
        '| Allocation | Arm | RMS x1 | RMS x16 | ESJD/grad x1 | ESJD/grad x16 | '
        'Mean B | Mean h | Mean T | Path acceptance |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for allocation in allocations:
        for arm in arms:
            move = cells[f'{arm}_{allocation}_4000000']['movement']['1']
            fields = ('rms_dx1', 'rms_dx16', 'pooled_esjd_x1_per_gradient',
                      'pooled_esjd_x16_per_gradient', 'mean_bounce_count',
                      'mean_realised_h', 'mean_realised_duration', 'pooled_acceptance')
            values = ['—' if move[field] is None else f'{move[field]:.6g}' for field in fields]
            report.append(f'| {allocation} | {arm} | ' + ' | '.join(values) + ' |')
    report += [
        '', '## Actual work and wall time', '',
        'Every gradient counts once in the logical model-call budget. All roots, '
        'including those allocated later, have zero reflections/gradients and one '
        'value call. Snapshots include the complete batch that first crosses the '
        'budget. The core wall time includes compilation and snapshot writes; '
        'analysis time includes full replay and evidence reduction. Hardware work '
        'in masked heterogeneous vmaps can exceed logical lane counts, so logical '
        'work alone does not establish a CPU-throughput gain.', '',
        '| Allocation | Arm | Mean calls (SD) | Mean gradients | Mean value calls | '
        'Max overshoot | Mean epochs | Median core s | Median analysis s | '
        'Max log tail fraction |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for allocation in allocations:
        for arm in arms:
            item = cells[f'{arm}_{allocation}_4000000']
            times = [row for row in full['runtimes']
                     if row['arm'] == arm and row['allocation'] == allocation]
            report.append(
                f"| {allocation} | {arm} | {item['mean_calls']:.0f} ({item['sd_calls']:.0f}) | "
                f"{item['mean_gradients']:.0f} | {item['mean_value_only_calls']:.0f} | "
                f"{item['maximum_overshoot']} | {item['mean_epochs']:.2f} | "
                f"{np.median([r['core_seconds'] for r in times]):.2f} | "
                f"{np.median([r['analysis_seconds'] for r in times]):.2f} | "
                f"{item['max_tail_log_fraction']:.4f} |"
            )
    report += [
        '', 'The conservative remaining-tail diagnostic is retained in log form. '
        'An overflowing linear representation is null with an explicit overflow flag; '
        'it is never clipped into a claimed finite bound.', '',
        '## Validation and provenance', '',
        'Twenty existing tests, eight original duration tests and the additional real '
        'six-new-root regression passed together (29 tests), without weakening '
        'tolerances. Two additional reporting regression tests also pass. All 21 held-out '
        'arm/contour cases pass both single-path and full-chain stationarity checks '
        '(42 laws), strict support/rejection and early-region parity. The independent '
        'heterogeneous-trajectory probe passed reversal and a separate NumPy phase map. '
        'Both allocation smokes pass full replay and evidence reduction. Stationarity '
        'from exact constrained starts does not establish effective mixing.', '',
        'In held-out single-path validation the duration and fixed arms have similar '
        'acceptance at gaps 5, 15 and 40; both accept 0/8192 paths at gap 0.1. '
        'This was descriptive validation, not a reason to change the frozen schedule.', '',
        f"The inherited offline schedule cost {schedule['model_calls']:,} model calls, "
        f"including {schedule['gradient_calls']:,} gradients. That cost was paid in the "
        'earlier experiment and is not newly incurred or hidden in these budgets. '
        'Diagnostic replay and evidence reduction are separately reported overhead.', '',
        'The initial accounting attempt incorrectly assigned reflection work to six '
        'new roots; the corrected mask leaves sampling and total likelihood calls '
        'unchanged. A subsequent early smoke exposed linear tail-bound JSON overflow. '
        'Both failed attempts, source freezes and raw records are preserved. '
        'Final-source arrays match the preceding corrected attempt exactly, and '
        'evaluation began only after independent review of the final gates.', '',
        'After all runs finished, aggregation exposed likelihood-block counts shadowing '
        'bounce counts in six derived movement B/T fields. Those six fields were '
        'corrected from unchanged raw B and per-cycle steps. Original derived outputs '
        'are preserved; DIAGNOSTIC_CORRECTION.json binds every before/after checksum '
        'and correction source. Sampling, replay, gradient counts, ESJD, posterior '
        'metrics, evidence draws and every uncertainty contrast are unchanged. '
        'There were no sampling, evidence or replay reruns for this correction.', '',
        f"Executed source: `{full['audit']['source_commit']}`. Protocol SHA256: "
        f"`{full['audit']['protocol_sha256']}`. Source-freeze SHA256: "
        f"`{full['audit']['source_freeze_sha256']}`. Validation SHA256: "
        f"`{full['audit']['validation_sha256']}`.", '',
        'Full resumable states, raw arrays/keys, evidence inputs and draws, replay '
        'diagnostics, manifests, progress, checksums, failed attempts and exact '
        'coordinator audit sources/receipts remain in the experiment output directory. '
        'SUMMARY.json contains all point estimates, coverage variants, standardized '
        'error diagnostics and paired intervals. No manuscript or production code '
        'was changed.', '',
    ]
    audit_path = root / 'coordinator-audits/COHORT_AUDIT.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get('passed'):
            report += ['Independent full-cohort coordinator audit passed; see '
                       '`coordinator-audits/COHORT_AUDIT.json` and its exact source.', '']
    path = root / 'REPORT.md'
    path.write_text('\n'.join(report))
    path.with_suffix('.md.sha256').write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n'
    )
