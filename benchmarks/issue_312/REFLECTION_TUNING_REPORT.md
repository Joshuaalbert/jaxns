# Frozen reflection step tuning: full audited CPU cohort

All 120 runs and 240 snapshots are included. Tuning to 70% path acceptance did not improve evidence RMSE at the prespecified 4M primary budget. Both controls and tuned arms are fresh ARM runs with matched prior populations.

The contour-dependent schedule was learned offline from independent exact constrained pilot draws and frozen before evaluation. It depends only on the parent contour, with common fixed h=0.003 above gap 49; it is not online adaptation. B=10 reflections and C=9 cycles stayed fixed. Reducing h therefore shortened the nominal trajectory time B*h rather than compensating with more reflections. Gaussian velocity refresh, independent step jitter and rejection repeats are unchanged. No evaluation-driven retuning or trajectory-length matching was performed.

## Evidence at the primary four-million-call budget

| Allocation | Arm | Estimator | Bias | RMSE | RMSE bootstrap SE | Mean SD | 95% coverage |
|---|---|---|---:|---:|---:|---:|---:|
| uniform | fixed | classic_moments | -0.11525 | 0.64302 | 0.09788 | 0.27193 | 20/30 |
| uniform | fixed | classic_mc | -0.11579 | 0.64326 | 0.09814 | 0.27190 | 20/30 |
| uniform | fixed | phantom | -0.15148 | 0.78097 | 0.11349 | 0.15998 | 9/30 |
| uniform | tuned | classic_moments | -0.08115 | 0.72223 | 0.07332 | 0.27136 | 16/30 |
| uniform | tuned | classic_mc | -0.08075 | 0.72246 | 0.07342 | 0.27068 | 16/30 |
| uniform | tuned | phantom | -0.21475 | 0.84988 | 0.10114 | 0.16452 | 7/30 |
| evidence_improving | fixed | classic_moments | -0.17406 | 0.70114 | 0.06681 | 0.24348 | 16/30 |
| evidence_improving | fixed | classic_mc | -0.17465 | 0.70152 | 0.06694 | 0.24361 | 15/30 |
| evidence_improving | fixed | phantom | -0.28134 | 0.90413 | 0.09728 | 0.14297 | 7/30 |
| evidence_improving | tuned | classic_moments | -0.23985 | 0.84558 | 0.10235 | 0.24389 | 12/30 |
| evidence_improving | tuned | classic_mc | -0.24055 | 0.84644 | 0.10229 | 0.24400 | 12/30 |
| evidence_improving | tuned | phantom | -0.44066 | 1.04782 | 0.12539 | 0.14763 | 8/30 |

## Prespecified paired RMSE contrasts

| Allocation | Estimator | Tuned minus fixed | Paired bootstrap 95% interval |
|---|---|---:|---|
| uniform | classic_moments | +0.07921 | [-0.08872, +0.24566] |
| uniform | classic_mc | +0.07920 | [-0.08942, +0.24610] |
| uniform | phantom | +0.06891 | [-0.03715, +0.19655] |
| evidence_improving | classic_moments | +0.14444 | [+0.00396, +0.28608] |
| evidence_improving | classic_mc | +0.14492 | [+0.00449, +0.28621] |
| evidence_improving | phantom | +0.14369 | [-0.01605, +0.29993] |

## Secondary two-million-call snapshot

| Allocation | Arm | Classic moments RMSE | Classic MC RMSE | Phantom MC RMSE |
|---|---|---:|---:|---:|
| uniform | fixed | 0.87040 | 0.87102 | 0.92141 |
| uniform | tuned | 0.75971 | 0.75959 | 0.96596 |
| evidence_improving | fixed | 0.89636 | 0.89456 | 1.10203 |
| evidence_improving | tuned | 1.02688 | 1.02729 | 1.27011 |

## Posterior widths and actual movement at four million calls

Medians over all 30 runs; movement uses the prespecified 5–15 parent-gap bin. The true flat-coordinate SD is 0.288675 and x5 SD is 0.05. Every chain uses 90 gradients; displacement per gradient is RMS/90.

| Allocation | Arm | SD x1 | SD x16 | SD x17 | SD x5 | RMS move x1 | RMS move x16 | Path acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 0.15410 | 0.18826 | 0.21645 | 0.04927 | 0.02962 | 0.04658 | 28.903% |
| uniform | tuned | 0.13958 | 0.16291 | 0.17697 | 0.04885 | 0.01921 | 0.02770 | 70.117% |
| evidence_improving | fixed | 0.10988 | 0.14756 | 0.16063 | 0.04929 | 0.03121 | 0.04827 | 32.023% |
| evidence_improving | tuned | 0.09823 | 0.12767 | 0.10401 | 0.04597 | 0.02047 | 0.02931 | 69.912% |

## Matched work and execution cost at four million calls

| Allocation | Arm | Mean calls | Call SD | Mean gradients | Mean completed epochs | Median core seconds | Median analysis seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 4,021,137 | 12,009 | 1,628,700 | 2.000 | 66.94 | 30.49 |
| uniform | tuned | 4,022,586 | 11,300 | 1,626,525 | 2.000 | 66.87 | 30.71 |
| evidence_improving | fixed | 4,012,580 | 9,994 | 1,705,503 | 2.033 | 86.94 | 35.35 |
| evidence_improving | tuned | 4,014,040 | 9,804 | 1,707,702 | 2.133 | 87.57 | 35.69 |

## Interpretation and limitations

The frozen 70% target did not improve the primary 4M result in this cohort. Classic and phantom RMSE point estimates are higher under both allocations. Only the evidence-allocation classic contrast excludes zero in its pointwise 95% paired interval; these intervals are not adjusted for multiple comparisons. At 2M, tuned uniform has a lower classic point RMSE, so the primary endpoint matters. The complete two-budget comparison is retained above.

The acceptance target was selected before evaluation. Stationarity and higher acceptance do not establish mixing. Compare absolute displacement, posterior CDF errors, marginal widths and evidence RMSE together; a collapsed seed cloud can have deceptively low endpoint correlation. Kish ESS here is weight ESS, not a correction for Markov-chain dependence.

In the 5–15 gap bin, tuning reduces median absolute x1 movement by about 35% and inert-coordinate movement by about 40%, despite raising path acceptance to 70%. The held-out exact ensembles show the same shorter-movement effect. Posterior widths generally narrow and marginal CDF errors worsen; higher acceptance has not repaired population coverage. This supports a movement-versus-acceptance tradeoff for fixed B=10 and C=9, not a universal optimal acceptance target.

At 4M, phantom conditioning increases RMSE relative to classic MC in all four cells, with positive pointwise paired intervals. Mean reported uncertainties shrink while coverage remains poor. The full contrasts and per-seed MCSE are in SUMMARY.json; the largest evidence-mean MCSE over both budgets is 0.00673 (classic) or 0.00390 (phantom).

Classic moments, classic Monte Carlo and phantom Monte Carlo are distinct estimator conventions. Arms share seeds and initial populations but generate different trees; classic/phantom contrasts within an arm use the same tree. Phantoms receive no posterior weights or seed access. The original eight-cycle prefix and cluster-weight evidence law are unchanged. See SUMMARY.json for both budgets, all uncertainty diagnostics, per-seed errors, paired phantom-minus-classic contrasts and MC errors.

## Work and provenance

Each sampling process and reduction was pinned to one CPU with CPU-only JAX. Logical model-call budgets include each gradient evaluation once; gradient cost is also reported separately. Diagnostic replay, evidence reduction, normalization probes and calibration are overhead, not hidden in these budgets. Each independent seed pays its compilation cost; these timings do not measure production steady-state throughput.

Calibration used 7,208,960 model calls, including 6,553,600 gradient calls, and took 12.96 seconds. Amortized over the 60 tuned runs that is 120,149 extra model calls/run (3.00% of a 4M nominal budget). The shared control comparison still uses identical evaluation budgets; calibration is disclosed separately.

Raw state.pkl files, complete arrays, evidence inputs/draws, replay diagnostics, manifests/checksums and source archive remain under this output directory. All three independent coordinator audits (pilot, held-out and complete cohort) passed; their sources and receipts are retained, including independent agreement on every reported paired interval. The failed serializer attempt is preserved. No production code, manuscript or GitHub artifact was changed.

## Reproducibility and review files

Prespecified protocol: [REFLECTION_TUNING_PROTOCOL.md](REFLECTION_TUNING_PROTOCOL.md).
The executed sampling/reduction source is commit
`da352c53ab98ecee12932abdbb2dd83ed670a689`, based on
`4e9cc2927b2d4ab4c50529af00201a4fd9960fe8`.
The frozen schedule SHA256 is
`21242cb5554c0b7c23c4750d1c8aa579e3d9856ff670a0f9c18133b1771a28d6`.

Complete output on dorrie:
`/largedata/albert/jaxns-reflection-tuning-20261027/`.
Read `HANDOFF_FINAL.json` for the incremental bundle and archive checksums.
`compact-records-20261027.tar.gz` contains all 120 run records, all 240 evidence-draw
archives, chain diagnostics and the complete summary; full states, classic sample
arrays and evidence inputs remain on dorrie with checksums.
`review-provenance-20261027.tar.gz` contains protocol, frozen schedule, source,
validation receipts, test logs, coordinator audits and the three PDF/PNG figures.
The modest [per-snapshot CSV](reflection_tuning_records.csv) is also versioned;
large numerical arrays are not in Git.

The reporting entrypoint is `summarise_reflection_tuning.py`. Its `--report-only`
option renders prose/tables from the audited `SUMMARY.json` without rerunning
sampling, reduction, bootstrap checks or figures. All 24 relevant existing,
new and reviewer checks passed before final reporting; Ruff and Flake8 passed.
