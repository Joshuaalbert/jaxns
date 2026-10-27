# Reflection resolution versus trajectory duration

Restoring duration did not produce a resolved evidence-RMSE improvement at the prespecified 4M-call budget: all twelve primary duration-minus-control intervals include zero. This is not an equivalence result. Absolute movement in the 5–15 parent-gap bin returns close to the fixed arm, but uses more gradients. Flat marginals remain too narrow. Within each of the six cells, phantom conditioning increases RMSE relative to classic MC with a positive paired 95% interval, while quoting smaller uncertainties.

All 180 fresh CPU runs (seeds 30–59 in six arm/allocation cells) and all 360 snapshots are included. The primary budget is four million logical model calls; two million is secondary. The analytic reference log evidence is zero. Results are paired by seed and initial population, not by completed race tree.

The fixed arm uses B=10 and h=0.003. The short arm uses the unchanged offline contour schedule h_old with B=10. The duration arm uses B=ceil(0.03/h_old) and h=0.03/B, restoring nominal trajectory time 0.03. B/h depend only on the parent contour and stay fixed throughout the nine-cycle chain. Independent U(0.8,1.2) step jitter, Gaussian velocities, one ordinary slice per cycle and retained rejection repeats are unchanged. All arms use B=10 and h=0.003 above gap 49. The minimum-gap clamp is inherited. No new calibration or retuning occurred; the duration arm is not a new 70%-acceptance target.

Ceil quantization lowers h by up to 7.128% on a dense 10,001-point log-gap grid. At gaps 0.1/5/15/40/49, B is 152/37/22/14/13; including the early region, the allowed B range is 10–152. The nominal duration is exactly 0.03 by construction, with jittered duration 0.024–0.036.

## Evidence estimates

Across-run error SD uses ddof=1. Reported SD is the mean internal uncertainty; coverage uses estimate ±1.959964 SD. Intervals are pointwise paired bootstrap percentile intervals from 100,000 resamples of the same 30 seeds (RNG 3122714). They are not multiplicity-adjusted. Each MC estimate uses 4,096 shrinkage draws; classic moments is a distinct deterministic estimator.

### 4M calls (primary)

| Allocation | Arm | Estimator | Bias | Error SD | RMSE | RMSE SE | Mean reported SD | 95% coverage | Mean MCSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | classic_moments | -0.35442 | 0.67172 | 0.74952 | 0.06911 | 0.27327 | 14/30 | — |
| uniform | fixed | classic_mc | -0.35437 | 0.67122 | 0.74906 | 0.06904 | 0.27429 | 14/30 | 0.00429 |
| uniform | fixed | phantom | -0.39254 | 0.84587 | 0.91964 | 0.09713 | 0.16049 | 7/30 | 0.00251 |
| uniform | short | classic_moments | -0.32110 | 0.70264 | 0.76181 | 0.09048 | 0.27292 | 15/30 | — |
| uniform | short | classic_mc | -0.32254 | 0.70208 | 0.76191 | 0.09069 | 0.27321 | 14/30 | 0.00427 |
| uniform | short | phantom | -0.46238 | 0.82861 | 0.93676 | 0.09765 | 0.16616 | 10/30 | 0.00260 |
| uniform | duration | classic_moments | -0.26369 | 0.74028 | 0.77413 | 0.08240 | 0.32789 | 16/30 | — |
| uniform | duration | classic_mc | -0.26443 | 0.74193 | 0.77591 | 0.08252 | 0.32912 | 16/30 | 0.00514 |
| uniform | duration | phantom | -0.42688 | 0.91704 | 0.99758 | 0.09745 | 0.19340 | 7/30 | 0.00302 |
| evidence_improving | fixed | classic_moments | -0.54828 | 0.67290 | 0.85925 | 0.08064 | 0.24705 | 9/30 | — |
| evidence_improving | fixed | classic_mc | -0.54784 | 0.67342 | 0.85936 | 0.08073 | 0.24820 | 9/30 | 0.00388 |
| evidence_improving | fixed | phantom | -0.72346 | 0.82311 | 1.08551 | 0.09940 | 0.14482 | 3/30 | 0.00226 |
| evidence_improving | short | classic_moments | -0.54144 | 0.72831 | 0.89773 | 0.09696 | 0.24648 | 13/30 | — |
| evidence_improving | short | classic_mc | -0.54085 | 0.72897 | 0.89789 | 0.09711 | 0.24627 | 14/30 | 0.00385 |
| evidence_improving | short | phantom | -0.75394 | 0.91078 | 1.17060 | 0.13020 | 0.14963 | 6/30 | 0.00234 |
| evidence_improving | duration | classic_moments | -0.54432 | 0.67768 | 0.86037 | 0.07573 | 0.26713 | 8/30 | — |
| evidence_improving | duration | classic_mc | -0.54486 | 0.67766 | 0.86069 | 0.07580 | 0.26704 | 8/30 | 0.00417 |
| evidence_improving | duration | phantom | -0.70707 | 0.88708 | 1.12278 | 0.10994 | 0.15701 | 4/30 | 0.00245 |

| Allocation | Contrast | Estimator | RMSE difference | Paired bootstrap 95% interval |
|---|---|---|---:|---|
| uniform | duration − short | classic_moments | +0.01233 | [-0.12527, +0.15522] |
| uniform | duration − short | classic_mc | +0.01400 | [-0.12461, +0.15808] |
| uniform | duration − short | phantom | +0.06082 | [-0.08928, +0.20012] |
| uniform | duration − fixed | classic_moments | +0.02461 | [-0.08016, +0.12338] |
| uniform | duration − fixed | classic_mc | +0.02685 | [-0.07766, +0.12509] |
| uniform | duration − fixed | phantom | +0.07794 | [-0.01935, +0.17507] |
| uniform | short − fixed | classic_moments | +0.01228 | [-0.14019, +0.15312] |
| uniform | short − fixed | classic_mc | +0.01285 | [-0.13976, +0.15360] |
| uniform | short − fixed | phantom | +0.01712 | [-0.13066, +0.16707] |
| evidence_improving | duration − short | classic_moments | -0.03736 | [-0.14520, +0.08293] |
| evidence_improving | duration − short | classic_mc | -0.03719 | [-0.14565, +0.08375] |
| evidence_improving | duration − short | phantom | -0.04782 | [-0.19123, +0.09874] |
| evidence_improving | duration − fixed | classic_moments | +0.00112 | [-0.12909, +0.12689] |
| evidence_improving | duration − fixed | classic_mc | +0.00133 | [-0.12884, +0.12711] |
| evidence_improving | duration − fixed | phantom | +0.03727 | [-0.06865, +0.13850] |
| evidence_improving | short − fixed | classic_moments | +0.03848 | [-0.13839, +0.20293] |
| evidence_improving | short − fixed | classic_mc | +0.03852 | [-0.13855, +0.20303] |
| evidence_improving | short − fixed | phantom | +0.08509 | [-0.09456, +0.26151] |

| Allocation | Arm | Phantom MC minus classic MC RMSE | Paired bootstrap 95% interval |
|---|---|---:|---|
| uniform | fixed | +0.17057 | [+0.07191, +0.26630] |
| uniform | short | +0.17484 | [+0.08031, +0.28092] |
| uniform | duration | +0.22166 | [+0.10639, +0.32218] |
| evidence_improving | fixed | +0.22614 | [+0.10712, +0.33466] |
| evidence_improving | short | +0.27271 | [+0.15334, +0.39019] |
| evidence_improving | duration | +0.26209 | [+0.15001, +0.37031] |

### 2M calls (secondary)

| Allocation | Arm | Estimator | Bias | Error SD | RMSE | RMSE SE | Mean reported SD | 95% coverage | Mean MCSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | classic_moments | -0.36440 | 0.77653 | 0.84598 | 0.07361 | 0.37490 | 16/30 | — |
| uniform | fixed | classic_mc | -0.36471 | 0.77595 | 0.84560 | 0.07401 | 0.37646 | 16/30 | 0.00588 |
| uniform | fixed | phantom | -0.44121 | 0.99802 | 1.07588 | 0.11582 | 0.21894 | 7/30 | 0.00342 |
| uniform | short | classic_moments | -0.38293 | 0.73456 | 0.81745 | 0.07773 | 0.37705 | 16/30 | — |
| uniform | short | classic_mc | -0.38393 | 0.73591 | 0.81910 | 0.07794 | 0.37767 | 16/30 | 0.00590 |
| uniform | short | phantom | -0.56738 | 0.94973 | 1.09263 | 0.10817 | 0.22874 | 10/30 | 0.00357 |
| uniform | duration | classic_moments | -0.27459 | 0.89073 | 0.91780 | 0.09336 | 0.43572 | 19/30 | — |
| uniform | duration | classic_mc | -0.27624 | 0.89317 | 0.92058 | 0.09382 | 0.43973 | 19/30 | 0.00687 |
| uniform | duration | phantom | -0.51192 | 1.09892 | 1.19559 | 0.10894 | 0.25501 | 8/30 | 0.00399 |
| evidence_improving | fixed | classic_moments | -0.57561 | 0.83736 | 1.00456 | 0.10235 | 0.35164 | 14/30 | — |
| evidence_improving | fixed | classic_mc | -0.57500 | 0.83635 | 1.00339 | 0.10235 | 0.35266 | 14/30 | 0.00551 |
| evidence_improving | fixed | phantom | -0.84855 | 0.97251 | 1.27840 | 0.11970 | 0.20467 | 2/30 | 0.00320 |
| evidence_improving | short | classic_moments | -0.78910 | 0.92227 | 1.20205 | 0.12547 | 0.35264 | 12/30 | — |
| evidence_improving | short | classic_mc | -0.78618 | 0.92182 | 1.19979 | 0.12473 | 0.35408 | 12/30 | 0.00553 |
| evidence_improving | short | phantom | -0.96602 | 1.09803 | 1.44868 | 0.16233 | 0.21244 | 6/30 | 0.00332 |
| evidence_improving | duration | classic_moments | -0.62482 | 0.85949 | 1.05095 | 0.09037 | 0.38372 | 11/30 | — |
| evidence_improving | duration | classic_mc | -0.62570 | 0.86022 | 1.05205 | 0.09059 | 0.38526 | 11/30 | 0.00602 |
| evidence_improving | duration | phantom | -0.86036 | 1.06533 | 1.35548 | 0.13762 | 0.22341 | 3/30 | 0.00349 |

| Allocation | Contrast | Estimator | RMSE difference | Paired bootstrap 95% interval |
|---|---|---|---:|---|
| uniform | duration − short | classic_moments | +0.10035 | [-0.03785, +0.24108] |
| uniform | duration − short | classic_mc | +0.10149 | [-0.03783, +0.24333] |
| uniform | duration − short | phantom | +0.10296 | [-0.03518, +0.24084] |
| uniform | duration − fixed | classic_moments | +0.07182 | [-0.07795, +0.20016] |
| uniform | duration − fixed | classic_mc | +0.07498 | [-0.07497, +0.20360] |
| uniform | duration − fixed | phantom | +0.11971 | [+0.00374, +0.25020] |
| uniform | short − fixed | classic_moments | -0.02853 | [-0.15770, +0.10187] |
| uniform | short − fixed | classic_mc | -0.02650 | [-0.15578, +0.10380] |
| uniform | short − fixed | phantom | +0.01675 | [-0.15241, +0.19838] |
| evidence_improving | duration − short | classic_moments | -0.15109 | [-0.32020, +0.03695] |
| evidence_improving | duration − short | classic_mc | -0.14774 | [-0.31657, +0.03910] |
| evidence_improving | duration − short | phantom | -0.09321 | [-0.32089, +0.11011] |
| evidence_improving | duration − fixed | classic_moments | +0.04639 | [-0.12050, +0.21371] |
| evidence_improving | duration − fixed | classic_mc | +0.04867 | [-0.11834, +0.21659] |
| evidence_improving | duration − fixed | phantom | +0.07708 | [-0.07047, +0.23302] |
| evidence_improving | short − fixed | classic_moments | +0.19749 | [-0.05962, +0.43228] |
| evidence_improving | short − fixed | classic_mc | +0.19640 | [-0.05949, +0.42990] |
| evidence_improving | short − fixed | phantom | +0.17029 | [-0.04627, +0.37409] |

| Allocation | Arm | Phantom MC minus classic MC RMSE | Paired bootstrap 95% interval |
|---|---|---:|---|
| uniform | fixed | +0.23028 | [+0.07873, +0.36789] |
| uniform | short | +0.27353 | [+0.15777, +0.39732] |
| uniform | duration | +0.27501 | [+0.11539, +0.42340] |
| evidence_improving | fixed | +0.27501 | [+0.08091, +0.46342] |
| evidence_improving | short | +0.24889 | [+0.07323, +0.41525] |
| evidence_improving | duration | +0.30343 | [+0.10497, +0.49101] |

## Posterior recovery and movement at four million calls

Posterior quantities use classic weights exclusively. The true SD is 1/sqrt(12)=0.288675 for flat coordinates x1/x16/x17, and approximately 0.05 for x5 (the model retains its original TFP scale precision). Marginal SDs and weight ESS below are medians over all 30 runs. Kish weight ESS does not measure chain mixing. Per-run CDF errors and Wasserstein distances for all four coordinates are in ALL_RECORDS.csv; all posterior CDFs are plotted.

| Allocation | Arm | SD x1 | SD x16 | SD x17 | SD x5 | Kish weight ESS |
|---|---|---:|---:|---:|---:|---:|
| uniform | fixed | 0.13400 | 0.19572 | 0.21186 | 0.04965 | 3317.4 |
| uniform | short | 0.12613 | 0.15417 | 0.18842 | 0.04766 | 3360.0 |
| uniform | duration | 0.14218 | 0.17337 | 0.19594 | 0.04954 | 2843.9 |
| evidence_improving | fixed | 0.12261 | 0.16762 | 0.16618 | 0.04905 | 1984.0 |
| evidence_improving | short | 0.09712 | 0.13109 | 0.12257 | 0.04676 | 1971.3 |
| evidence_improving | duration | 0.10774 | 0.14505 | 0.16598 | 0.04962 | 1698.6 |

Movement below uses the prespecified 5–15 parent-gap bin. RMS, B, h and jittered trajectory time T are medians of per-run values. ESJD/gradient pools squared displacement sums and actual gradient counts across runs with at least 30 eligible pairs, including all rejected self-transitions. The numerator and denominator are retained, along with all six gap bins and endpoint correlations. No normalization by population variance is used.

| Allocation | Arm | RMS x1 | RMS x16 | ESJD/grad x1 | ESJD/grad x16 | Mean B | Mean h | Mean T | Path acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 0.031236 | 0.0466943 | 1.06628e-05 | 2.38883e-05 | 10 | 0.00300001 | 0.0300001 | 0.289596 |
| uniform | short | 0.0195435 | 0.0278182 | 4.27847e-06 | 8.47763e-06 | 10 | 0.00110388 | 0.0110388 | 0.701007 |
| uniform | duration | 0.0330929 | 0.0471801 | 4.37276e-06 | 8.6341e-06 | 28.5415 | 0.00107753 | 0.0300097 | 0.297289 |
| evidence_improving | fixed | 0.0321958 | 0.048528 | 1.13113e-05 | 2.61544e-05 | 10 | 0.00300007 | 0.0300007 | 0.318112 |
| evidence_improving | short | 0.0193647 | 0.0291683 | 4.29971e-06 | 9.43742e-06 | 10 | 0.00117207 | 0.0117207 | 0.696001 |
| evidence_improving | duration | 0.0343804 | 0.0488008 | 4.86126e-06 | 9.94917e-06 | 26.6977 | 0.00114723 | 0.0299938 | 0.324328 |

## Actual work and wall time

Every gradient counts once in the logical model-call budget. All roots, including those allocated later, have zero reflections/gradients and one value call. Snapshots include the complete batch that first crosses the budget. The core wall time includes compilation and snapshot writes; analysis time includes full replay and evidence reduction. Hardware work in masked heterogeneous vmaps can exceed logical lane counts, so logical work alone does not establish a CPU-throughput gain.

| Allocation | Arm | Mean calls (SD) | Mean gradients | Mean value calls | Max overshoot | Mean epochs | Median core s | Median analysis s | Max log tail fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 4021843 (11947) | 1628106 | 2393737 | 37048 | 2.00 | 66.72 | 30.48 | -9.8624 |
| uniform | short | 4017239 (10581) | 1627494 | 2389745 | 35764 | 2.00 | 67.22 | 30.79 | -9.5710 |
| uniform | duration | 4041642 (25521) | 2299644 | 1741998 | 89890 | 1.00 | 62.52 | 25.96 | -9.3537 |
| evidence_improving | fixed | 4013359 (8929) | 1707279 | 2306080 | 32959 | 1.93 | 87.46 | 35.25 | -5.5900 |
| evidence_improving | short | 4015418 (9777) | 1703574 | 2311844 | 31234 | 1.80 | 87.17 | 34.65 | -5.0220 |
| evidence_improving | duration | 4014272 (9493) | 2041161 | 1973111 | 32759 | 1.27 | 76.41 | 29.27 | -5.1190 |

The conservative remaining-tail diagnostic is retained in log form. An overflowing linear representation is null with an explicit overflow flag; it is never clipped into a claimed finite bound.

## Validation and provenance

Twenty existing tests, eight original duration tests and the additional real six-new-root regression passed together (29 tests), without weakening tolerances. Two additional reporting regression tests also pass. All 21 held-out arm/contour cases pass both single-path and full-chain stationarity checks (42 laws), strict support/rejection and early-region parity. The independent heterogeneous-trajectory probe passed reversal and a separate NumPy phase map. Both allocation smokes pass full replay and evidence reduction. Stationarity from exact constrained starts does not establish effective mixing.

In held-out single-path validation the duration and fixed arms have similar acceptance at gaps 5, 15 and 40; both accept 0/8192 paths at gap 0.1. This was descriptive validation, not a reason to change the frozen schedule.

The inherited offline schedule cost 7,208,960 model calls, including 6,553,600 gradients. That cost was paid in the earlier experiment and is not newly incurred or hidden in these budgets. Diagnostic replay and evidence reduction are separately reported overhead.

The initial accounting attempt incorrectly assigned reflection work to six new roots; the corrected mask leaves sampling and total likelihood calls unchanged. A subsequent early smoke exposed linear tail-bound JSON overflow. Both failed attempts, source freezes and raw records are preserved. Final-source arrays match the preceding corrected attempt exactly, and evaluation began only after independent review of the final gates.

After all runs finished, aggregation exposed likelihood-block counts shadowing bounce counts in six derived movement B/T fields. Those six fields were corrected from unchanged raw B and per-cycle steps. Original derived outputs are preserved; DIAGNOSTIC_CORRECTION.json binds every before/after checksum and correction source. Sampling, replay, gradient counts, ESJD, posterior metrics, evidence draws and every uncertainty contrast are unchanged. There were no sampling, evidence or replay reruns for this correction.

Executed source: `724d5e9a94d2db417573cd8cc473224e82a60a13`. Protocol SHA256: `44154ae79a7d91408c43fcb81f7a8edcf2c264c95205aa22f47d0723bfa5782b`. Source-freeze SHA256: `5714ed9a45e0e6fc2ed189ccfdf5f5686b2c74310aac698a6d88b0b80d48f3fc`. Validation SHA256: `e53f9b386deec41da78bb7237eee02b16d561fd4202db90e68eb61f58d47b3dc`.

Full resumable states, raw arrays/keys, evidence inputs and draws, replay diagnostics, manifests, progress, checksums, failed attempts and exact coordinator audit sources/receipts remain in the experiment output directory. SUMMARY.json contains all point estimates, coverage variants, standardized error diagnostics and paired intervals. No manuscript or production code was changed.

Independent full-cohort coordinator audit passed; see `coordinator-audits/COHORT_AUDIT.json` and its exact source.
