# Online reflection resolution and duration tuning

Online learning did not resolve an evidence-RMSE improvement over the fixed control at 4M calls under either allocation. The independently trained frozen policy with evidence allocation reduced classic-MC RMSE from 1.00179 to 0.65577: difference -0.34602, paired 95% interval [-0.54913, -0.14091]. Its classic-moments and phantom-MC contrasts agree in direction and exclude zero; these are correlated estimators on the same cohort, not independent replications. No frozen-policy advantage was resolved under uniform allocation.

At the prespecified 4M-call endpoint, 3 of the twelve primary arm-minus-fixed contrasts have a negative paired 95% RMSE interval, and 0 have a positive interval. The other intervals include zero. These are pointwise intervals, without multiplicity adjustment; inclusion of zero does not establish equivalence. All contrasts are reported below.

The complete cohort contains 180 CPU runs: three arms, two allocations and the same 30 fresh seeds 60–89. All 360 snapshots are retained. Four million logical model calls is primary; two million is secondary. The analytic reference log evidence is zero. Arms are paired by seed and initial population, not by completed race tree.

The fixed arm uses h=0.003, B=10 and nominal T=0.03. Online learns h and T from whole-chain squared displacement summed over all 18 cube coordinates divided by actual total model work. Nine cycles, one isotropic slice per cycle, independent U(0.8,1.2) jitter, Gaussian velocities, strict endpoint rejection and retained self-transitions remain unchanged. No replay or scouting model calls feed the controller. Eight phantoms condition evidence only: they never supply seeds, posterior weights or controller feedback.

Online and frozen both use 90% central settings and four 2.5% perturbations. Frozen lookup uses actual parent likelihood and a table averaged equally in log parameters across four independent training runs per allocation; it never reconstructs compression from its evaluation population. The four training seeds are paired across allocations and separate from evaluation; they are independent within each allocation, not eight mutually independent training seeds. Online uses only the classic tree available before dispatch. The information clock sums log1p(1/K_parent) over new completed nonroot chains. Revisits supply new observations, while deepest frontier compression is reported separately.

Desired h is bounded to [0.001,0.03] and requested T to [0.003,0.1]. B=ceil(T/h), with the prespecified near-integer rounding guard; effective h=T/B precedes jitter. Effective and jittered h can be below the desired lower bound. The old B≤256 cap is inactive because the new domain has B≤100. This domain was fixed before training and evaluation following an independent numerical range failure; no integrator or tolerance was changed.

## Evidence estimates

Error SD uses ddof=1. Gaussian coverage uses estimate ±1.959964 reported SD; MC coverage uses the empirical 2.5%–97.5% draw interval. Each MC estimator uses 4,096 shrinkage draws. Classic moments, classic MC and phantom MC are distinct estimators. Bootstrap SEs and intervals use the same 100,000 paired seed resamples throughout, RNG 3122903. Per-seed errors, reported SDs, MCSEs and standardized-error diagnostics are retained in SUMMARY.json and records.

### 4M calls

| Allocation | Arm | Estimator | Bias | Error SD | RMSE | SE | Reported SD | Gaussian coverage | MC coverage | Mean MCSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | classic_moments | -0.07925 | 0.60350 | 0.59863 | 0.06485 | 0.27142 | 19/30 | — | — |
| uniform | fixed | classic_mc | -0.07932 | 0.60184 | 0.59701 | 0.06492 | 0.27131 | 19/30 | 19/30 | 0.00424 |
| uniform | fixed | phantom | -0.19339 | 0.71225 | 0.72649 | 0.07526 | 0.15927 | 9/30 | 9/30 | 0.00249 |
| uniform | online | classic_moments | -0.06192 | 0.75149 | 0.74145 | 0.10933 | 0.27904 | 17/30 | — | — |
| uniform | online | classic_mc | -0.06201 | 0.75111 | 0.74108 | 0.10941 | 0.27847 | 17/30 | 17/30 | 0.00435 |
| uniform | online | phantom | -0.15901 | 0.87407 | 0.87397 | 0.12148 | 0.16320 | 8/30 | 8/30 | 0.00255 |
| uniform | frozen | classic_moments | -0.21519 | 0.62049 | 0.64690 | 0.07178 | 0.27429 | 19/30 | — | — |
| uniform | frozen | classic_mc | -0.21603 | 0.62164 | 0.64824 | 0.07198 | 0.27579 | 19/30 | 19/30 | 0.00431 |
| uniform | frozen | phantom | -0.38302 | 0.68398 | 0.77391 | 0.07271 | 0.16055 | 9/30 | 10/30 | 0.00251 |
| evidence_improving | fixed | classic_moments | -0.50057 | 0.88392 | 1.00291 | 0.10728 | 0.24652 | 9/30 | — | — |
| evidence_improving | fixed | classic_mc | -0.49954 | 0.88320 | 1.00179 | 0.10712 | 0.24763 | 9/30 | 9/30 | 0.00387 |
| evidence_improving | fixed | phantom | -0.70518 | 1.06469 | 1.26216 | 0.12070 | 0.14470 | 4/30 | 4/30 | 0.00226 |
| evidence_improving | online | classic_moments | -0.29432 | 0.97487 | 1.00265 | 0.12393 | 0.25054 | 10/30 | — | — |
| evidence_improving | online | classic_mc | -0.29403 | 0.97597 | 1.00361 | 0.12387 | 0.25109 | 10/30 | 10/30 | 0.00392 |
| evidence_improving | online | phantom | -0.46760 | 1.18133 | 1.25207 | 0.14281 | 0.14686 | 4/30 | 4/30 | 0.00229 |
| evidence_improving | frozen | classic_moments | -0.22499 | 0.62566 | 0.65500 | 0.11132 | 0.25324 | 20/30 | — | — |
| evidence_improving | frozen | classic_mc | -0.22560 | 0.62627 | 0.65577 | 0.11130 | 0.25365 | 20/30 | 20/30 | 0.00396 |
| evidence_improving | frozen | phantom | -0.26864 | 0.75196 | 0.78661 | 0.12628 | 0.14635 | 12/30 | 12/30 | 0.00229 |

| Allocation | Contrast | Estimator | RMSE difference | Paired 95% CI |
|---|---|---|---:|---|
| uniform | online − fixed | classic_moments | +0.14282 | [-0.11185, +0.40740] |
| uniform | online − fixed | classic_mc | +0.14407 | [-0.11047, +0.40888] |
| uniform | online − fixed | phantom | +0.14747 | [-0.13526, +0.44886] |
| uniform | frozen − fixed | classic_moments | +0.04827 | [-0.12742, +0.22142] |
| uniform | frozen − fixed | classic_mc | +0.05123 | [-0.12447, +0.22472] |
| uniform | frozen − fixed | phantom | +0.04742 | [-0.17791, +0.27190] |
| uniform | online − frozen | classic_moments | +0.09455 | [-0.15846, +0.32022] |
| uniform | online − frozen | classic_mc | +0.09284 | [-0.16089, +0.31931] |
| uniform | online − frozen | phantom | +0.10005 | [-0.15867, +0.34446] |
| evidence_improving | online − fixed | classic_moments | -0.00026 | [-0.19134, +0.18826] |
| evidence_improving | online − fixed | classic_mc | +0.00182 | [-0.18890, +0.19008] |
| evidence_improving | online − fixed | phantom | -0.01010 | [-0.23025, +0.18671] |
| evidence_improving | frozen − fixed | classic_moments | -0.34791 | [-0.55132, -0.14263] |
| evidence_improving | frozen − fixed | classic_mc | -0.34602 | [-0.54913, -0.14091] |
| evidence_improving | frozen − fixed | phantom | -0.47555 | [-0.72141, -0.25285] |
| evidence_improving | online − frozen | classic_moments | +0.34765 | [+0.16717, +0.53612] |
| evidence_improving | online − frozen | classic_mc | +0.34784 | [+0.16711, +0.53641] |
| evidence_improving | online − frozen | phantom | +0.46545 | [+0.23757, +0.69202] |

| Allocation | Arm | Phantom − classic MC RMSE | Paired 95% CI |
|---|---|---:|---|
| uniform | fixed | +0.12948 | [+0.03261, +0.22301] |
| uniform | online | +0.13289 | [+0.06927, +0.18630] |
| uniform | frozen | +0.12567 | [+0.01858, +0.23251] |
| evidence_improving | fixed | +0.26038 | [+0.19026, +0.33057] |
| evidence_improving | online | +0.24846 | [+0.16748, +0.32244] |
| evidence_improving | frozen | +0.13084 | [+0.05079, +0.20168] |

### 2M calls

| Allocation | Arm | Estimator | Bias | Error SD | RMSE | SE | Reported SD | Gaussian coverage | MC coverage | Mean MCSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | classic_moments | -0.10098 | 0.71504 | 0.71024 | 0.07918 | 0.36916 | 20/30 | — | — |
| uniform | fixed | classic_mc | -0.10141 | 0.71505 | 0.71031 | 0.07932 | 0.36930 | 20/30 | 20/30 | 0.00577 |
| uniform | fixed | phantom | -0.22236 | 0.85032 | 0.86509 | 0.09931 | 0.21526 | 11/30 | 11/30 | 0.00336 |
| uniform | online | classic_moments | -0.06324 | 0.88897 | 0.87631 | 0.13516 | 0.38191 | 18/30 | — | — |
| uniform | online | classic_mc | -0.06295 | 0.88852 | 0.87585 | 0.13500 | 0.38338 | 19/30 | 19/30 | 0.00599 |
| uniform | online | phantom | -0.16594 | 1.00623 | 1.00314 | 0.14789 | 0.22273 | 10/30 | 10/30 | 0.00348 |
| uniform | frozen | classic_moments | -0.21434 | 0.87176 | 0.88351 | 0.10538 | 0.37733 | 17/30 | — | — |
| uniform | frozen | classic_mc | -0.21503 | 0.87221 | 0.88410 | 0.10565 | 0.37824 | 17/30 | 17/30 | 0.00591 |
| uniform | frozen | phantom | -0.46148 | 0.88990 | 0.98918 | 0.09382 | 0.21918 | 10/30 | 10/30 | 0.00343 |
| evidence_improving | fixed | classic_moments | -0.64406 | 0.99924 | 1.17474 | 0.11632 | 0.35191 | 11/30 | — | — |
| evidence_improving | fixed | classic_mc | -0.64436 | 0.99906 | 1.17475 | 0.11659 | 0.35480 | 11/30 | 11/30 | 0.00554 |
| evidence_improving | fixed | phantom | -0.92584 | 1.24376 | 1.53381 | 0.13577 | 0.20501 | 4/30 | 4/30 | 0.00320 |
| evidence_improving | online | classic_moments | -0.38571 | 1.15725 | 1.20139 | 0.14500 | 0.35577 | 14/30 | — | — |
| evidence_improving | online | classic_mc | -0.38585 | 1.15695 | 1.20116 | 0.14423 | 0.35667 | 14/30 | 14/30 | 0.00557 |
| evidence_improving | online | phantom | -0.58696 | 1.42246 | 1.51673 | 0.16594 | 0.20631 | 5/30 | 4/30 | 0.00322 |
| evidence_improving | frozen | classic_moments | -0.16331 | 0.78163 | 0.78565 | 0.12988 | 0.36014 | 20/30 | — | — |
| evidence_improving | frozen | classic_mc | -0.16132 | 0.78160 | 0.78521 | 0.12987 | 0.36041 | 20/30 | 20/30 | 0.00563 |
| evidence_improving | frozen | phantom | -0.28327 | 0.90177 | 0.93077 | 0.15344 | 0.20819 | 14/30 | 14/30 | 0.00325 |

| Allocation | Contrast | Estimator | RMSE difference | Paired 95% CI |
|---|---|---|---:|---|
| uniform | online − fixed | classic_moments | +0.16607 | [-0.15155, +0.49030] |
| uniform | online − fixed | classic_mc | +0.16554 | [-0.15132, +0.48957] |
| uniform | online − fixed | phantom | +0.13805 | [-0.23657, +0.52609] |
| uniform | frozen − fixed | classic_moments | +0.17327 | [-0.07070, +0.42393] |
| uniform | frozen − fixed | classic_mc | +0.17379 | [-0.07099, +0.42529] |
| uniform | frozen − fixed | phantom | +0.12409 | [-0.17144, +0.41315] |
| uniform | online − frozen | classic_moments | -0.00720 | [-0.33929, +0.29986] |
| uniform | online − frozen | classic_mc | -0.00825 | [-0.34079, +0.29913] |
| uniform | online − frozen | phantom | +0.01395 | [-0.33261, +0.34507] |
| evidence_improving | online − fixed | classic_moments | +0.02666 | [-0.23562, +0.27235] |
| evidence_improving | online − fixed | classic_mc | +0.02641 | [-0.23695, +0.27356] |
| evidence_improving | online − fixed | phantom | -0.01708 | [-0.31757, +0.25875] |
| evidence_improving | frozen − fixed | classic_moments | -0.38908 | [-0.66861, -0.13525] |
| evidence_improving | frozen − fixed | classic_mc | -0.38954 | [-0.67104, -0.13403] |
| evidence_improving | frozen − fixed | phantom | -0.60304 | [-0.93906, -0.29426] |
| evidence_improving | online − frozen | classic_moments | +0.41574 | [+0.23209, +0.59403] |
| evidence_improving | online − frozen | classic_mc | +0.41595 | [+0.23301, +0.59431] |
| evidence_improving | online − frozen | phantom | +0.58596 | [+0.34556, +0.83736] |

| Allocation | Arm | Phantom − classic MC RMSE | Paired 95% CI |
|---|---|---:|---|
| uniform | fixed | +0.15478 | [+0.00722, +0.29425] |
| uniform | online | +0.12729 | [+0.04921, +0.19560] |
| uniform | frozen | +0.10508 | [-0.03069, +0.23759] |
| evidence_improving | fixed | +0.35906 | [+0.25431, +0.46042] |
| evidence_improving | online | +0.31557 | [+0.23052, +0.39676] |
| evidence_improving | frozen | +0.14556 | [+0.03249, +0.24838] |

## Posterior recovery at 4M calls

Only classic posterior weights participate. SD, CDF error and W1 below are medians across all 30 runs. Flat-coordinate reference SD is 0.288675; x5 is approximately 0.05 with the unchanged model's TFP scale precision. CDF sup error checks both sides of every weighted empirical jump. The unchanged W1 reduction uses 8,192 target midpoint quantiles, a deterministic approximation to continuous-target W1. Kish ESS measures weight concentration, not Markov-chain mixing.

| Allocation | Arm | Coordinate | SD | CDF sup error | W1 |
|---|---|---|---:|---:|---:|
| uniform | fixed | x1 | 0.17383 | 0.38299 | 0.18473 |
| uniform | fixed | x16 | 0.19142 | 0.36938 | 0.17990 |
| uniform | fixed | x17 | 0.20337 | 0.28265 | 0.14472 |
| uniform | fixed | x5 | 0.04896 | 0.04162 | 0.00416 |
| uniform | online | x1 | 0.14344 | 0.36539 | 0.17668 |
| uniform | online | x16 | 0.22834 | 0.31856 | 0.18089 |
| uniform | online | x17 | 0.21026 | 0.31486 | 0.17666 |
| uniform | online | x5 | 0.04855 | 0.04651 | 0.00461 |
| uniform | frozen | x1 | 0.16847 | 0.40290 | 0.19183 |
| uniform | frozen | x16 | 0.24805 | 0.25359 | 0.13453 |
| uniform | frozen | x17 | 0.22575 | 0.28957 | 0.14914 |
| uniform | frozen | x5 | 0.04994 | 0.05189 | 0.00476 |
| evidence_improving | fixed | x1 | 0.10566 | 0.47026 | 0.21089 |
| evidence_improving | fixed | x16 | 0.15593 | 0.39327 | 0.19464 |
| evidence_improving | fixed | x17 | 0.14431 | 0.40057 | 0.19191 |
| evidence_improving | fixed | x5 | 0.04948 | 0.04948 | 0.00377 |
| evidence_improving | online | x1 | 0.10702 | 0.51368 | 0.24938 |
| evidence_improving | online | x16 | 0.18009 | 0.35116 | 0.18173 |
| evidence_improving | online | x17 | 0.15063 | 0.36759 | 0.18052 |
| evidence_improving | online | x5 | 0.04858 | 0.04323 | 0.00352 |
| evidence_improving | frozen | x1 | 0.12204 | 0.46967 | 0.22588 |
| evidence_improving | frozen | x16 | 0.16596 | 0.37729 | 0.19675 |
| evidence_improving | frozen | x17 | 0.16381 | 0.37463 | 0.18450 |
| evidence_improving | frozen | x5 | 0.04939 | 0.05909 | 0.00514 |

## Controller learning and movement

The following are per-run medians except flags, which pool all children. Real updates change h/T; consuming observations with a zero delta is not counted as learning. Information exposure and deepest compression have different meanings. Full chronological receipts preserve eligibility and suppression reasons, raw jump/work sums and every parameter history.

| Allocation | Arm | Real updates (range) | Exposure | Deepest c | Mean B | Mean effective h | Mean nominal T | Extrapolated / clamped |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 0.0 (0–0) | 154.61 | 44.73 | 10.000 | 0.003000 | 0.030000 | 0.000% / 0.000% |
| uniform | online | 13.0 (8–18) | 153.28 | 44.85 | 11.489 | 0.003082 | 0.034649 | 42.968% / 34.659% |
| uniform | frozen | 0.0 (0–0) | 154.72 | 45.04 | 10.452 | 0.003117 | 0.032274 | 0.002% / 0.002% |
| evidence_improving | fixed | 0.0 (0–0) | 264.16 | 44.68 | 10.000 | 0.003000 | 0.030000 | 0.000% / 0.000% |
| evidence_improving | online | 10.0 (6–17) | 251.12 | 44.52 | 10.888 | 0.002955 | 0.032022 | 39.435% / 31.960% |
| evidence_improving | frozen | 0.0 (0–0) | 253.98 | 44.28 | 11.811 | 0.003035 | 0.035535 | 0.007% / 0.007% |

Across final snapshots: bound clipping 0, duration caps 0, compression-range clips 0. Maximum observed ceil reduction in effective h versus bounded desired h was 19.863%. All occurrences are retained.

Gap-bin movement and controller-compression-stratum movement are both retained. Whole-chain displacement includes the ordinary slice; endpoint rejection does not imply zero whole-chain movement. Compression aggregates retain all 18 coordinate and squared-coordinate jump sums, actual total/gradient/value work, and per-run correlations in the underlying analyses. Ratios pool jump sums and denominators, not individual ratios. Figures show posterior CDFs, gap-bin movement, compression objectives and learned parameter curves.

## Work and runtime

Every gradient counts once in logical model work. Roots, including later allocations, have one value call and zero B/h/T/gradients. Complete batches cross each budget; overshoot is retained. Masked heterogeneous vectorized execution can do more hardware work than these logical counts. Core wall time includes its compilation and writes; separately logged full-process XLA compilation also includes setup outside that timer and is not subtracted from it. Prepare, feedback and update timers are within core work; replay and reduction are separately charged analysis overhead.

| Allocation | Arm | Mean calls (SD) | Mean gradients | Max overshoot | Core s | Analysis s | Full-process XLA s | Prepare s | Feedback s | Update s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | fixed | 4018511 (10275) | 1627314 | 36039 | 71.634 | 31.707 | 58.944 | 5.161 | 0.262 | 0.043 |
| uniform | online | 4021010 (11155) | 1758365 | 39675 | 73.277 | 31.462 | 60.233 | 5.619 | 0.261 | 0.069 |
| uniform | frozen | 4018715 (10033) | 1676812 | 36919 | 73.068 | 32.050 | 60.058 | 5.620 | 0.264 | 0.043 |
| evidence_improving | fixed | 4013072 (7890) | 1707429 | 31221 | 96.356 | 36.188 | 77.421 | 7.642 | 0.439 | 0.079 |
| evidence_improving | online | 4014371 (9599) | 1800471 | 32094 | 79.697 | 32.938 | 62.772 | 6.342 | 0.369 | 0.100 |
| evidence_improving | frozen | 4015086 (9355) | 1874305 | 32306 | 79.280 | 32.465 | 62.204 | 6.202 | 0.359 | 0.070 |

Independent frozen-policy training cost 32,080,141 model calls across eight 4M runs, plus their full replay/reductions. Amortized over the 60 frozen evaluation runs, this is 534,669 additional calls per run (13.37% of 4M). Online learning uses no extra model calls for feedback; it still incurs host bookkeeping and can change the work per chain. Neither training nor held-out ensembles are hidden inside the evaluation budgets.

## Validation, interpretation and limits

Pre-release focused/original tests passed, including fixed child/phantom/count/key parity, phantom-off/on continuation parity, fresh checkpoint restoration, new-root accounting, chronological updates, bounds and same-shape compilation stability. Both allocation smokes pass complete chain and phantom replay. Independent audits pass all eight training ledgers, both frozen-table reconstructions and 21 held-out cases (42 single-path and full-chain laws), at the original stationarity and reversal tolerances. The final combined suite passed all 45 tests, including the direct constant-correlation regression added after sampling and reduction.

The original tiny-step domain failed reversal at B256, T0.00256, h1e-5, including accepted paths. All failed probes and raw arrays are preserved. The revised independent 4,480-path range audit passed original tolerances before training. The intermediate TFP/manual arithmetic comparison used different vector fields and is labeled separately. Passing these probes supports the tested range; it does not prove every floating-point path.

Online is experimental history-dependent adaptive MCMC. Freezing parameters for one chain does not establish stationarity of the adaptive history. Frozen-versus-online differs in training histories and is diagnostic, not a pure causal test of adaptation bias. Better local displacement or endpoint acceptance alone does not establish population coverage. Interpret evidence error together with posterior widths and CDF error; smaller reported evidence uncertainty alone is not improved accuracy.

The controller was exercised: online runs made 8–18 real updates under uniform allocation and 6–17 under evidence allocation. Therefore the unresolved online benefit is not explained by zero learning. In the 5–15 gap bin, the frozen evidence arm has slightly less x1 movement and lower x1 displacement per gradient than fixed, despite its lower evidence RMSE. A single local movement bin does not explain that benefit. The final frozen configuration and its independent training history are promising under evidence allocation, but this cohort does not identify which policy region or mechanism caused the improvement.

Even the frozen evidence arm remains under-dispersed: its median x1 SD is 0.12204 versus the 0.288675 reference. Phantom conditioning increases 4M RMSE relative to classic MC in all six cells, with positive paired intervals, while reducing reported uncertainty. It therefore does not repair the incomplete population coverage in this experiment.

The fixed-versus-learned comparisons also change the randomized parameter mixture: fixed has no perturbations, while online and frozen share the 90/10 mixture. They assess the complete configurations and cannot isolate the learned central schedule from that mixture. Only four independent training runs per allocation informed the frozen tables; training uncertainty and independent replication remain open.

Source 47a9384 and its 144-file freeze remained unchanged throughout sampling and reductions. The complete 180-run/360-snapshot checks verify all hashes, paired initial populations, draws, root/gradient/value counts and controller history. Original raw states, ledgers, policies, draws and audit sources remain on dorrie. No seeds were discarded, no evaluation retuning occurred, and no manuscript or GitHub artifact was changed.

One frozen/evidence seed-72 reduction initially stopped before chain replay because two undefined squared-coordinate correlations could not be encoded in strict JSON. Three identical seed coordinates had a roundoff-positive SD but zero SD after squaring. A separately hashed operational wrapper encoded only nonfinite correlation diagnostics as null and rejected every other nonfinite field. The retry passed full replay and both reductions; source and raw inputs were unchanged. The original failing log, diagnosis and wrapper receipt are retained. Successful-analysis timings above exclude this failed diagnostic attempt. No sampling run was repeated or omitted.

## Figures

[Evidence RMSE](reflection_online/evidence_comparison.png) shows both budgets and all three estimators; shaded bands are paired-seed bootstrap RMSE intervals. [Posterior CDFs](reflection_online/posterior_cdfs.png) show every seed at 4M against the reference curves, without selecting examples.

[Gap-bin movement](reflection_online/chain_movement.png) separates displacement, cost, midpoint count and realized duration. [Compression-stratum movement](reflection_online/compression_movement.png) pools raw jump and total-work sums. [Controller policies](reflection_online/controller_policies.png) show all final online policies versus classic compression and the independently frozen tables versus actual likelihood; these are different lookup coordinates.
