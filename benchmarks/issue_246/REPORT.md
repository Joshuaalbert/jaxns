# Issue 246 validation report

## Decision

The warm GMM/ellipsoidal direction kernel is suitable as an opt-in sampler
policy. It gives statistically clear likelihood-evaluation reductions on six
of the ten standard problems, including 25.6% on `basic3`, 19.2% on
`basic_mvn`, and 6.7% on `spike_slab10`. It is not a good default yet: these
cheap analytic likelihoods expose added CPU coordination cost, and posterior
mode-mass point estimates are mixed. No evidence- or mode-RMSE difference is
statistically resolved from the isotropic reference at 30 seeds.

The retained defaults are four components, ten bounded warm-EM iterations, a
fixed 1024-row fit population, and a 1% per-transition isotropic safety draw
once eligible geometry exists. A three-seed tuning sweep found that halving
the fit to five iterations and 512 rows did not materially improve steady
time and weakened the `basic_mvn` evaluation reduction. Lowering the initial
effective-sample gate improved that Gaussian reduction but worsened pilot
mode-mass RMSE, so the conservative gate remains.

## Protocol

- 30 paired seeds per problem, direction policy, and phantom setting (`n=30`).
- All ten cases from `cicd/tests/test_ns_standard_problems.py`.
- Root allocation `30 * dimension`, replacement width `10 * dimension`, and
  slice transitions `5 * dimension` for both policies.
- Identical default termination, `dlogZ = log1p(1e-3)`.
- Phantom-off rows use classic MC shrinkage. Phantom-on rows retain the first
  `dimension` eligible chain transitions and use phantom-conditioned MC
  shrinkage. Each result uses 1,000 shrinkage draws.
- Likelihood evaluations are the hardware-independent efficiency measure.
  `median_steady_run_s` excludes seed zero, which pays compilation.
- Host: 6-core/12-thread Intel i7-8750H CPU; Python 3.12.9; JAX/JAXLIB 0.10.0;
  x64 enabled.

The validator confirmed 40 cells and 1,200 finite records, matched seed sets
and scientific settings, and exact equality of the classic chain between
phantoms-off/on for every problem, policy, and seed. Full per-seed records are
in `raw/`; full calibration, timing, memory, fit, fallback, classic-sample,
and phantom-count aggregates are in `results.csv`.

## Per-problem comparison

Negative evaluation change is better. CPU timing includes coordination for a
very cheap analytic likelihood and therefore does not represent an expensive
user likelihood. Mode RMSE is listed only where relative mixture weights are
part of the problem design.

| Problem | Phantoms | n | RMSE isotropic | RMSE ellipsoidal | Likelihood eval change | Steady CPU change | Mode RMSE isotropic | Mode RMSE ellipsoidal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| basic | off | 30 | 0.0391 | 0.0350 | -0.1% | +324.1% | -- | -- |
| basic | on | 30 | 0.0396 | 0.0351 | -0.1% | +169.2% | -- | -- |
| basic2 | off | 30 | 0.0689 | 0.0691 | +0.3% | +145.2% | -- | -- |
| basic2 | on | 30 | 0.0691 | 0.0691 | +0.3% | +161.3% | -- | -- |
| basic3 | off | 30 | 0.1774 | 0.1982 | -25.6% | +147.9% | -- | -- |
| basic3 | on | 30 | 0.1698 | 0.2018 | -25.6% | +121.5% | -- | -- |
| basic_mvn | off | 30 | 0.2817 | 0.2554 | -19.2% | +7.8% | -- | -- |
| basic_mvn | on | 30 | 0.1546 | 0.1554 | -19.2% | +7.7% | -- | -- |
| plateau | off | 30 | 0.0324 | 0.0324 | 0.0% | +233.4% | -- | -- |
| plateau | on | 30 | 0.0323 | 0.0323 | 0.0% | +281.2% | -- | -- |
| spike_slab | off | 30 | 0.2774 | 0.2498 | +0.1% | +27.7% | 0.1494 | 0.1761 |
| spike_slab | on | 30 | 0.2759 | 0.2539 | +0.1% | +26.2% | 0.1494 | 0.1761 |
| spike_slab10 | off | 30 | 0.2276 | 0.2725 | -6.7% | +18.1% | 0.1365 | 0.1279 |
| spike_slab10 | on | 30 | 0.2376 | 0.2149 | -6.7% | +23.3% | 0.1365 | 0.1279 |
| weak_curved_mvn8 | off | 30 | 0.1428 | 0.1798 | -1.3% | +37.4% | -- | -- |
| weak_curved_mvn8 | on | 30 | 0.1342 | 0.1331 | -1.3% | +34.0% | -- | -- |
| weak_curved_spike_slab8 | off | 30 | 0.1685 | 0.1600 | -0.8% | +34.9% | 0.0740 | 0.0976 |
| weak_curved_spike_slab8 | on | 30 | 0.1486 | 0.1246 | -0.8% | +28.7% | 0.0740 | 0.0976 |
| weak_curved_spike_slab10 | off | 30 | 0.1814 | 0.1357 | -1.0% | +24.1% | 0.0997 | 0.1193 |
| weak_curved_spike_slab10 | on | 30 | 0.1147 | 0.1384 | -1.0% | +31.4% | 0.0997 | 0.1193 |

Paired 95% bootstrap intervals make the interpretation less dependent on
point estimates. Evaluation reductions exclude zero for `basic3`
([-27.3%, -23.9%]), `basic_mvn` ([-19.9%, -18.4%]), `spike_slab10`
([-7.9%, -5.4%]), `weak_curved_mvn8` ([-1.65%, -1.01%]), and both curved
spike--slabs (roughly [-1.3%, -0.36%]). Every evidence-RMSE difference and
every mode-mass-RMSE difference has a paired 95% interval containing zero.
The exact intervals are retained in `comparisons.csv`.

## JAX execution evidence

For one complete initial `basic_mvn` depth epoch, the isotropic and
ellipsoidal programs respectively measured:

| Metric | Isotropic | Ellipsoidal | Change |
| --- | ---: | ---: | ---: |
| Compile | 2.21 s | 3.14 s | +42.2% |
| Median execution | 1.304 s | 1.470 s | +12.8% |
| StableHLO text | 445,808 B | 611,323 B | +37.1% |
| Compiler temporary memory | 1,047,544 B | 1,478,544 B | +41.1% |

The fit remains outside the replacement `vmap`; all constrained chains are
still sampled concurrently. A placement microbenchmark using the exact
1024-by-8, four-component update payload compared the scalar device
conditional with returning to Python. Compile time (0.3825 vs 0.3733 s) and
temporary memory (800,872 vs 800,680 B) were equivalent. The device update was
faster (5.36 vs 5.98 ms median); its skipped branch cost 0.194 ms vs 0.091 ms
for a host check. The sub-millisecond skip difference does not justify a
synchronization boundary, so the simpler `lax.cond` placement is retained.

The final intent review also moved the unused cold initializer behind the warm
update conditional. The per-problem CPU rows were collected before that dead
work was removed and are therefore conservative; direction draws, likelihood
evaluations, and accuracy are unchanged. A final-tree `basic_mvn` replay
matched the committed classic-sample count, likelihood-evaluation count,
evidence summaries, fit count, and fallback count exactly.

## Regression and packaging checks

- The complete repository suite passes (232 tests), including all four
  reviewer autochecks.
- All 20 existing standard-problem tests pass without tolerance changes.
- The focused GMM, ellipsoid, constrained sampler, state/growth/checkpoint,
  and optional-dependency tests pass (54 tests).
- Capacity growth and pickle/resume preserve fitted sampler state and the
  scientific continuation.
- The built wheel declares Matplotlib in base `Requires-Dist`, omits the old
  plotting extra, and does not duplicate Matplotlib in example/test extras.
- Plot imports remain lazy and tests use the non-interactive Agg backend.
