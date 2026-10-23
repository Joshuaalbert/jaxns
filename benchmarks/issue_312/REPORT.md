# Contour-covariance experiment: results

Tracks [issue #312](https://github.com/Joshuaalbert/jaxns/issues/312).

The proposed covariance directions substantially improve the reported toy, but
do not resolve its curved-ridge failure. The straight-ridge control recovers
approximately correct evidence and posterior widths. Keep this implementation
experimental.

## Method

Both arms use 150 roots, 90 slice transitions, the default replacement width and
depth condition, and stiff width `1e-5`. The curved comparison uses seeds 0–29 in
each arm; the straight control uses seeds 0–9. Every recorded run completed its
first depth epoch without a hard termination. The two pilot records are excluded
from all aggregates.

The only scientific intervention is the direction law described in
[README.md](README.md). At each refresh, fit the unweighted empirical covariance
`C` of all stored classic points strictly above the shallowest constrained parent
in the dispatch batch. Use

```text
L = cholesky(C + 1e-8 * trace(C) / D * I)
direction = normalise(L @ Normal(0, I))
```

Refresh after the expected log prior volume moves by one e-fold. One factor is
shared across the dispatch batch and remains frozen throughout each chain.
The implementation retains the existing bracket, slice count, random-key
splitting, seed-selection rules, allocation, stopping, phantom retention, and
evidence calculations. Evidence below is the classic result, without phantom
conditioning.

The toy has fifteen Gaussian residual dimensions and three flat coordinates.
Its log evidence is zero up to negligible Gaussian tails outside the unit cube.
The straight control changes only the sinusoidal curvature amplitude from
`0.3` to `0`.

## Evidence and likelihood work

Evidence RMSE is calculated across seeds relative to log Z = 0. Quoted
uncertainty is the mean of the individual runs' reported standard deviations.
Coverage counts runs whose error is at most twice their own quoted uncertainty.
Evaluations are mean logical likelihood calls per run.

| Geometry | Directions | Seeds | Mean log Z | Evidence RMSE | Quoted uncertainty | Within 2 quoted SD | Evaluations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curved | Isotropic | 30 | -14.36 | 15.61 | 0.51 | 0/30 | 10.16M |
| Curved | Covariance | 30 | -0.69 | 5.50 | 0.44 | 7/30 | 5.95M |
| Straight | Isotropic | 10 | -16.28 | 17.12 | 0.52 | 0/10 | 10.09M |
| Straight | Covariance | 10 | +0.06 | 0.47 | 0.46 | 9/10 | 1.26M |

On the curved toy, covariance directions reduce RMSE by 10.11 nats and logical
likelihood work by 41.4%. A paired bootstrap over the 30 seeds gives a 95%
percentile interval of [-12.47, -7.83] nats for the RMSE change (10,000 resamples,
bootstrap seed 312). This interval describes these replicates, not a correctness
guarantee.

The curved covariance mean of -0.69 nats hides a between-seed standard deviation
of 5.55 nats. Its quoted uncertainty of 0.44 nats remains badly miscalibrated.
Median peak deficit improves from 4.36 to 2.03 nats. A near-zero average evidence
is not sufficient to call the curved experiment successful.

On the straight control, covariance directions require 8.0 times fewer
likelihood evaluations and recover evidence scatter consistent with the quoted
uncertainty in this ten-seed check.

## Posterior widths

Each entry is the median across runs of the posterior standard deviation.
The expected widths are 0.288675 for x1, x16 and x17, and 0.05 for x5.

| Geometry | Directions | x1 | x16 | x17 | x5 |
| --- | --- | ---: | ---: | ---: | ---: |
| Curved | Isotropic | 0.0283 | 0.0348 | 0.0351 | 0.0266 |
| Curved | Covariance | 0.0198 | 0.0671 | 0.0729 | 0.0376 |
| Straight | Isotropic | 0.0239 | 0.0321 | 0.0330 | 0.0284 |
| Straight | Covariance | 0.2869 | 0.2867 | 0.2885 | 0.0504 |

The curved toy's inert directions improve but remain severely collapsed; the
ridge coordinate x1 remains collapsed too. The straight control recovers all
four tested widths.

The isotropic core prevents singular geometry. It does not ensure that a
population which has lost coverage can recover it. The straight/curved contrast
is consistent with global covariance handling linear anisotropy while failing
to follow the full curved contour; these runs do not establish that curvature
is the only remaining cause.

The analytic interior-contour volume diagnostic provides an additional check
on shrinkage:

```text
X(ell) = volume(unit 15-ball) * sigma_stiff * 0.05**14
         * (2 * (peak - ell))**(15/2)
```

This holds for `peak - ell < 50`, where the residual ellipsoid fits inside the
cube for every ridge coordinate. The run's posterior-weighted mean difference
between inferred and analytic log X, averaged over runs, changes from -12.64
to +0.05 on the curved toy, and from -14.66 to +0.16 on the straight control.
As with mean evidence, the curved average alone conceals substantial failures.

## Core sensitivity

Three-seed exploratory checks vary only the covariance ridge coefficient:

| Coefficient | Seeds | Mean log Z | Evidence RMSE | Median x16 width | Median x17 width |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-6 | 3 | -5.48 | 10.87 | 0.1636 | 0.1161 |
| 1e-10 | 3 | +1.80 | 2.89 | 0.0614 | 0.0649 |

These small cohorts do not identify an optimal coefficient, and both still
collapse posterior directions. The main experiment retains its initial `1e-8`
coefficient.

## Reproducibility and limits

The base is `9efa45503def9ecae943afb4b6e1d45f61b1efa3`; the scientific
implementation is commit `0b16e67fbe955651312f14354708d78a14fb79cc`.
Commit `492d9df` changes benchmark execution and aggregation only. Some control
records were generated with that benchmark change uncommitted; their
`diff_sha256` records the source state. No sampler code changed between cohorts.
The last two isotropic curved seeds were rerun after an interrupted shard;
all final groups contain exactly the intended unique seeds.

Runs used JAX 0.11.1, Python 3.12.9, and the CPU backend with x64 enabled.
Covariance and evidence arithmetic use float64, but JAXCTX supplies float32
unit-cube coordinates in this model. Both arms preserve that dtype. This differs
from the reporter's environment and is a qualitative reproduction, not a
bitwise comparison with their reported runs.

Cohorts ran concurrently. The initial curved benchmark set CPU affinity after
imports had initialized backend thread pools; those elapsed times are not
single-core throughput evidence. The benchmark now sets affinity before
backend imports. Raw timing and warmup metadata are retained, but this report
compares logical likelihood work and makes no wall-time speedup claim.

Geometry is an empirical approximation from stored classics, not an independent
sample of the current contour. Different parent contours in one batch share
the geometry fitted at the shallowest parent. Refresh checks reconstruct and
sort the classic block path, adding experimental overhead. The implementation
is opt-in and local only; periodic priors, simultaneous GMM geometry, and
distributed continuation are explicitly rejected. No pRT retrieval was run.

The isotropic seed-0 baseline reproduces the earlier local diagnostic:
log Z = -21.3052286, uncertainty 0.5857316, and 12,173,777 likelihood calls.

All 86 non-pilot records are in [raw/](raw/). Machine-readable aggregates are
[summary.csv](summary.csv) and [summary.json](summary.json). Regenerate them with:

```bash
conda run -n jaxns_py python benchmarks/issue_312/summarise.py
```

See [README.md](README.md) for running individual experiments.

## Validation

149 targeted tests passed across covariance geometry, constrained sampling,
run control, state, local core, periodic sampling, and distributed core.
Four repository coverage checks also passed. These include strict contour
selection, regularisation, refresh cadence, dtype preservation, frozen-chain
geometry, scalar/continued sampling agreement, and checkpoint/resume.

The package's fatal-error Flake8 gate passes. New Python files pass full Flake8
with a 100-character line limit and Ruff. Package-wide Ruff reports the same
eight pre-existing violations as the base, with none introduced. The full
repository test suite was not run.
