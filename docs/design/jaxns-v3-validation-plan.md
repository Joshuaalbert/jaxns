# JAXNS v3 Validation Plan

Status: paper-derived design draft.
Source: `docs/design/paper.tex`, Sections 5 and 6.

This document turns the paper's experimental protocol into a design-level
validation plan. It is agnostic of current tests and current implementation
state.

## Compared Methods

The paper states that experiments will compare:

- baseline race-tree nested sampling;
- phantom-conditioned shrinkage;
- dynamic contour allocation;
- gradient-informed constrained sampling.

All reported experiments should average over many independent random seeds.

## Correctness and Calibration

The first experiment class validates the conditioning mechanism on problems for
which evidence or the survival curve is known.

The key diagnostic is the evidence z-score:

```text
z_logZ = (hat logZ - logZ_ref) / hat sigma_logZ
```

where `hat logZ` and `hat sigma_logZ` come from Monte Carlo shrinkage.

For repeated independent seeds, the expected calibration behavior is:

```text
mean(z_logZ) ~= 0
std(z_logZ) ~= 1
```

Reports should include:

- bias in `hat logZ`;
- empirical root-mean-square error;
- relationship between reported and empirical uncertainty;
- comparison of expectation-based estimates to Monte Carlo shrinkage estimates;
- per-block Kish participating-cluster counts and phantom gate activation.

## Plateau Tests

Plateau tests should include likelihoods with equality atoms.

They should measure:

- recovery of equality masses;
- evidence z-score calibration;
- bias in `hat logZ`;
- empirical root-mean-square error;
- comparison of expectation-based estimates to Monte Carlo shrinkage estimates;
- relationship between reported and empirical uncertainty;
- per-block Kish participating-cluster counts and phantom gate activation;
- whether phantom equality counts improve atom-mass inference without biasing
  shrinkage.

These tests are included because plateau censorship is a central motivation for
the Bayesian shrinkage model.

## Evidence-Efficiency Benchmarks

The second experiment class uses difficult evidence benchmarks chosen to stress
constrained-prior exploration.

For each problem and method setting, report:

- likelihood evaluations per effective sample size;
- wall-clock timing where meaningful;
- root-mean-square error of `logZ`;
- cost-error tradeoffs.

The paper proposes Pareto plots of RMSE against likelihood evaluations. It also
states that, because log-evidence variance is inversely proportional to
effective sample size, a method is considered dominated when another method has
lower `MSE * likelihood_evals`.

## Posterior-Estimation Benchmarks

The third experiment class measures posterior quality on problems with difficult
modes and high-resolution reference posterior samples.

For posterior sample set `hat Pi` and reference sample set `Pi_ref`, the paper's
main discrepancy is a Monte Carlo estimate of Wasserstein distance:

```text
hat W(hat Pi, Pi_ref)
```

Posterior-quality metrics should be reported separately from evidence
calibration. The paper explicitly notes that a method can improve evidence
uncertainty without improving posterior representation.

## Parallel and Load-Balanced Scaling

The JAXNS v3 section describes heterogeneous distributed likelihood evaluation
with a central coordinator, load-balanced workers, ZMQ communication, Python
`pickle` serialization, and overlapping constrained-sampling calls from
different parent contours. The paper's experimental protocol only requires
wall-clock timing where meaningful; when distributed scaling is being evaluated,
the following reporting dimensions are useful:

- worker count and node configuration;
- compute-sector configuration and multi-tenant sharing mode;
- likelihood-evaluation throughput;
- wall-clock time to reach the selected goal condition;
- scheduler overhead relative to likelihood time;
- serialization and communication failures, if any;
- whether asynchronous child completion preserves the known in-flight parent
  target and phantom cluster identity until the parent out-degree is updated.

## Reporting Schema

The paper includes a placeholder evidence-calibration table. The design-level
schema is:

| Problem | Method | Seeds | Mean z_logZ | SD z_logZ | RMSE logZ | Likelihood evals |
| --- | --- | --- | --- | --- | --- | --- |
| To be filled | Baseline | To be filled | To be filled | To be filled | To be filled | To be filled |
| To be filled | Phantom-conditioned | To be filled | To be filled | To be filled | To be filled | To be filled |
| To be filled | Dynamic allocation | To be filled | To be filled | To be filled | To be filled | To be filled |
| To be filled | Gradients and phantoms | To be filled | To be filled | To be filled | To be filled | To be filled |

The paper also calls for:

- Pareto-front plots of evidence error versus likelihood evaluations;
- posterior Wasserstein distance versus likelihood evaluations per effective
  sample.

## Method Dimensions to Report

The paper names dynamic allocation as a method setting, and the JAXNS v3 section
breaks it into:

- depth-first uniform allocation;
- evidence-improving allocation based on `U^Z_g`, the expected reduction in
  delta-approximated log-evidence variance;
- posterior-improving allocation based on `U^P_g`, the expected increase in
  Kish effective sample size from splitting posterior shells;
- parent-block reach probabilities `T^Z` and shell probabilities `T^P`;
- unit-peak utility normalization, linear interpolation to `bar U(L)`, and
  target construction `K_*^k(L) = d_0 + k DeltaK bar U(L)`;
- depth conditions for small remaining evidence and small remaining posterior
  mass;
- user-specified goal conditions such as target effective sample size or target
  evidence uncertainty.

The paper also names constrained-sampler choices that can affect benchmarks:

- isotropic Gaussian direction kernels;
- GMM-based ellipsoidal Gaussian direction kernels fit to the current
  posterior approximation from accumulated accepted classic samples and
  nested-sampling posterior weights, under the v3 policy of attempting updates
  every five completed shells, with component choice proportional to integrated
  volume and kernel snapshots frozen for each chain and shell epoch;
- straight-line perfect bracketing;
- Galilean sampling, including path-length-uniform sampling after Hausdorff
  reflections and U-turn trajectory construction;
- mode evaporation or tunneling concerns, which adaptive allocation can
  ameliorate.

## Assumption Checks

The paper's expected diagnostics depend on the underlying assumptions:

- classic children are approximately independent draws from strict constrained
  priors;
- race-tree accounting preserves sentinel and sample out-degrees, in-flight
  parent targets until acceptance, and active lineage counts;
- plateau or no-seed fallback applies the out-degree increment to the actual
  dispatch parent after sentinel fallback, without requiring stored parent
  links in persisted sample state;
- non-isotropic direction-kernel adaptation is owned by the coordinator, uses
  the current posterior-weighted approximation from accumulated accepted
  classic samples, excludes phantom coordinates, and is frozen during chains
  and shell epochs so it cannot adapt to a current chain point;
- GMM direction-kernel component probabilities are proportional to integrated
  ellipsoidal volume; component count, covariance regularization, bounded
  fitting-row policy, invalid-fit thresholds, and fallback to the previous
  valid kernel or isotropic kernel follow the constrained-sampling design; and
  fallback activation is reported;
- phantom states are stationary enough after burn-in;
- Galilean trajectory-building points are excluded from phantom clusters; only
  retained post-burn-in constrained-sampler states that are uniform under the
  parent contour may condition shrinkage;
- phantom clusters are approximately independent between clusters and may be
  arbitrarily correlated within cluster;
- phantom conditioning uses parent-contour-gated per-cluster counts,
  `Gamma(1, 1)` cluster weights, and a Kish participating-cluster threshold;
- plateau posterior weighting uses `X_g = X_{g-1} p_{>g}`, splits equality atom
  mass equally across plateau samples, and uses `1 - p_{>g}` for non-plateau
  mass.

## Expected Checks

The paper does not give numeric pass thresholds or a pass/fail acceptance
framework. It describes the following expected checks:

- evidence z-scores are centered near zero across seeds;
- evidence z-score standard deviation is near one across seeds;
- reported uncertainty tracks empirical uncertainty;
- switching from isotropic to GMM-based non-isotropic direction kernels does
  not measurably bias known-reference evidence estimates, while any benefit is
  reported as mixing, likelihood-evaluation efficiency, or posterior-exploration
  improvement;
- plateau equality mass recovery improves when informative phantom equality
  counts are present;
- singleton independent phantom clusters match the iid Dirichlet posterior,
  while correlated multi-sample clusters have larger shrinkage variance without
  a systematic mean shift;
- phantom conditioning does not introduce measurable log-evidence bias in known
  reference problems;
- efficiency comparisons are reported as cost-error curves, not only as single
  headline timings;
- posterior quality is evaluated independently from evidence quality.

The exact benchmark problems, seed counts, and pass thresholds remain open until
the placeholder experimental section is filled in.
