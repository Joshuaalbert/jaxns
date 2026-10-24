# Issue 312: next experiments

1. Implement a reversible reflection trajectory in the experimental benchmark.
   First use the true toy gradient, check time reversal, volume preservation,
   constrained-prior stationarity and movement, then compare nested-sampling
   evidence and marginal distributions across seeds. Account for gradients
   separately from value-only likelihood evaluations.
2. After the reflection experiment, investigate a better learned representation
   for the scaled-Hessian direction kernel. Try a smooth neural-network
   approximation as an alternative to the GP. Improve it incrementally as new
   samples arrive, keeping each approximation fixed during a chain. Audit
   curvature as well as held-out likelihood error: the GP experiment showed
   that fitting values on a thin population does not identify its normal
   curvature. Retain the endpoint direction-density MH correction and the
   true likelihood constraint. Learning a useful representation, potentially
   with additional off-contour observations, is a separate question from
   simply increasing optimiser iterations.

Requested on 23 October 2026. Track results in issue #312 and PR #314; these
are experimental kernels, not a change to the production sampler defaults.

## Results and remaining directions

- The [reflection experiment](benchmarks/issue_312/REFLECTION_REPORT.md) is
  implemented and tested: 30 seeds with nine slices plus 90 midpoint
  reflections give ln Z RMSE 0.922 at 1.465M model calls, including 0.586M
  gradient calls. Flat posterior marginals still mix poorly.
- The subsequent [NN representation audit](benchmarks/issue_312/NN_REPRESENTATION_REPORT.md)
  tests warm-started residual features and off-contour training observations.
  Added observations recover local stiff curvature, but broad-ridge accuracy
  and earlier fits remain poor.
- The [NN-driven sampling comparison](benchmarks/issue_312/NN_SAMPLING_REPORT.md)
  now tests both fits across six seeds with the existing MH correction and
  one-e-fold refits. Evidence RMSE is 104.52 nats stored-only and 110.96 with
  probes, with severe marginal collapse. A frozen-kernel stationarity check
  passes while endpoint correlations exceed 0.998. False moderate/flat
  curvature remains large despite recovering the dominant stiff eigenvalue.
- The [local GMM experiment](benchmarks/issue_312/LOCAL_GMM_REPORT.md) fits ten
  components for 100 EM iterations and corrects local responsibility-based
  direction selection. Six-seed evidence RMSE is 6.24 with warm starts and
  4.59 with fresh fits, versus 3.20/2.76 for matched global selection. Fresh
  fits avoid collapse to one component, but neither protocol recovers the
  ridge marginal. All 24 new runs and independent kernel/EM checks are saved.
- Next representation investigations: replay older/off-contour observations,
  preserve useful prior fits rather than selecting solely by interpolation
  loss, and test local or lower-dimensional residual features. Audit flat and
  moderate curvature as well as the dominant stiff eigenvector. Compare any
  further representation changes against the completed sampling failures.
