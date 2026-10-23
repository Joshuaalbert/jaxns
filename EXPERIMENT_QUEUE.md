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
