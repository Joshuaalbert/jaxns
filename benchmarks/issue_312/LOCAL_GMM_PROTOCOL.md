# Local GMM direction experiment

Specified before the sampling cohort on 25 October 2026. Tracks issue #312
and draft PR #314. This is benchmark tooling, with no production API change.

- Same 18D curved ridge, stiff width 1e-5, seeds 0–5, 150 roots, 90 slices,
  actual float64 cube coordinates, uniform allocation, one default depth
  epoch, replacement width 150, no phantoms.
- Ten full-covariance Gaussian components, 100 EM iterations per fit, existing
  `fit_gmm` implementation and relative covariance regularization 1e-6.
  Fit all classic points above the completed batch's shallowest contour with
  equal weights. Initial fit uses all 150 roots; later fits warm-start the
  previous parameters after at least one expected-volume e-fold.
- This lower-level benchmark allows regularized ten-component fits even at
  150 points; unlike the public staged GMM API, it does not impose a minimum
  of K*(D+1) observations. Record component effective membership and smallest
  covariance eigenvalues, since those initial fits can be underdetermined.
- At x, choose k with responsibility proportional to pi_k N(x; mu_k, C_k),
  draw v ~ N(0, C_k), and normalize v to a direction. A fixed 1% branch draws
  an isotropic direction. The GMM mean chooses locality; it is not a proposed
  position or a displacement mean. No likelihood derivative is required.
- Propose y using the unchanged true-constrained full-cube slice. Accept with
  min(1, q(direction | y) / q(direction | x)), using the full mixture of
  angular central Gaussian densities plus the isotropic branch. Retain
  rejected self-transitions. Both endpoint laws use the same frozen fit.
- Run a paired six-seed control selecting components by global mixture masses
  pi_k. Its direction law is independent of x, so the MH ratio is identically
  one. Both arms use identical fitting settings and key splitting; their
  subsequently sampled populations can differ.
- Compare evidence, quoted errors, peak gaps, likelihood calls, flat and
  Gaussian marginal widths/CDFs against the existing seeds 0–5 controls.
  Fitting and direction arithmetic add cost but no likelihood evaluations.
- Check full mixture densities against an independent dense reference,
  uniform-target stationarity with a deliberately uncorrected negative
  control, rejected self-transitions and callback accounting. Run a separate
  exact constrained-prior stationarity/movement audit on an archived fit.

This uses above-contour refits, unlike the earlier posterior-weighted GMM
fitted once after evidence uncertainty reached 0.1. It changes component
selection from the released contour-volume rule to local responsibility or
global mixture mass. The global arm isolates local selection under the new
fitting protocol; neither arm is a replay of the older staged experiment.

## Follow-up after the first completed warm-started seeds

Local seed 0 and its global control assign essentially all current points to
one component at their last fits. Their other components retain tiny masses
and stale geometry. Add seeds 0–5 with fresh initialization at every refresh,
again in both local and global arms, with the same 100-iteration budget and
all other settings. Keep and report the original warm-started cohort in full.
This tests whether repeated warm starts obscure the intended local-mixture
comparison; it is not a post-hoc replacement of unfavorable runs.
