# Ticket 0012: Galilean and Gradient-Informed Sampler Implementation

Branch: `feature/v3-galilean-gradient-sampler`
Priority: after baseline sampler modes
Depends on: Ticket 0007, Ticket 0008
Design docs:

- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-validation-plan.md`

## Goal

Implement the advanced constrained-sampler method described by the paper as
Galilean sampling. This is the v3 gradient-informed constrained-sampling method
for validation unless a later design explicitly adds another gradient-informed
mode.

Use the existing `UniDimSliceSampler` boundary where possible, and treat
`DistributedUniDimSliceSampler` as a migration reference for worker-backed
execution. The minimal path is to add a Galilean trajectory mode behind explicit
configuration, reuse existing seed/phantom/result contracts, and replace the
current `gradient_guided` placeholder only when the tests below pass.

## Current Code Context

Relevant files likely include:

- `src/jaxns/constrained_sampler.py`
- `src/jaxns/constrained_sampler_distributed.py` as a migration reference only
- `src/jaxns/model.py`
- `tests/test_constrained_sampler.py`
- worker-runtime tests introduced by Ticket 0009

Current `gradient_guided` paths raise `NotImplementedError`. Preserve that
failure for unsupported configurations until the Galilean mode is complete.

## Required Behavior

- Galilean sampling uses Hausdorff reflections off hard likelihood constraints.
- Construction proceeds from the seed in both `n_hat` and `-n_hat` directions.
- A single side follows the paper algorithm:
  propose a step, halve step size until back inside if the proposal leaves the
  contour, double step size until leaving if the proposal remains inside,
  reflect from the inner point, normalize the direction, and continue while
  `n^i . n^0 >= 0`.
- The trajectory stops at the paper's U-turn criterion:
  the current direction is anti-aligned with the initial direction.
- Sampling on the full non-straight trajectory is uniform by path length:
  choose a segment proportional to segment length, then sample uniformly within
  that segment.
- Galilean trajectory-building points are not retained as phantom samples. They
  do not uniformly sample `pi_lambda`.
- Retained phantom clusters, if enabled, remain post-burn-in Markov chain states
  from completed constrained-sampler transitions, not internal reflection or
  bracketing points.
- Failures, maximum reflection/step limits, degenerate gradients, and boundary
  numerical cases fail clearly or use a documented fallback that preserves the
  sampler contract.
- Unsupported gradient-informed variants fail clearly.

## Test Plan

Write tests before implementation.

Required tests:

- Round-trip/reversal tests for reflected trajectories.
- Detailed-balance or symmetry checks on simple geometries where the expected
  behavior is analytically clear.
- U-turn termination tests with fixed seeds and toy geometry.
- Uniform-on-trajectory tests that do not merely check in-slice validity.
- Segment-length weighting tests for non-uniform segment lengths.
- Strict contour validity for final classic samples and retained phantoms.
- Tests proving Galilean internal trajectory-building points are excluded from
  phantom clusters.
- Negative tests for max-reflection failure, boundary degeneracy, and unsupported
  mode selection.
- Parity or explicit-difference tests for local and worker-backed sampler
  behavior, without creating a separate sampler algorithm.

## Acceptance Criteria

- Galilean/gradient-informed terminology is unambiguous for validation.
- Tests prove reversibility/symmetry properties, not only that returned points
  satisfy the slice.
- Existing straight-line sampler tests still pass without changing default
  behavior.
