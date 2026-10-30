# Coordinator protocol review 01

Read REFLECTION_ONLINE_PROTOCOL.md review version 1 in full. The bounded
joint h/T controller, whole-chain ratio-of-sums objective, chronology,
exploration, cohorts and benchmark-only integration are approved in principle.
Proceed with implementation/tests with the following correction before freeze.

1. IMPORTANT: the proposed frozen arm is not independently frozen if its
   parameter lookup still consumes c(lambda) reconstructed from its own
   evolving evaluation race tree. That introduces production-history
   dependence even with a frozen table in compression coordinates. Freeze an
   actual lambda -> (log h, log T) mapping using only independent training
   runs, then evaluate that mapping using the requested contour alone. Keep
   evaluation c/K only for diagnostics, never frozen-arm parameter choice.
   A simple solution is to use final training race trees to map their final
   compression policies to a common fixed likelihood-knot table, average log
   predictions equally across the four training runs, and freeze interpolation
   and bounded tails with it. Define the common likelihood grid and handling
   of untrained tails before observing training outcomes; no oracle peak/gap
   may determine the policy. Or use actual contour anchors throughout if
   simpler. Record that frozen-vs-online is a diagnostic comparison with
   different training histories, not a pure causal adaptation-bias test.

2. The exposure clock is a useful explicitly defined experimental rule, but
   it is NOT actual frontier compression. Retain this distinction in every
   report. Please keep both the accumulated new-chain exposure and observed
   compression frontier in raw histories. No claim that a reversed frontier
   or allocation round itself is another shrinkage e-fold.

3. Verify controller learning is actually exercised in a smoke/training
   preflight. With 2.5% per neighbor and >=8 for each sign, sparse observations
   plus resetting affected strata may prevent updates. Report per-stratum
   attempted/successful updates and reasons for suppression; do not silently
   count a never-updating run as a meaningful adaptive comparison. If a
   mechanical feasibility fix is needed, expose it before evaluation and
   preserve previous source/attempts. Do not tune using evaluation evidence.

4. A meaningful independent audit must be able to reconstruct each parameter
   and each update from chronological ledgers: save the actual predispatch
   coordinate inputs, generation/version, selected arm, requested/effective
   h/T/B, squared jump sum, work, and old statistics consumed by each update.
   Record native source/compiled-value parity; no evaluation outcomes released
   until these gates and the corrected frozen policy are reviewed.

No production-boundary expansion is needed: the existing gp_refresh and
allocation_observer establish the benchmark-only pre/post-batch seam.
One typed optional hook is an appropriate small extension, with unchanged
default behavior and parity tests. No extra online model evaluations/replays.
