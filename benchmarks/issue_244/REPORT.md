# Issue 244 likelihood-batching experiment

## Decision

The continuation design is strongly positive for long perfect-bracketing
`UniDimSliceSampler` batches. Keep the logical replacement width unchanged and
use the same width for likelihood evaluation. A completed lane evaluates the
filler point `U=0.5`; filler calls are included in every physical-work figure
below. Short chains retain complete-chain `vmap`: a follow-up end-to-end matrix
showed 1.5--2.2x bookkeeping regressions on cheap 1D/2D problems, so the
evidence-backed continuation boundary is currently 32 slice transitions.
Worker batches narrower than eight retain complete-chain execution because the
measured wall overhead outweighed their smaller physical-work saving. Size one
therefore retains the scalar path because no cross-lane imbalance exists.

The selected design keeps each chain's PRNG stream and output position fixed,
continues independently across slice-transition boundaries, precomputes the
already-determined direction stream once, and retains only the requested
phantom prefix.  The race-tree scheduler continues to see one atomic completed
replacement batch.

## Work accounting

The existing public counter is a logical counter.  It does not expose masked
work performed by other lanes while a rejection loop waits for its slowest
lane.  The experiment replayed the current implementation with an additional
per-transition count `n[i,t]`, and verified that this instrumentation was
bitwise identical to the public sampler.

For `S` logical chains and `T` slice transitions:

- logical work is `sum(i,t, n[i,t])`;
- current complete-chain `vmap` physical scalar-lane work is
  `S * sum(t, max(i, n[i,t]))`;
- selected same-width continuation work is
  `S * max(i, sum(t, n[i,t]))`.

The last expression includes all tail filler calls.  The experiment separately
recorded `physical - logical` as filler work.  No imputed current JAXNS counter
was used as the physical denominator.

## Final matrix

Environment: JAX/JAXLIB 0.10.0, x64, CPU.  Each row has 10 independent chain
batches and seven timed steady executions per batch.  Width and slice count
match the standard-problem defaults: `S=10D`, `T=5D`.  GMM rows use fitted
contour geometry and `prob_isotropic=1e-2`.

| Problem | contour quantile | direction | physical calls reduced | median wall improvement | compile ratio candidate/current | temporary MiB current -> candidate |
|---|---:|---|---:|---:|---:|---:|
| basic MVN 8D | 0.50 | isotropic | 56.8% | 52.4% | 1.70x | 0.17 -> 0.52 |
| basic MVN 8D | 0.90 | GMM | 51.9% | 57.2% | 1.44x | 0.18 -> 1.37 |
| spike--slab 8D | 0.50 | isotropic | 58.8% | 51.9% | 1.57x | 0.18 -> 0.52 |
| spike--slab 8D | 0.90 | GMM | 59.0% | 44.2% | 1.29x | 0.18 -> 1.37 |
| spike--slab 8D | 0.99 | isotropic | 58.5% | 46.2% | 1.65x | 0.18 -> 0.52 |
| weak curved spike--slab 8D | 0.90 | GMM | 60.0% | 45.4% | 1.32x | 0.18 -> 1.37 |
| spike--slab 10D | 0.90 | GMM | 59.9% | 48.5% | 1.47x | 0.32 -> 3.03 |

Across all 70 batches, median physical reduction was 58.8% and median wall
ratio was 0.521.  Individual physical reductions ranged from 35.6% to 65.7%; no
batch regressed in steady wall time (wall ratios 0.350 to 0.869).

Logical likelihood counts matched in 70/70 batches. Discrete outputs matched
in 70/70. This initial mechanism matrix batched both the chain and transition
axes when precomputing GMM directions. Forty isotropic batches were bitwise
identical, while the GMM batches were within `rtol=1e-5, atol=1e-6`, with
maximum absolute difference `5.16e-5`.

CI against JAX/JAXLIB 0.11.1 showed why that apparently small discrepancy was
not acceptable: the changed matrix-reduction order sent the deterministic
weak-curved spike--slab chain down a different path and failed its unchanged
standard-problem tolerance. The final implementation scans direction
transitions in the same order as the complete-chain reference while still
vmapping lanes. The failing standard problem then passed unchanged on 0.11.1,
and the focused GMM discrepancy fell to `2.78e-17` on 0.10.0. A subsequent
production-path rerun gave bitwise equality in all 40 GMM batches:

| Problem | width | transitions | physical calls reduced | median wall ratio | bitwise batches |
|---|---:|---:|---:|---:|---:|
| spike--slab 8D + GMM | 80 | 32 | 58.2% | 0.834 | 10/10 |
| spike--slab 8D + GMM | 8 | 40 | 39.8% | 1.075 | 10/10 |
| weak curved spike--slab 8D + GMM | 80 | 40 | 60.0% | 0.721 | 10/10 |
| spike--slab 10D + GMM | 100 | 50 | 59.9% | 0.748 | 10/10 |

The width-eight result is an explicit 7.5% bookkeeping trade-off on a cheap
likelihood for a 39.8% reduction in physical likelihood calls. It remains the
selected lower boundary because issue 244 prioritises likelihood efficiency
provided wall time does not regress materially; width four had already shown a
larger 36.9% wall regression for only 26.7% physical savings.

## Decision boundaries

Each row below contains 10 independent batches and seven steady executions per
batch. The T=32 row directly validates the selected transition boundary. Worker
widths use the fitted spike--slab GMM direction law expected for expensive
distributed problems.

| Problem | width | transitions | physical calls reduced | median wall ratio | decision |
|---|---:|---:|---:|---:|---|
| basic 1D | 10 | 5 | 31.3% | 1.098 | complete chains |
| basic 2D | 20 | 10 | 46.2% | 1.371 | complete chains |
| spike--slab 8D + GMM | 80 | 32 | 58.2% | 0.834 | continuation |
| spike--slab 8D + GMM | 2 | 40 | 12.9% | 1.208 | complete chains |
| spike--slab 8D + GMM | 4 | 40 | 26.7% | 1.369 | complete chains |
| spike--slab 8D + GMM | 8 | 40 | 39.8% | 1.075 | continuation |
| spike--slab 8D + GMM | 16 | 40 | 47.7% | 0.662 | continuation |

The full 30-seed standard-problem table is in `END_TO_END.md`. Current and
candidate `basic_mvn` runs were additionally alternated in one process because
separate-process timing initially exaggerated its regression. The controlled
median candidate/current ratios were 1.076 without phantoms and 1.139 with
phantoms. Sampler-only measurements on the same problem still removed 51--63%
of physical calls and ran 31--54% faster across contour quantiles 0.1, 0.5,
0.9, and 0.99. This is the explicit wall-time trade-off retained because the
primary objective is reducing expensive likelihood work; it is not hidden in
an aggregate timing.

## Alternatives rejected

- Smaller physical likelihood batches reduce filler work, but add many more
  compiled event-loop iterations.  On the representative spike--slab batch,
  widths 1--32 saved slightly more physical calls but materially lost wall
  time.  Same-width continuation retained a 63% physical reduction without
  that trade-off.
- Gathering ready chains is unnecessary when physical and logical widths are
  equal.  Direct fixed-lane evaluation removed bookkeeping and is the selected
  ordinary and distributed-worker path.
- Drawing the next GMM direction inside a vmapped conditional computes both
  branches.  This caused a measured 10D GMM wall regression.  The direction
  stream is independent of accepted coordinates, so precomputing it once is
  both scientifically equivalent and faster.
- Retaining all accepted transitions is unnecessary.  Only the start-of-chain
  phantom prefix is stored.

## Validation and performance-intent review

- The full CICD directory passes: 264 tests, including all 20 unchanged
  standard-problem release gates, phantom collection off/on, MC evidence
  checks, and real distributed worker lifecycle tests.
- Continuation dimensions and slice counts are static per compilation. Worker
  programs retain their existing batch-width cache, so the strategy adds no
  data-dependent retracing or host callbacks.
- Direction and phantom storage is temporary O(S T D) and O(S P D) work, not
  persistent nested-sampling state. Measured compiler temporary memory rises
  from 0.17--0.32 MiB to 0.52--3.03 MiB in the representative matrix, and
  compile time rises 1.29--1.70x. These are accepted explicit trade-offs for
  the steady physical-work reduction.
- Vmapped conditions select between cheap state updates, but fitted direction
  draws are hoisted because batching a condition evaluates both branches. The
  transition stream uses a scan matching the reference reduction order, while
  the independent chain dimension remains vmapped. This preserves one fixed
  direction law per parent contour without repeating GMM direction work on
  rejected proposals or changing difficult chains through roundoff.
- The local depth loop and worker program remain the sole JIT boundaries.
  Helper-level nested JIT was rejected because registered distributed `args`
  may intentionally contain notebook functions captured by the worker session.
- Static capacity-tail rows are masked from likelihood and phantom totals.
  Their device filler work is physical benchmark work, never a classic sample
  or phantom cluster in user-facing logical accounting.

Raw mechanism records are in `suite_final.json`. End-to-end accuracy and
performance use `run_end_to_end.py` with the same standard-problem builders,
sampler defaults, termination conditions, and 30 PRNG seeds for the candidate
and a fresh `develop` checkout. The raw files are `end_to_end.json` plus the
final short-chain and `basic_mvn` overrides, `develop_end_to_end.json`, and the
same-process `paired_basic_mvn.json`. `boundaries.json` contains the transition-
and worker-width decision matrix. `scan_order_summary.json` records the final
post-CI GMM rerun summarised above.
