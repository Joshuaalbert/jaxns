# Issue 244 likelihood-batching experiment

## Decision

The continuation design is strongly positive and should replace complete-chain
`vmap` for perfect-bracketing `UniDimSliceSampler` batches.  Keep the logical
replacement width unchanged and use the same width for likelihood evaluation
by default.  A completed lane evaluates the filler point `U=0.5`; filler calls
are included in every physical-work figure below.

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

Logical likelihood counts matched in 70/70 batches.  Discrete outputs matched
in 70/70.  Forty batches were bitwise identical; every floating output was
within `rtol=1e-5, atol=1e-6`, with maximum absolute difference `5.16e-5`.
Fixed-stream floating differences arise from differently grouped vector
arithmetic and will be followed by end-to-end standard-problem calibration in
the implementation PR.

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

Raw mechanism records are in `suite_final.json`. End-to-end accuracy and
performance use the maintained `benchmarks/issue_247/run_current_standard.py`
and `run_matrix.sh` drivers so the candidate and `develop` receive identical
standard-problem configurations and termination conditions.
