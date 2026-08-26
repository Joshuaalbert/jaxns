# Issue 262: pure streaming GMM investigation

## Decision

Do not adopt the pure streaming update in JAXNS.

The candidate made an individual update much cheaper and improved point
estimates of mode-mass RMSE, but it was not a Pareto improvement over the
warm, bounded-replay implementation from issue 246. It never reduced mean
likelihood evaluations on a non-trivial standard problem and increased them
by 2.1% to 29.3%. Evidence cost-error was materially worse on `spike_slab`
and `weak_curved_spike_slab10` without phantom conditioning. The complete
depth program was also larger and slower to compile.

The exact tested candidate is preserved in commit
`f03551600f5731e4e8f233b5f733fffce4105bb6`. The production path was removed
after the evidence review, so the warm implementation remains the only
ellipsoidal update policy in the branch tip.

## Statistical target tested

A fixed-size online state cannot reproduce the warm fit's current expected
posterior weights. Adding a race-tree edge changes old block weights
non-uniformly, so old component statistics cannot be corrected exactly
without storing and replaying observations.

The candidate therefore used an explicit alternative target:

- one equal contribution per accepted classic sample;
- cumulative component mass, squared mass, first moment, and second moment;
- soft responsibilities evaluated once when a sample arrived;
- no decay or unexplained forgetting;
- a bounded initial EM fit, followed by one update per replacement batch;
- strict component eligibility by observed likelihood peak;
- ellipsoid-volume component selection and 1% isotropic safety fallback.

This state has fixed storage complexity `O(K D^2)`. Tests covered a NumPy
sufficient-statistic reference, later mode discovery, singular populations,
empty geometry, transparent capacity growth, checkpoint/resume, and frozen
slotted pytree behavior.

## Protocol

- 10 standard problems.
- Isotropic, warm ellipsoidal, and streaming ellipsoidal directions.
- Phantom retention off and on.
- 30 paired independent seeds per matrix cell (`n=30`).
- 1,800 runs in total.
- Identical termination, root allocation, replacement width, slice count,
  random seeds, and 1,000 MC evidence draws across policies.
- `root_degree = 30 D`, `replacement_width = 10 D`, and
  `num_slices = 5 D`.
- JAX/JAXlib 0.10.0, CPU backend, x64, Python 3.12.9.
- Phantom-on/off cells were required to have identical classic sample count,
  likelihood evaluations, deterministic evidence, ESS, posterior metrics,
  and direction-update count for each seed.

The raw records are in [`raw/`](raw/), full absolute summaries are in
[`results.md`](results.md), and paired 95% bootstrap comparisons against both
references are in [`comparisons.md`](comparisons.md). The CSV files include
the complete intervals and Bonferroni family intervals.

## Per-problem marginal value over warm replay

`RMSE off/on` is streaming divided by warm for MC evidence with phantoms off
and on. `Cost off/on` is the corresponding ratio of
`RMSE^2 * likelihood evaluations`. Posterior and mode ratios below one favor
streaming. Direction efficiency and classic posterior metrics are identical
between phantom-off and phantom-on runs.

| Problem | Delta classic bias | Classic RMSE ratio | MC RMSE off/on | Likelihood eval delta | Steady time off/on | Cost off/on | Posterior mean ratio | Mode-mass ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| basic | 0.0000 | 1.000 | 1.000 / 1.000 | 0.0% | -29.6% / -10.8% | 1.000 / 1.000 | 1.000 | -- |
| basic2 | +0.0020 | 0.912 | 0.907 / 0.892 | +0.1% | -8.4% / +5.5% | 0.824 / 0.797 | 0.956 | -- |
| basic3 | +0.0076 | 0.834 | 0.837 / 0.696 | +29.3% | +1.9% / +5.6% | 0.906 / 0.626 | -- | -- |
| plateau | 0.0000 | 1.000 | 1.000 / 1.000 | 0.0% | +19.5% / +11.4% | 1.000 / 1.000 | 1.000 | -- |
| basic_mvn | -0.0085 | 0.866 | 0.869 / 0.748 | +19.8% | +0.2% / +1.1% | 0.906 / 0.671 | 1.017 | -- |
| spike_slab | -0.0489 | 1.178 | 1.188 / 1.094 | +3.1% | -10.3% / -9.2% | 1.455 / 1.233 | 0.648 | 0.651 |
| spike_slab10 | -0.0326 | 0.959 | 0.955 / 0.914 | +10.9% | -0.9% / -2.9% | 1.011 / 0.926 | 0.862 | 0.860 |
| weak_curved_mvn8 | +0.0900 | 0.803 | 0.805 / 1.019 | +2.1% | -23.5% / -17.4% | 0.662 / 1.060 | 1.094 | -- |
| weak_curved_spike_slab8 | +0.0253 | 0.972 | 0.969 / 0.896 | +4.0% | -15.6% / -11.2% | 0.977 / 0.834 | 0.773 | 0.764 |
| weak_curved_spike_slab10 | +0.0137 | 1.200 | 1.215 / 0.923 | +4.5% | -1.9% / -5.3% | 1.542 / 0.890 | 0.829 | 0.828 |

The median likelihood-evaluation increase over warm was 3.54%; all eight
non-trivial non-plateau problems increased. CPU time often decreased
because a streaming update is cheap, but likelihood evaluation count is the
more portable cost measure and is especially relevant to future distributed
or expensive scientific likelihoods.

No Bonferroni family interval excluded zero for the deterministic classic
bias difference or classic RMSE difference across the 20 warm-comparison
rows. Thus the experiment does not demonstrate a systematic bias change.
One nominal 95% bias interval missed zero, which is not persuasive in this
comparison family. Calibration and coverage results for every cell remain in
the full results table.

Streaming improved all four multimodal mode-mass point ratios over warm
(`0.651` to `0.860`), but warm was itself worse than isotropic on two of those
problems, and the improvement came with more likelihood evaluations. Only the
8D `spike_slab` mode-mass interval clearly excluded no change at nominal 95%.
This is useful evidence about the warm model, but not enough to justify the
streaming target and hotpath complexity.

## Isolated update programs

The isolated update evidence is stored in
[`update_programs.jsonl`](update_programs.jsonl). Times are medians of 20
steady executions after explicit device synchronization.

| Dimension | Warm refresh | Streaming increment | Time ratio | Warm temporary bytes | Streaming temporary bytes | Memory ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.208 ms | 0.190 ms | 0.086 | 215,632 | 3,752 | 0.017 |
| 8 | 6.473 ms | 0.186 ms | 0.029 | 800,912 | 46,312 | 0.058 |
| 10 | 10.541 ms | 0.302 ms | 0.029 | 999,952 | 72,232 | 0.072 |
| 20 | 20.592 ms | 1.633 ms | 0.079 | 2,002,832 | 288,232 | 0.144 |
| 40 | 61.850 ms | 7.053 ms | 0.114 | 4,046,928 | 1,152,232 | 0.285 |

This local result is real, but a run performs roughly 49 to 79 streaming
updates on the multidimensional standard problems versus about 2 to 3 warm
refreshes. It therefore does not predict end-to-end value on its own.

## Complete depth program

The `basic_mvn` depth-program evidence is stored in
[`depth_programs.jsonl`](depth_programs.jsonl).

| Policy | Lower | Compile | HLO bytes | Temporary bytes | Median depth execution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isotropic | 0.976 s | 2.544 s | 445,808 | 1,047,544 | 1.357 s |
| Warm | 1.404 s | 3.553 s | 611,323 | 1,478,544 | 1.550 s |
| Streaming | 1.481 s | 3.907 s | 644,612 | 1,457,760 | 1.577 s |

Relative to warm, streaming increased HLO size by 5.4% and compile time by
10.0%, reduced whole-program temporary memory by only 1.4%, and did not
improve depth execution. Process peak RSS was mixed across problems and did
not show a consistent streaming advantage.

## Intent review

The candidate added persistent online moments and a second direction-update
schedule to the compiled core. It required roughly 480 production lines in
the sampler state and depth hotpath, plus new numerical edge cases around raw
moment cancellation, component identity drift, singular covariance, and peak
ownership. Those costs would be justified by a strong scientific or
performance frontier improvement; the measurements do not show one.

The warm bounded replay remains preferable because it:

- uses the current expected posterior weights;
- can recompute responsibilities after race-tree weights change;
- updates only when contour eligibility makes the current geometry stale;
- has much less state and scheduling intent to maintain;
- used fewer likelihood evaluations in every non-trivial comparison.

## Reproduction

The candidate must be checked out because it is intentionally absent from the
branch tip:

```bash
git worktree add /tmp/jaxns-262-candidate f03551600f5731e4e8f233b5f733fffce4105bb6
cd /tmp/jaxns-262-candidate
PYTHONPATH=/tmp/jaxns-262-candidate/src:/tmp/jaxns-262-candidate \
  MPLCONFIGDIR=/tmp/mplconfig \
  conda run -n jaxns_py python benchmarks/issue_262/run_suite.py \
  --output-dir /tmp/jaxns-262-raw --quiet
```

The committed summaries can be regenerated from the committed raw records:

```bash
conda run -n jaxns_py python benchmarks/issue_262/summarise.py \
  benchmarks/issue_262/raw \
  --csv /tmp/results.csv \
  --markdown /tmp/results.md \
  --comparisons-csv /tmp/comparisons.csv \
  --comparisons-markdown /tmp/comparisons.md
```
