# Frozen scheduler evidence

## Question

Does freezing and advancing compressed logical threads remove repeated
population-scale coordination without changing scientific accuracy?

## Retained architecture

One planning round freezes the absolute allocation target, expected-volume
depth prefix, block index, and compressed maximal `T(a, b)` thread runs. An
80-slot execution window starts every newly materialised logical thread before
resuming old heads from a FIFO continuation ring. This breadth-first rotation
spreads work across the stationary population without widening the vmapped
constrained sampler merely to hold every logical head.

A new logical start samples without replacement from the complete stationary
population frozen at planning time. Allowing a newly accepted child to open
another thread in the same generation creates genealogical mode reinforcement
on non-mixing problems. Same-contour starts therefore mark used frozen sample
identities directly, so distinct eligible seeds are exhausted before reuse
even when one start group is wider than the physical sampler batch.

A continuation does not open another logical lineage and must not acquire a
stale seed law merely because its schedule is long. It samples from the exact
union of the frozen likelihood suffix and every append-only row accepted since
planning, rejecting candidates unless their birth contour and likelihood prove
stationarity at the requested contour. This is random access into two compact
ranges, not a scan of the accepted suffix. A small value-independent reservoir
provides only a conservative lower bound for simultaneous distinct-seed
reservations; reservoir membership never defines scientific eligibility.

A source publication becomes eligible after the larger of `4R` accepted rows,
where `R` is the greater of execution width and root degree, and 25% growth in
its frozen population. A schedule that fills its target earlier simply drains.
Successive long generations are geometric, so the sum of all O(N) race
publications within one allocation target is itself O(N), rather than becoming
quadratic under a fixed row cadence. Each internal publication projects the
unchanged absolute target onto refined contours and bulk-merges only its
accepted suffix into the persistent likelihood order. It does not evaluate the
depth condition or advance the user goal.

The continuation ring covers the same generation boundary plus one in-flight
window. Its static capacity is
`max(4R, ceil(storage capacity / 4)) + R`, which is sufficient across automatic
storage growth because the frozen source cannot exceed the capacity at which
its schedule was built. Block lineage counts use the exact additive exclusive
prefix of `-m_g + d_g`, avoiding one sequential device-loop iteration per
block.

The local runner grows every sample-indexed carry and the continuation ring at
one Python boundary. An interrupted schedule therefore resumes with the same
Pytree shapes that the next generation will use, avoiding a second compiled
specialisation for the same sample capacity.

## Coordination protocol

`run_scheduler.py` constructs a valid deterministic race and removes
likelihood cost with a deterministic constrained sampler. Develop and the
candidate use width 80, receive the same allocation increment, and accept the
same rows. Each executable is warmed twice; six synchronised CPU runs then
determine the median and full observed range. Lowering and compilation are
measured separately. The following historical experiment used JAX 0.10.0 on
CPU and predates the final seed-reservation and bounded-publication fixes.

| samples at entry | implementation | lower (s) | compile (s) | warm 10-batch median [range] (s) | HLO bytes | temporary bytes | executable bytes |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 53,034 | `develop` | 1.002 | 1.254 | 0.532 [0.527, 1.024] | 232,209 | 9,483,584 | 13,790,619 |
| 53,034 | frozen FIFO | 1.380 | 2.450 | 0.346 [0.343, 0.371] | 311,493 | 3,487,176 | 14,806,574 |
| 424,272 | `develop` | 1.138 | 1.451 | 4.348 [4.262, 4.437] | 240,612 | 74,820,736 | 108,826,811 |
| 424,272 | frozen FIFO | 1.261 | 2.557 | 3.032 [2.998, 3.080] | 315,535 | 24,044,040 | 112,952,180 |

Both implementations accepted exactly 800 rows. The retained repeated path is
1.54x faster at 53k samples and 1.43x faster at 424k. Compiler temporary
memory falls by 63.2% and 67.9%. Explicit continuation data raises argument +
output + temporary executable bytes by 7.4% and 3.8%. HLO text is 34.1% and
31.1% larger, and compilation is 2.0x and 1.8x slower. Compilation occurs once
per physical shape; coordination repeats throughout a run.

The historical fixed-cadence candidate was also measured over longer schedules
at 53,034 entry samples. These timings explain why frozen scheduling was kept,
but the final source-generation policy above supersedes its publication rule:

| accepted batches | implementation | schedule calls | warm median [range] (s) |
|---:|:---|---:|---:|
| 50 | `develop` | 1 | 2.696 [2.658, 2.748] |
| 50 | frozen FIFO | 2 | 1.349 [1.328, 1.378] |
| 100 | `develop` | 1 | 5.913 [5.815, 6.121] |
| 100 | frozen FIFO | 4 | 2.748 [2.711, 2.790] |

That candidate was 2.00x faster over 50 batches and 2.15x faster over 100
batches. It established that refresh can project the same target without
exposing an intermediate state to the user goal. The final geometric rule is
assessed separately below.

The final design was remeasured against `develop` under JAX 0.11.1, the local
environment used for the release-gate and demanding long-form runs. This
version materially
penalises the sequential per-block recurrence in `develop`, so the comparison
is reported separately rather than mixing JAX versions. The three `develop`
executions were stable at 42.269, 42.286, and 42.014 seconds; six final-candidate
measurements follow two warmups.

| samples at entry | implementation | lower (s) | compile (s) | warm 10-batch median [range] (s) | HLO bytes | temporary bytes | total executable bytes |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 53,034 | `develop` | 0.832 | 1.381 | 42.269 [42.014, 42.286] | 253,212 | 10,346,504 | 15,514,891 |
| 53,034 | frozen FIFO | 1.227 | 2.930 | 0.463 [0.460, 0.469] | 365,362 | 3,467,056 | 15,532,664 |

Both implementations again accept exactly 800 rows. The final candidate is
91.3x faster on the repeated path and uses 66.5% less compiler temporary
memory. Its explicit schedule, frozen-sample reservation mask, and sufficient
quarter-capacity continuation ring leave total executable bytes within 0.2%
of develop; HLO text grows by 44.3%, and compile time by 2.1x. The observed
execution ranges do not overlap.

Final-candidate scaling was then measured with the same six-after-two protocol:

| samples at entry | accepted batches | warm median [range] (s) |
|---:|---:|---:|
| 53,034 | 10 | 0.463 [0.460, 0.469] |
| 424,272 | 10 | 3.938 [3.884, 3.963] |

An 8.00x population increase costs 8.50x more time. The retained scheduler is
therefore near-linear in frozen population size rather than retaining the
measured superlinear device-loop behavior. The sufficient queue adds 27% at
53k and 33% at 424k versus the earlier fixed `33R` ring; this cost is reported
rather than hidden because it is required for the geometric source guarantee.

For `R=80`, reaching 53,034 and 424,272 samples requires 21 and 30 geometric
source publications respectively, versus 166 and 1,326 under the scientifically
passing but rejected fixed-`4R` cadence. The sum of frozen populations processed
is below 4.5 times the final population in both cases, matching the geometric
bound of at most five times final N. Publication bulk-merges the accepted
suffix into persistent likelihood order; it never re-sorts the full population.

### End-to-end allocation scaling audit

An instrumented high-accuracy G8 prefix used `d_0=240`, replacement width
80, 80 slice transitions, 72 retained phantoms, and the unchanged expected
depth threshold `dlogZ=log(1 + 10^-3)`. This exposed and then verified a
scientifically material boundary bug. Treating source publication as a drained
target made the samples at goals 1--10 grow as
`6,400, 8,000, 10,000, 12,560, 15,760, 19,760, 24,720, 30,960, 38,720,
48,400`: after the first goal, work was effectively 25% of all accumulated
samples rather than the requested fixed lineage increment.

With publication retained as an internal projection of the same absolute
target, the corresponding sequence is
`7,611, 15,854, 23,786, 31,897, 40,261, 47,939, 56,032, 64,433, 72,608,
80,883`. The increment remains approximately 8,000 samples per +240 lineage
allocation; it does not compound with population size. At goal 10 the corrected
run has performed 58,782,696 likelihood evaluations in 207.8 s and reaches
expected log-evidence uncertainty 0.10080. The invalid boundary had performed
only 25,818,439 evaluations in 129.0 s and remained at 0.12208, so its shorter
time represented omitted scientific work rather than an optimisation.

The same run crossed capacities 5,360, 10,720, 21,440, 42,880, and 85,760.
After the growth-shape fix, its continuation widths changed directly as
`1,580 -> 2,920 -> 5,600 -> 10,960 -> 21,680`; there is no intervening
old-width compilation at a new sample capacity. A complete run to the 0.05
0.05 uncertainty goal is reported below once the final source revision is
fixed.

## Scientific checks

The unchanged standard basic problem was run for ten paired seeds with the
same model, sampler, default stopping policy, and 1,000-draw classic MC
evidence calculation.

| implementation | log-evidence bias | RMSE | mean standardised error | SD standardised error | mean classic samples | mean likelihood evaluations |
|:---|---:|---:|---:|---:|---:|---:|
| `develop` | +0.0022 | 0.0410 | +0.116 | 1.063 | 248.4 | 5,117.3 |
| frozen FIFO | -0.0066 | 0.0425 | -0.092 | 1.090 | 212.1 | 3,848.7 |

Bias, RMSE, and standardised-error dispersion remain comparable at ten seeds;
the candidate uses 24.8% fewer likelihood evaluations. This small paired run
is supporting evidence rather than a new tolerance.

The release gate is the complete unchanged
`cicd/tests/test_ns_standard_problems.py`: all 22 cases pass, covering every
problem with phantom conditioning disabled and enabled. Its problem
definitions and tolerances are byte-for-byte unchanged from `develop`. The
difficult 8-D spike-slab, which exposed mode-weight loss in earlier scheduler
variants, passes in both modes.

The 10-D spike--slab exposed a second failure mode in the first frozen
scheduler: starts split across physical batches could reuse a small seed
subset. The final frozen-sample reservation mask removes that batch-width
bound. Jones then exposed the opposite failure: holding one source generation
fixed for too long left phantom clusters genealogically stale. The retained
4-window/25%-growth promotion rule passes both unchanged phantom gates:

| problem | truth | deterministic estimate | phantom MC estimate | phantom z | classic samples | likelihood evaluations |
|:---|---:|---:|---:|---:|---:|---:|
| Jones scalar | 34.80395 | 34.71417 +/- 0.23525 | 35.08204 +/- 0.14646 | +1.899 | 1,560 | 182,649 |
| spike--slab 10-D | -23.07212 | -23.20715 +/- 0.21620 | -23.11293 +/- 0.13366 | -0.305 | 6,400 | 901,442 |

This is not a random-key workaround: the source-generation policy is the
smallest tested rule that prevents within-generation descendant reproduction,
refreshes Jones soon enough, and retains linear cumulative publication work.

## Rejected alternatives

- Rebuilding likelihood blocks, allocation gaps, expected depth, and
  likelihood order after every replacement batch is the population-linear
  develop path.
- Restricting seeds to ongoing heads is invalid: survival of a head censors
  its likelihood below the thread terminal.
- Using only the initially frozen seed population caused mode loss on the
  unchanged high-dimensional tests.
- Admitting any post-freeze stationary row to a new start fixes Jones but
  reintroduces spike--slab mode reinforcement: its phantom estimate moves to
  `-22.42536` for truth `-23.07212`. Restricting admission to rows born at a
  strictly shallower contour produces the same failure, showing that the
  required separation is by source generation rather than contour depth.
- A source generation held for 32, 16, or 8 reservoir windows never promotes
  during Jones; its phantom estimate is `35.30745 +/- 0.14769`, or `z=+3.409`.
  A fixed 4-window cadence passes but would rebuild O(N) state after O(1) rows.
  Requiring 25% growth after the minimum retains the passing trajectory while
  making generation sizes geometric.
- Waiting for 100% population growth is slightly too stale on the deterministic
  Jones gate (`z=+2.035`). Carrying a full-capacity continuation ring is also
  rejected: at 53k it is 1.77x slower than the sufficient quarter-capacity
  bound and 2.24x slower than the earlier fixed ring.
- Keeping the old fixed `33R` ring and applying start backpressure avoids the
  larger carry, but a long deterministic schedule then resumes deep heads from
  a stale source and stationary-seed rejection effectively stalls. Source
  maturity and continuation capacity cannot be decoupled that way.
- Scanning every accepted post-freeze row preserved the scientific gates but
  made long schedules approach quadratic work: 100 and 200 batches at 53k
  samples took 1.705 s and 5.018 s in that experiment.
- Keeping only recently appended rows was fast (0.345 s and 0.644 s for 100
  and 200 batches), but membership carried genealogy information. It failed
  the phantom-conditioned Jones and spike-slab checks.
- Selecting directly from a random reservoir was also rejected: the 8-D
  spike-slab failed with phantoms both off and on. The retained sampler uses
  the complete frozen race as a fallback and checks exact stationarity.
- Advancing the same 80 heads until completion starved roughly 160 other root
  threads. It failed the phantom-conditioned 8-D spike-slab. Widening `vmap`
  to 240 made that check pass but raised likelihood evaluations from 334,789
  to 440,376 because all lanes wait for the slowest chain. The retained FIFO
  obtains breadth with the original width 80.
- Materialising every logical thread makes memory depend on gap magnitude.
  Compressed rise/fall runs emit at most one run per likelihood block; only
  the bounded active window and FIFO are materialised.
