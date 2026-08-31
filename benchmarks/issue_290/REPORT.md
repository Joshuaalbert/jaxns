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

A bounded, value-independent random reservoir admits rows accepted after the
freeze. Seeds are sampled from the union of that reservoir and the complete
frozen race, with exact stationary eligibility checked for every proposal.
Same-contour thread starts mark each used frozen sample directly, so distinct
eligible seeds are exhausted before reuse even when one start group is wider
than the physical sampler batch. After 32 reservoir-width windows, all
accepted rows become a new frozen source and the unchanged absolute target is
projected onto the refined contours.
The continuation ring has `33R` slots, where `R` is the greater of execution
width and root degree: before refresh there can be fewer than `32R` committed
continuations and no more than `R` in-flight continuations. Its memory is thus
independent of the number of samples and the integer allocation magnitude.

## Coordination protocol

`run_scheduler.py` constructs a valid deterministic race and removes
likelihood cost with a deterministic constrained sampler. Develop and the
candidate use width 80, receive the same allocation increment, and accept the
same rows. Each executable is warmed twice; six synchronised CPU runs then
determine the median and full observed range. Lowering and compilation are
measured separately. The environment used JAX 0.10.0 on CPU.

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

The bounded source refresh was also measured over longer schedules at 53,034
entry samples:

| accepted batches | implementation | schedule calls | warm median [range] (s) |
|---:|:---|---:|---:|
| 50 | `develop` | 1 | 2.696 [2.658, 2.748] |
| 50 | frozen FIFO | 2 | 1.349 [1.328, 1.378] |
| 100 | `develop` | 1 | 5.913 [5.815, 6.121] |
| 100 | frozen FIFO | 4 | 2.748 [2.711, 2.790] |

The candidate is 2.00x faster over 50 batches and 2.15x faster over 100
batches. Refresh projects the same target rather than exposing an intermediate
state to the user goal. Doubling work from 50 to 100 batches costs 2.04x for
the candidate and 2.19x for develop; the candidate does not reintroduce the
previous growing post-freeze scan.

The final reservation design was also remeasured against `develop` under JAX
0.11.1, the local environment used for the release-gate and paper runs. This
version materially penalises the repeated population-scale coordination in
`develop`, so the comparison is reported separately rather than mixing JAX
versions:

| samples at entry | implementation | lower (s) | compile (s) | warm 10-batch median [range] (s) | HLO bytes | temporary bytes | total executable bytes |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 53,034 | `develop` | 0.771 | 1.379 | 42.623 [41.699, 42.814] | 232,401 | 9,483,584 | 13,790,476 |
| 53,034 | frozen FIFO | 1.090 | 2.772 | 4.253 [4.146, 4.296] | 341,793 | 3,487,536 | 14,860,896 |

Both implementations again accept exactly 800 rows. The final candidate is
10.02x faster on the repeated path and uses 63.2% less compiler temporary
memory. Its explicit schedule and one-bit-per-frozen-sample reservation mask
raise total executable bytes by 7.8%, HLO text by 47.1%, and compile time by
2.0x. The six post-warmup timings have non-overlapping observed ranges.

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
bound. On the unchanged phantom-enabled release case, whose reference is
`log Z = -23.07212`, the deterministic estimate is
`-23.09034 +/- 0.21708` and the 1,000-draw phantom-conditioned estimate is
`-23.07795 +/- 0.13577` (`z = -0.043`). The run records 7,571 classic samples,
72,710 phantoms, and 1,315,320 likelihood evaluations.

## Rejected alternatives

- Rebuilding likelihood blocks, allocation gaps, expected depth, and
  likelihood order after every replacement batch is the population-linear
  develop path.
- Restricting seeds to ongoing heads is invalid: survival of a head censors
  its likelihood below the thread terminal.
- Using only the initially frozen seed population caused mode loss on the
  unchanged high-dimensional tests.
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
