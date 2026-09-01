# Frozen thread-scheduler evidence

## Question

Can one allocation round materialise and advance the race-tree work once,
instead of rebuilding population-sized coordination after every
replacement batch, without changing constrained-prior or race-tree semantics?

## Retained architecture

### Allocation and depth ownership

A planning boundary builds the likelihood blocks, expected classic volume
path, depth-relevant prefix, allocation target, and allocation gap. Uniform
allocation freezes the absolute target
`d0 * delta_K * allocation_iteration`. Utility allocation freezes the direct
work request `ceil(delta_K * unit_peak_utility)` on the current race.

The gap is decomposed into compressed maximal `T(a, b)` runs. The compiled
replacement loop advances those frozen logical threads; it does not recompute
the full race, utility, or expected-volume path after every batch. If newly
inserted contours expose a gap before expected depth is reached, the same
absolute target is projected onto the refined contour grid. Only an empty
projected gap advances the internal allocation iteration. Only actual expected
depth completion reaches the user-provided Python goal.

`DepthCondition` therefore contains only `dlogZ` and `cummax_XL_frac`. Both use
the expected classic shrinkage path at planning or drain boundaries. Final
Monte Carlo evidence and richer user policy remain outside the repeated JAX
loop. When all observed likelihoods are negative infinity, the remaining
fraction is deliberately one: absence of finite support is ignorance, not a
scientific reason to stop exploring.

### Thread execution

Compressed runs are opened shallowest first so a wider allocation spreads
across the constrained population before deep survivors are reinforced. A
fixed-width head window maps exactly to replacement sampler lanes. Every newly
available logical start is admitted before a continuation. Unfinished chains
enter a growable minimum-effective-contour heap, so the next available lanes
advance the shallowest frontier without materialising every logical thread or
sorting the heap after each batch.

The head and continuation dimensions grow independently. Increasing
distributed worker capacity cannot shrink an already enlarged continuation
heap. Parent indices and thread identities are transient coordination data;
scientific samples retain only their parent likelihood contour and out-degree.

### Stationary seeds

At publication, samples are ordered by birth contour and indexed by likelihood
rank in a wavelet matrix. A query at contour `lambda` counts and selects exactly
the stored edges satisfying

`birth_contour <= lambda < sample_likelihood`

in logarithmic rank-query depth, without a population mask per lane. Root and
non-root contours use the same index. A value-independent bounded reservoir
represents accepted rows since publication until they enter the complete
index.

New starts at one effective contour use a randomized permutation of one
systematically rotated rank lattice. This preserves uniform marginal seed
selection, spreads finite lanes across the eligible population, and avoids
coupling lane contour order to likelihood rank. An exact growable
open-addressed reservation set prevents same-contour seed reuse across local
batches and pending distributed work. Its scalar generation performs a
logical clear when that contour group finishes. Thus reservation memory follows
the unfinished distinct-seed group rather than total sample capacity.

### Publication and growth

A seed source becomes publishable after the larger of four reservoir windows
and 25% growth of the published population. Successive population-sized index
builds are therefore geometric, so their cumulative input size is linear in
the final population apart from sorting factors.

Publication bulk-merges every accepted identity into the persistent likelihood
order. The fixed merge bound is

`max(ceil(capacity / 4), 4 * reservoir_width) + dispatch_width`,

capped by capacity. This covers both refresh criteria plus one dispatch
overshoot; the source watermark cannot pass an unmerged row. Publication
refreshes only stationary-seed data. Frozen targets, compressed runs, current
heads, and continuations survive unchanged.

Sample capacity, head width, continuation heap, exact reservations, and the
wavelet height can each grow at a Python coordination boundary. These returns
do not advance the random stream, allocation iteration, depth completion, or
user goal. `State` is the single owner of `ThreadSchedule`; planning does not
return a duplicate schedule Pytree across the device boundary.

## Reproducible coordination benchmark

`run_scheduler.py` constructs a valid deterministic race, removes likelihood
cost with a deterministic constrained sampler, and makes both implementations
accept exactly 800 rows in ten width-80 batches. Every timing is synchronized.
The candidate is warmed twice and measured six times. `develop` is also warmed
twice; its population recurrence is slow enough that one subsequent execution
is reported. Lowering and compilation are separate from warm device execution.
All final measurements below use JAX 0.11.1, x64, float64 measure arrays, and
the CPU backend.

At 53,034 entry samples:

| path | lower (s) | compile (s) | warm median [range] (s) | HLO bytes | argument bytes | output bytes | temporary bytes |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `develop` repeated depth | 0.660 | 1.290 | 35.738 | 253,212 | 2,584,113 | 2,584,274 | 10,346,504 |
| candidate planning | 0.274 | 0.790 | 0.064 [0.060, 0.083] | 84,860 | 2,584,122 | 16,079,927 | 2,585,032 |
| candidate repeated depth | 0.732 | 2.087 | 0.228 [0.203, 0.237] | 384,245 | 16,079,310 | 16,079,927 | 91,672 |
| candidate planning + depth | -- | -- | 0.291 [0.264, 0.320] | -- | -- | -- | -- |

The candidate repeated depth is 156.7 times faster than `develop`; including
its once-per-round planning cost it is 122.7 times faster. Repeated-depth
temporary memory falls 99.1%. The explicit frozen schedule raises total
argument, output, and temporary executable bytes from 15.5 MB to 32.3 MB at
this population, and candidate planning plus depth compilation totals 2.88 s
versus 1.29 s for `develop`. These costs are stated explicitly: compilation is
paid at a physical shape boundary, while the eliminated coordination recurred
for every replacement batch.

Candidate scaling is:

| entry samples | repeated depth median [range] (s) | planning median [range] (s) | end-to-end median [range] (s) | repeated temporary bytes | planning output bytes |
|---:|---:|---:|---:|---:|---:|
| 53,034 | 0.228 [0.203, 0.237] | 0.064 [0.060, 0.083] | 0.291 [0.264, 0.320] | 91,672 | 16,079,927 |
| 424,272 | 0.257 [0.252, 0.269] | 0.511 [0.497, 0.542] | 0.768 [0.753, 0.797] | 91,672 | 131,809,739 |

An 8.00-times larger population costs 1.13 times more in the repeated depth
path, 7.94 times more in once-per-round planning, and 2.64 times more end to
end. Population work is now isolated at the geometric planning/publication
boundaries and has the intended approximately linear relationship; it is not
multiplied by every replacement batch.

The compact reservation set was selected against the earlier exact
sample-capacity mask. At 53,034 and 424,272 samples it reduced repeated depth
from 0.249 to 0.226 s and from 0.302 to 0.274 s in the controlled comparison.
At the larger size, end-to-end time fell from 0.985 to 0.832 s, and compiled
temporary memory fell from 3.49 MB to 0.091 MB. Adversarial width-80 schedules
selected exactly 80 distinct seeds before reuse at both populations.

## Diagnostic-reference bottleneck

The unchanged standard gate computes a high-resolution truth for one problem.
Its default had applied independent grid cells through scalar `lax.map`, so a
500 by 500 truth grid could dominate or appear to stall the entire scientific
gate before nested sampling started. The default now uses `vmap`; callers may
still request an explicit `batch_size` for bounded peak memory.

On the same 200 by 200 grid, both paths return exactly
`-3.490996085294162`. `develop` takes 16.796 s and the vectorized path takes
0.235 s, a 71.5-times reduction. A direct test compares vectorized and bounded
results, transformed samples, and posterior log weights to their floating-point
reduction precision.

## Scientific and end-to-end evidence

The standard-problem file and its tolerances are byte-for-byte identical to
`develop`. All 22 cases pass on commit `5af0892`, covering eleven problems with
phantom collection both disabled and enabled; the synchronized local run took
461.74 s. The long-form workload records are inserted here only after the
committed candidate completes them.

The final workload records, per goal boundary, allocation iteration, classic
sample count, likelihood-evaluation count, expected evidence uncertainty, and
elapsed wall time. This is the acceptance check for practical completion and
for non-pathological scaling beyond the synthetic benchmark.

## Rejected alternatives

- Rebuilding blocks, utility, expected depth, seed masks, and likelihood order
  after every replacement batch is the measured population-sized repeated
  path on `develop`.
- Selecting only current heads is invalid because head survival adds censoring
  below the logical thread terminal.
- Letting newly accepted descendants immediately seed new logical starts
  reinforces surviving genealogies and failed multimodal scientific gates.
- Holding one frozen source indefinitely made phantom clusters genealogically
  stale; publishing after a fixed number of rows instead makes cumulative
  population rebuilds quadratic. Four reservoir windows plus 25% growth was
  the smallest passing geometric policy tested.
- A FIFO continuation queue makes dispatch order depend on completion history.
  A minimum-effective-contour heap advances the scientific frontier directly.
- A full-capacity continuation heap or exact reservation mask makes every
  repeated scatter and output depend on maximum sample capacity. Both were
  slower and used materially more compiled memory than growable compact data.
- Sampling eligible ranks independently per lane can correlate likelihood rank
  with lane contour order. The randomized systematic lattice improved the
  30-seed Jones RMSE from 0.243 to 0.200 while using 1% fewer likelihood
  evaluations in the paired control.
- Replacing vectorized independent work with scalar `lax.map` avoids peak
  materialisation but makes cheap scientific references scale with every
  scalar dispatch. Explicit bounded batching remains available when memory,
  rather than elapsed time, is the constraint.
