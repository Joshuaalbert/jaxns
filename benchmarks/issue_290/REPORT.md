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
`d0 + delta_K * allocation_iteration`. Utility allocation freezes the direct
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

While one schedule is active, its seed source becomes publishable after the
larger of four reservoir windows and 25% growth of the published population.
Successive population-sized index builds within that schedule are therefore
geometric. Once a projected schedule drains, all newly stationary classic
samples are published before its exposed gaps are filled.

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

At 53,034 entry samples, rerun from the final candidate source:

| path | lower (s) | compile (s) | warm median [range] (s) | HLO bytes | argument bytes | output bytes | temporary bytes |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `develop` repeated depth | 0.530 | 0.928 | 20.980 | 253,212 | 2,584,113 | 2,584,274 | 10,346,504 |
| candidate planning | 0.224 | 0.522 | 0.048 [0.043, 0.050] | 85,031 | 2,584,122 | 16,079,943 | 2,585,032 |
| candidate repeated depth | 0.624 | 1.515 | 0.161 [0.157, 0.165] | 387,341 | 16,079,318 | 16,079,943 | 91,680 |
| candidate planning + depth | -- | -- | 0.202 [0.197, 0.214] | -- | -- | -- | -- |

The candidate repeated depth is 130.5 times faster than `develop`; including
its once-per-round planning cost it is 103.9 times faster. Repeated-depth
temporary memory falls 99.1%. The explicit frozen schedule does increase
compiled argument and output buffers, and planning plus depth compilation
totals 2.04 s versus 0.93 s for `develop`. Compilation is paid at a physical
shape boundary, while the eliminated coordination recurred for every
replacement batch.

Candidate scaling is:

| entry samples | repeated depth median [range] (s) | planning median [range] (s) | end-to-end median [range] (s) | repeated temporary bytes | planning output bytes |
|---:|---:|---:|---:|---:|---:|
| 53,034 | 0.161 [0.157, 0.165] | 0.048 [0.043, 0.050] | 0.202 [0.197, 0.214] | 91,680 | 16,079,943 |
| 424,272 | 0.212 [0.205, 0.230] | 0.438 [0.405, 0.468] | 0.647 [0.601, 0.699] | 91,680 | 131,809,755 |

An 8.00-times larger population costs 1.32 times more in the repeated depth
path, 9.21 times more in once-per-round planning, and 3.21 times more end to
end. Population work is now isolated at planning and active-schedule geometric
publication boundaries; it is not multiplied by every replacement batch.

Reservoir selection also no longer materialises one proposal-by-reservoir
boolean matrix for every sequential sampler lane. `run_seed_selection.py`
compares the exact dense reference with a cumulative-count rank lookup. It
alternates execution order, warms twice, synchronizes every call, and uses the
same JAX 0.11.1 CPU/x64/float64 environment:

| proposal width | reservoir size | dense median [range] (s) | compact median [range] (s) | speedup | dense HLO bytes | compact HLO bytes | dense temporary bytes | compact temporary bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 300 | 0.000473 [0.000367, 0.000534] | 0.000229 [0.000203, 0.000287] | 2.06x | 2,184 | 5,129 | 5,136 | 81,992 |
| 64 | 3,000 | 0.003184 [0.002757, 0.003691] | 0.000371 [0.000296, 0.000438] | 8.59x | 2,198 | 5,144 | 5,136 | 81,992 |

The compact lookup intentionally trades 76.9 KB of extra compiled temporary
memory and about 3 KB of HLO text for time that no longer grows with the dense
proposal-by-reservoir product. Parameterized tests compare every selected slot
with the former dense reference for empty, sparse, dense, and out-of-range
ranks.

Finally, the compiled depth loop accepts only values it consumes at device
runtime. `run_depth_specialisation.py` compares it with an exact legacy wrapper
that also marks shell size, allocation target, root degree, and allocation
increment static even though they do not occur in the lowered computation.
For same-shape schedule values, the current cache stays at one executable and
its second call takes 0.0010 s. Changing only those unused legacy policy values
grows the legacy cache from one to two and takes 1.722 s for the second
compilation. Current and legacy HLO text are effectively identical (383,025
and 383,036 bytes), confirming that the removed specialization was compile
cache churn rather than runtime work. This measurement also uses JAX 0.11.1,
CPU, x64, and float64 measure arrays.

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
`develop`. All 22 cases pass after the final direction-state refactor, covering
eleven problems with phantom collection both disabled and enabled; the final
synchronized local run took 352.23 s. The focused core, sampler, state, and
distributed-runtime set passes 132 tests in 397.15 s.

A separate staged GMM CI test runs isotropically to expected log-evidence
uncertainty 0.2, proves that the explicit fit does not call the user likelihood,
and resumes the same state with fitted directions to 0.1. It requires the final
evidence error to be less than three times the uncertainty claimed by that
resumed state. It is deliberately outside the unchanged standard suite.

The unchanged basic standard problem was also run with the same 30 integer
seeds on exact `develop` and candidate sources. Each result used 1,000 classic
Monte Carlo evidence draws. Standard errors quantify the finite-seed bias and
RMSE measurements:

| implementation | bias +/- SE | RMSE +/- SE | SD of error / MC uncertainty | mean classic samples | mean likelihood evaluations | cold core (s) | warm core median (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `develop` | -0.00273 +/- 0.00722 | 0.03900 +/- 0.00537 | 0.924 | 248.4 | 5,081.0 | 3.043 | 0.0181 |
| candidate | -0.00209 +/- 0.00723 | 0.03900 +/- 0.00525 | 0.933 | 197.6 | 3,295.1 | 5.252 | 0.0171 |

Accuracy and calibration are statistically indistinguishable. The candidate
uses 35.1% fewer likelihood evaluations and 20.5% fewer classic samples. Its
first compile is 73% slower, while its median warm core execution is 5.9%
faster. A few candidate seeds cross a new physical capacity and compile that
shape; those events appear in its warm range but do not alter the scientific
record or median steady execution.

Distributed completion order was tested separately. FIFO and reversed
completion orders submit identical requests and produce exactly equal leaves
in the trimmed scientific state. This establishes arrival-order invariance;
it does not claim bitwise equivalence between local batched and distributed
scalar execution, whose random-key topologies differ while targeting the same
transition law.

### Explicit direction staging

Fitted directions are state-owned and entirely user driven:

1. `state.fit_gmm_directions()` fits or warm-refines from all stored classic
   `(U, log L)` observations and enables the retained law.
2. `state.iso_directions()` uses exact isotropic directions without discarding
   a fit.
3. `state.gmm_directions()` re-enables a retained fit and fails visibly when no
   successful fit exists.

No fit, likelihood probe, or automatic staging occurs inside local or
distributed depth execution. A distributed state permits fitting or toggling
only after all in-flight work and scheduling state have drained. The default
first fit uses one component; additional components are an explicit scientific
choice rather than an automatic partition of a single mode.

At `D=8`, 4,096 stored rows, one component, and ten EM iterations, an explicit
Python-boundary fit takes 7.97 ms versus 8.17 ms inside a device conditional.
The no-fit Python path takes 0.088 ms versus 0.215 ms for the device
conditional, with slightly smaller HLO and temporary memory. This small timing
difference supports the simpler explicit boundary, while user control and the
absence of hidden likelihood work are the primary reasons for it.

Component selection uses the covariance ellipsoid as a proxy. Stored
likelihood values fit the value at each component mean; a component is eligible
only above the requested contour. Its selection weight is its fitted ellipsoid
volume trimmed to that contour. It does not use an empirical assigned-sample
maximum, a sample-enclosing hull, or new likelihood evaluations.

When a state retains a fit but `iso_directions()` disables it, one shared JAX
conditional selects the true isotropic runtime branch outside the chain vmap.
The disabled branch is key-stream identical to a state with no fit. The cost
of retaining the alternate compiled branch is visible in the controlled
benchmark below (two warmups and 12 alternated measurements):

| path | plain runtime (ms) | retained-disabled runtime (ms) | runtime ratio | plain HLO bytes | retained HLO bytes |
|:---|---:|---:|---:|---:|---:|
| continuation initializer, 8 chains x 32 slices | 0.446 | 0.497 | 1.11x | 706,154 | 1,513,111 |
| complete reference, 2 chains x 8 slices | 0.097 | 0.129 | 1.33x | 1,201,842 | 2,994,361 |

New states with no fit retain the smaller pure-isotropic executable. Keeping a
fit while toggled off therefore avoids GMM device work, but intentionally pays
compile size for both user-selectable branches.

### Correlated-Gaussian canary

One deliberately difficult seed of an eight-dimensional Gaussian with 0.99
off-diagonal correlation was followed through the staged interface. Before any
fit, at uncertainty 0.190, its isotropic prefix already had log-evidence error
-2.35. A one-component fit at 0.2 and continuation to 0.1 reduced the error to
-1.02. Continuing to 0.05 produced error -0.415 with 214.57 million likelihood
evaluations. Explicitly refitting at 0.1 reduced total work to 205.06 million
evaluations, but the final error remained -0.430.

This seed is not evidence that fitted directions remove constrained-chain
stationarity error. The retained pre-change isotropic and GMM states for the
same seed both ended near error -0.305 at reported uncertainty 0.05. The useful
conclusions are narrower:

- the new scheduler advances regularly to 424k samples rather than developing
  a repeated full-history timing wall;
- later explicit fitting improves the proxy and reduces likelihood work by
  about 4.4% in this seed;
- neither scheduling nor a symmetric direction law can retroactively repair a
  badly mixed prefix, so multi-seed calibration remains the scientific test.

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
- Preserving the old seed index after a frozen schedule drained made its next
  projected boundary about 1.5 times faster, but failed three of 22 unchanged
  fixed-seed standard cases. On Jones it shifted the evidence outside the
  allowed error and increased likelihood work from 154,494 to 266,910. The
  retained design instead publishes all newly stationary classic clusters at
  that scientific boundary.
- Replacing vectorized independent work with scalar `lax.map` avoids peak
  materialisation but makes cheap scientific references scale with every
  scalar dispatch. Explicit bounded batching remains available when memory,
  rather than elapsed time, is the constraint.
