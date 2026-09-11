# Efficient all-phantom seed selection: design and measured implementation

Revised design, 2026-09-10, after cancellation of the slow SS10 ladder.
The exact block index and lazy rank lookup are implemented and validated in
`650ae05`, with 30 paired SS10 one-goal checks at 0.02. The current PR ports
that sampler into develop while retaining develop's additive allocation.
[Implementation and evidence](../../benchmarks/phantom_seeding/README.md).
The body below records the original design; replay and hierarchical indexing
remain unimplemented follow-ups.
Implementation plan: [Ticket 0021](../implementation_orchestration/tickets/0021-efficient-all-phantom-seeding.md),
following the original [phantom-seeding tracking issue #292](https://github.com/Joshuaalbert/jaxns/issues/292).
The new ticket preserves all-A behavior and replaces #292's proposed bounded
reservoir as the first implementation candidate.

The first recommendation is **exact block indexing plus lazy candidate
lookup, initially retaining full phantom coordinates**. This addresses the
repeated work while preserving A's complete population and existing seed
choices. Coordinate replay is a later memory experiment, not a prerequisite.
The hierarchical rank index described below is a further option if block queries
still scale too slowly.

The initial proposal put the codec check first. That was the wrong priority
for the observed runtime problem: smaller coordinate records do not eliminate
eligibility scans and reconstruction adds work. The initial large-index
proposal also needs a simpler measured baseline before its engineering cost
is justified by A's modest scientific gain.

The reference implementation is `paper/evidence10-AC300` at
`aa79a0dc395c0d64b85cbb88d0e3e732decd7361`, in the adjacent
`jaxns-paper-evidence10-AC300` worktree. The main checkout currently has a
different core layout. Paths and function names below refer to that frozen
experimental source; this document does not imply that main already contains
A. The workers, dispatcher, and automatic report helpers were stopped at the
user's request on 2026-09-10 at 12:06:44 UTC. Completed 0.05/0.02 cohorts and
seven full-capacity 0.01 checkpoints are preserved. The ladder must not restart
without a new instruction; partial 0.01 trees are not completed results.

## What the evidence supports

The [archived 240-root comparison](paper-results/with-AC240/COMPARISON.md)
supports a modest, problem-dependent benefit from A. For its revised SS8,
classic mode RMSE falls from 0.0479 to 0.0315; for its revised CSS8, from
0.0405 to 0.0282. G8 classic evidence RMSE is almost unchanged, while CG8's
phantom-conditioned evidence RMSE is worse for A. These are historical 8D
problem definitions, not the current G10/CG10/SS10 suite.

Likelihood counts are close between A and R in that comparison. A's extra
expense is consequently an implementation concern, not a demonstrated saving
in likelihood work. Cohort wall times include different host loads and
compilation, so their ratios are not controlled selector benchmarks.

A previous [isolated selector benchmark](
/largedata/albert/git/jaxns-paper-ablation-A960/benchmarks/paper_reproduction/reports/validation/A/performance/README.md)
does demonstrate a relevant tradeoff: batching 80 contour queries over
185,985 rows with 79 phantoms reduced median time from 3.273 s to 0.465 s,
but increased compiler temporaries from 4.7 MB to 238.2 MB. All 400 tested
seed identities, coordinates, and likelihoods were bitwise equal. This
measures that particular selector change, not the proposal here.

Current 10D results use A+C; they do not independently measure A's contribution.
Phantom conditioning of evidence and phantom use as seeds must remain
separate comparisons. Posterior diagnostics still use classic samples.

Recent progress before cancellation showed increasing cost per added classic
as allocated capacity grew. This is consistent with the repeated capacity-wide
seed scans visible in source. It does not isolate their fraction of end-to-end
time: likelihood work, classic indexing, state copies, and checkpoints also
need separate timing.

## Recommended first design

### 1. Cache exact counts by blocks of chains

Partition append-order classic rows into fixed blocks of B rows, initially
benchmarking B=64, 256, and 1024. This groups storage, not geometry: no modes
or clusters are inferred. Keep every phantom identity and full U coordinate.

For each sealed block, store its sorted birth contours with cumulative valid
phantom counts and its sorted valid phantom likelihoods. If m_i is row i's
valid phantom count, eligibility at lambda is exactly:

`sum_i m_i * 1[b_i <= lambda] - sum_(i,j)_valid 1[ell_ij <= lambda]`.

Each valid interval satisfies b_i < ell_ij, which makes the subtraction exact.
Two binary searches yield a block's count without examining each chain.
Likelihood enters only these membership tests; it does not weight seeds.

Use the block cumulative counts to locate the block containing the existing
phantom proposal rank. Inspect that block's original likelihood/validity array
to find the row and transition slot in their original order. The selected
identity is exactly the identity returned by the current full-row cumulative
table. Gather its U only after reservation checks succeed.

Keep an exact unsealed tail of fewer than B rows and include it on every query.
At acceptance, seal any newly completed block and retain the remaining tail.
No phantom waits for sealing or classic publication to become visible.
Handle roots and invalid slots explicitly; births need not be monotone.

The block-sorted likelihoods replace the current row-sorted seed cache. They
must not become a third retained likelihood array. Original transition-order
likelihoods and validity remain available for evidence and local rank lookup.
The same block-count operation replaces the phantom-existence scan in
`_effective_parent_contour`, with its fallback policy unchanged.

| Cost | Current A | Block design |
|---|---|---|
| Batch cumulative table | `[S,C]` integers | `[S,ceil(C/B)]` integers, plus bounded tail/local scratch |
| Block/row count work | `O(S C log P)` | `O(S (C/B) log(BP) + S BP)` |
| Phantom rank lookup | Row search, then P slots | Block search, then at most BP slots |
| Sorted phantom likelihood storage | CP scalars | CP scalars plus O(C) birth/count metadata |
| Index construction across N rows | Per-row sorting | `O(NP log(BP))` with fixed B |

For the cancelled runs' capacity C=1,715,200, S=100 and B=256, an int64
cumulative table falls from **1.278 GiB to 5.11 MiB**. This is a 256-fold
reduction of that array, not a runtime or total-RSS prediction. Local scans
must have bounded scratch rather than a shell-by-proposals-by-block-by-slots
temporary.

The query still scales linearly in C/B. Across an ever-growing run, this
retains a quadratic accumulated term with a smaller coefficient. It is the
first performance candidate, not the final answer to unlimited-N scaling.
Measure the crossover to the hierarchical index below.

### 2. Resolve candidates only as needed

`_sample_stationary_seeds` currently generates 64 proposal fractions per
lane/attempt, resolves identities for all 64, and checks reservations before
taking the first admissible one. It also computes classic and phantom
candidates before choosing the population each rank belongs to. These are
source-observed costs; their share of runtime has not been profiled.

Keep the exact 64-value random draw and first-proposal strata override. Resolve
proposal zero first, in its actual classic/reservoir/phantom population. If it
is admissible, return that identity. Only a collision invokes the remaining
63 proposals, in their existing order; exhausting all 64 advances the same
attempt counter and uses the same next key.

This changes neither candidate fractions nor the first admissible identity.
Do not replace the 64-value draw with a scalar draw: JAX's shape-dependent
random stream would change. Do not relax same-contour reservations, change
exhaustion, or infer that unused random draws represent new independent keys.

The existing lane loop is scalar, so a scalar conditional can skip fallback
lookup. Check the lowered program: batching that conditional can evaluate
both branches. Delaying unused candidates is especially important with a
BP-sized local block scan. If collisions are frequent, benchmark modest
fallback tiles while preserving the logical proposal order.

### 3. Measure memory ownership before introducing replay

The seven saved full-capacity checkpoints are about 9.14 GiB each; resident
workers around the preceding status check occupied roughly 28–33 GiB each.
Those quantities are not directly interchangeable, but show why coordinate
record size alone is an inadequate process-memory model.

After removing large query temporaries, inspect buffers retained across the
compiled depth boundary, growth, classic-only views, and checkpointing.
Consider buffer reuse only where ownership permits it: ordinary state reuse
must not become invalid through an undocumented donation contract. Measure
peak simultaneous buffers, including checkpoint serialization and index build
workspace, before forecasting how many workers can run.

Keep the full-U representation as the exact first reference. Replay remains
useful if persistent coordinates then dominate, provided its numerical parity
and added decode cost pass the gates below. There is no promise that these
changes alone make the cancelled precision ladder practical.

## The behavior to preserve

For phantom slot `j` of accepted chain `i`, let `b_i` be its birth
log-likelihood constraint and `ell_ij` its stored log likelihood. Eligibility
at requested constraint `lambda` is exactly:

`valid_ij AND b_i <= lambda AND lambda < ell_ij`.

This is the existing stationarity-based eligibility condition. Recording
coordinates, steps, or likelihoods is not itself a selection rule. Neither
step length, likelihood height, spatial separation, age, nor a clustering
label may change a candidate's selection weight.

The storage optimization must preserve the following details:

- Every valid accepted phantom is immediately eligible under that predicate,
  including phantoms belonging to classics not yet published as seed sources.
- Population order remains frozen classics, the existing recent-classic
  reservoir, then phantoms in classic append order and original transition
  order. Identity remains logically `(classic_row, slot)`; slot zero is
  classic. The existing scalar encoding is `row * (P + 1) + slot`.
- The same random keys, lane permutation, shared rotated strata, collision
  proposals, and same-contour reservations select the same identities.
  Retained reservations across unfinished start groups remain valid through
  index rebuilding. Replacement starts under the existing exhaustion rule.
- Classic publication, the recent-classic reservoir, effective-contour
  fallback, thread scheduling, allocation, stopping, and likelihood accounting
  keep their current behavior. An index rebuild is not a seed publication.
- Phantom likelihoods retain their original chain membership, validity, and
  transition order for evidence conditioning. Phantoms do not become
  posterior observations or race-tree nodes.

The current sampler uses systematic stratification and conditional collision
retries. Equivalence therefore means preserving its ordered rank-to-identity
mapping and its random schedule, not merely producing a marginally uniform
candidate in a new order. Eligibility also does not make correlated phantom
states independent or establish new guarantees about adaptive reseeding.

No phantom can be retired simply because the current thread has passed its
likelihood. Later allocation can request a shallower contour. Removing one
would require proof that no future request can intersect its eligibility
interval; that proof is absent here.

## Where A currently pays

Let `N` be accepted classic rows, `C` allocated row capacity, `P` retained
phantoms per chain, `D` U-space dimension, `S` shell width, and `T` total
slice transitions. These are separate quantities; the current 10D runs use
`P=99, T=100, S=100`.

| Path in the experimental source | Current cost |
|---|---|
| `PhantomSamples.U_samples`, `samples.py` | Persistent `[C,P,D]` coordinates |
| Original likelihoods, sorted seed likelihood cache, validity | Two `[C,P]` likelihood arrays and one validity mask |
| `phantom_seed_cumulative_batch`, `sampling/phantom_seeds.py` | Approximately `O(S C log P)` work and a `[S,C]` integer cumulative table per selection batch |
| `phantom_rank_to_identity` | Finds the row, then scans only its original-order phantom likelihoods |
| `_effective_parent_contour`, `algorithm/depth.py` | An additional row scan for existence of an eligible phantom |
| Growth, generic state transformations, checkpoints | Can copy or serialize the large coordinate and cache arrays |

The existing batched selector already avoids a full `[S,C,P]` eligibility
mask. Replacing it with another such mask would undo a measured improvement.
Only final selected coordinates need decoding: collision proposals operate
on identities, not U vectors.

For illustration, with float32 U, float64 likelihoods, and one byte per valid
flag, the three phantom payloads above cost
`C * (4 P D + 17 P)` bytes. At D=10, P=99 this is 5,643 bytes per capacity
row, excluding classic state, scheduler/index metadata, alignment, and
temporaries. A `[100,1_000_000]` count table adds 0.373 GiB with 32-bit counts
or 0.745 GiB with 64-bit counts. Actual dtypes must be taken from each state.

These are array-size calculations, not predictions of process RSS. The
[current measured memory forecast](paper-results/evidence10-AC300-20260909/MEMORY_FORECAST.md)
includes much larger peaks from the complete execution and analysis paths.
It describes the cancelled worker pool, not a forecast for a changed
implementation.

## Later memory option: compact isotropic coordinate records

In `sampling/slice.py`, an ordinary accepted proposal has the form
`u_next = u_anchor + direction * t_accepted`. Shrinkage changes the scalar
proposal interval; it does not change that transition's direction. In
`sampling/continuation.py`, the accepted scalar is already available as
`SliceBatchState.t`.

The isotropic direction stream is determined by the logical chain key,
the U pytree structure/dtypes, the transition count, and the existing key
schedule. It does not depend on rejected likelihood values. This suggests
the following persistent record for each accepted chain:

| Field | Proposed shape / role |
|---|---|
| Original chain key | `[2]` uint32 for the current Threefry key representation |
| Accepted scalar step lengths | `[P]`, in their actual original dtype |
| Literal coordinate anchors | `[H,D]`, or the equivalent U pytree |
| Codec metadata | Version, sampler configuration, PRNG implementation, U structure/dtypes, and anchor rule; shared where constant |
| Existing phantom likelihoods and validity | `[P]` each, with their original meaning and order |

Use at most a fixed number `H` of anchors per chain. An illustrative
choice is H=8, with anchor spacing `q=ceil(P/H)`. Each anchor stores the
literal U immediately before that segment's first transition. In particular,
the initial anchor is a full coordinate, even when the chain was seeded
from an older phantom. An ancestry reference would create unbounded
recursive reconstruction.

After the existing selector chooses an identity, start from its nearest
preceding anchor, regenerate the required original directions, and replay
at most `q` accepted steps in order. Return the existing stored likelihood
alongside the reconstructed point. This performs no likelihood calls and
does not rerun bracket shrinking, rejection decisions, or the chain itself.
The original rejected proposal history need not be retained if the scalar
codec passes the exactness gate below.

Regenerating keys must reproduce the original `random.split` shapes for T,
including when only a prefix P was retained. Splitting into P keys or replacing
the schedule with `fold_in` is not equivalent. An initial implementation may
regenerate the T transition keys, then generate only the required directions.
Budget `O(T + q D)` arithmetic per decoded seed, plus bounded key buffers,
until an exactly equivalent indexed key derivation is established.

JAX documents deterministic reuse of a key with the same random operation,
but does not promise equivalence when scalar draws are replaced with a
different-shaped random draw. Reconstruction intentionally reproduces old
random values; it must never advance or alter the sampler's future key stream.
See the [JAX random-number documentation](https://docs.jax.dev/en/latest/random-numbers.html).

### Numerical identity is the gate

The algebraic expression alone is insufficient to claim a lossless codec.
Direction normalization, pytree reduction order, fused multiplication/addition,
and rounding of the stored scalar can change U bits. The existing continuation
code explicitly preserves transition ordering because batching reductions
has previously changed chain trajectories.

The encoder and decoder must reproduce the existing compiled point construction,
including its precision and operation order. Compare every retained point,
not only points subsequently selected. Equality must survive scalar and batched
execution, changed active-lane masks, and supported checkpoint resumes.

If the accepted scalar alone cannot reproduce the original expression,
evaluate a richer record containing the final proposal interval and its draw
key, with the original proposal primitive. Account for the additional bytes.
If numerical identity still fails, retain full coordinates for that supported
path rather than call a nearby floating-point trajectory equivalent to A.
The interval index can proceed independently of the coordinate codec.

The first codec should cover the experiments' nonperiodic isotropic,
straight-line, no-step-out sampler. Periodic charts require reproducing the
chart keys and both modular transforms. GMM directions require the exact
historical fitted geometry. Those paths keep full coordinates until separately
designed and validated; the initial codec should not create an archive of all
past GMM fits. No coordinate quantization is proposed.

### Memory and decode tradeoff

For D=10, P=99, float32 U and scalars, and one 8-byte chain key:

| Representation | Coordinate record bytes per row | Maximum replayed transitions | Including the current 1,683-byte likelihood/cache/mask payload |
|---|---:|---:|---:|
| Full phantom U | 3,960 | 0 | 5,643 |
| One initial anchor | 444 | 99 | 2,127 |
| At most four anchors | 564 | 25 | 2,247 |
| At most eight anchors | 724 | 13 | 2,407 |

These illustrative subtotals exclude new index overhead and temporary memory.
They predict a 5.5-fold reduction of the coordinate record with eight anchors,
and about a 2.3-fold reduction of these combined payloads before additional
index costs. They do not predict a 2.3-fold reduction of peak RSS or wall time.
Float64 U/scalars require a separate calculation.

With fixed H, coordinate-record storage is `O(N(P + H D))`; when P scales
with D this is linear in D. Decode arithmetic then grows as `O(P D/H)`.
Keeping a fixed spacing q instead would bound replay length but retain
`O(N P D/q)` coordinate storage, which is still quadratic in D when P
scales with D. H=1, 4, and 8 are profiling candidates, not settled defaults.

A compatibility encoder can initially compress one completed shell of
`[S,P,D]` coordinates. That does not remove the transient shell history or
the continuation sampler's `[S,T,D]` direction buffer. A later encoder may
record steps and anchors at acceptance, once trajectory parity is established.
The full memory plan must account for both phases.

## Later scaling option: an exact hierarchical rank index

For every valid retained phantom, `b_i < ell_ij`. Therefore:

`1[b_i <= lambda < ell_ij] = 1[b_i <= lambda] - 1[ell_ij <= lambda]`.

For any fixed set of complete rows V, its eligible count is consequently:

`count(V, lambda) = sum_i_in_V m_i * 1[b_i <= lambda]
                   - sum_(i,j)_valid_in_V 1[ell_ij <= lambda]`,

where `m_i` is that row's number of valid phantoms. This remains true when
birth contours are not monotone in append order. It is the basis of this
proposed index, not a new allocation utility or likelihood weighting.

Store sorted birth endpoints with cumulative weights m, and sorted phantom
likelihood endpoints. Two upper-bound searches give the count for a row set.
Use the original likelihood dtype, exact comparisons, and integer subtraction.
Equality at birth includes the point; equality at its likelihood excludes it.
Invalid slots must contribute to neither endpoint count. Endpoint searches
must be bounded by the actual valid endpoint count, so padded infinities do
not become deaths at a positive-infinite query. Root rows without phantoms,
infinities, ties, and empty sets require explicit handling.
Reject invalid NaN data; do not silently repair a malformed interval.

### Append in small blocks; merge index runs geometrically

Keep an exact pending tail of fewer than B complete rows; B=256 is an initial
candidate. It is scanned directly and remains visible immediately. A full
block becomes an immutable indexed run. Adjacent equal-sized runs merge, as
in binary carrying, so there are at most `1 + log2(N/B)` completed runs.
Each run covers a contiguous range of canonical row identities.

The current classic implementation already contains a wavelet matrix in
`_build_seed_rank_index`, `_seed_count_below_rank`, and
`_seed_rank_quantile` in `algorithm/depth.py`. Reuse its rank/traversal
ideas and reference cases where contracts match. It orders classics by
likelihood rank and stores dense int32 prefix counts at every element/level;
neither choice can simply be copied for full-A phantom rank selection.
Phantoms must keep append/transition order, and a dense prefix table over NP
phantoms would add O(NP log N) integer words. Packed rank metadata is the
memory-sensitive part of the proposed extension.

For a query, count each run and the tail, in append order. Their cumulative
counts locate the run containing a requested phantom rank. Within that run,
descend a binary partition of its row range: compute the eligible count of the
left child, then choose left or right and subtract that count as appropriate.
At the selected row, scan its original-order P likelihoods to select the
phantom slot. This recovers precisely the existing row-then-transition rank.

The same count operation answers the phantom-existence check currently used
by `_effective_parent_contour`. Its zero/nonzero answer must preserve the
existing fallback contour rule, including when only classic predecessors
are considered for fallback.

This index never sees the random rotation or changes a proposal rank. It
simply maps the existing rank to the same identity. Reservations operate on
that stable identity after lookup. Decode U only after collision resolution.

### Compact representation of the row partition

Storing a separate sorted float64 endpoint array at every tree node would
duplicate each phantom likelihood at every level. That undermines the memory
goal. The proposed scalable representation instead stores one sorted endpoint
array per run and compact row-routing information.

For the likelihood endpoints, retain the sequence of row labels in endpoint
order as a binary wavelet tree. Each level stores bits routing labels to the
left or right row range, with prefix-count directories. The prefix ending at
`upper_bound(ell, lambda)` can then be routed down the row tree; its count
in a child is that child's number of passed likelihood endpoints.

For births, route the birth-sorted row labels in the same way. Store cumulative
weights m in each node's birth-ordered subsequence. Routing the qualifying
birth prefix yields the born phantom weight in each child. Subtract the
child's passed-likelihood count to obtain its exact eligible count. Both
endpoint prefixes descend alongside the requested rank.

Wavelet trees provide the underlying compact rank operations; the weighted
birth-minus-likelihood traversal here is a proposed application that still
needs an independent reference implementation and validation. See
[Gagie, Navarro and Puglisi, New Algorithms on Wavelet Trees](https://users.dcc.uchile.cl/~gnavarro/ps/tcs11.2.pdf).

Under constant-time bit-vector rank with bounded local popcounts, for L runs:

| Operation / storage | Intended bound |
|---|---|
| Count all completed runs at one contour | `O(L log(NP))` |
| Pending-tail count | `O(BP)` with direct scanning |
| Select a row within the chosen run | `O(log N)` after endpoint searches |
| Select original phantom slot | `O(P)` |
| Sorted likelihood endpoints | `O(NP)` likelihood scalars, replacing the current sorted seed cache |
| Likelihood routing | `O(NP log N)` bits plus rank directories |
| Birth routing and weights | `O(N log N)` count words, plus routing bits and node metadata |

These bounds exclude the existing collision-retry count, coordinate decoding,
and index construction. Collision retries can still be expensive near
population exhaustion. With a straightforward sort-and-build implementation,
construction across geometric merges has a conservative total bound of
`O(NP log(NP) log N)`. Peak merge workspace is `O(NP)` for a large merge,
not constant. Account for old and new indexes coexisting during that merge.

This exchanges the per-shell full `[S,C]` table for small run-count and
traversal buffers, plus a persistent index. Its memory growth in N is
near-linear with logarithmic routing/weight overhead; it is not fixed memory.
The weighted birth metadata and rank directories must be included in actual
byte estimates. A naive pointer-heavy or unpacked implementation could lose
the expected savings.

The recommended first design above supplies the independent fixed-block
baseline. Compare it against this hierarchical extension at realistic large
states before choosing the additional representation and merge machinery.

## JAX, state, and integration boundaries

Use three narrow responsibilities: immutable phantom evidence records,
seed-coordinate records/decoding, and eligibility count/rank lookup. They
share stable row/slot identities. Keep scheduling and allocation outside
these components. Preserve a simple NumPy reference for the nontrivial
index and replay logic.

The existing `seed_stride` infers whether A is enabled from whether
`phantom_samples.U_samples` is present. A compact record must not accidentally
turn A off by setting that field to None. Seed participation and logical slot
count need an explicit representation-independent contract; full-coordinate
and replay decoders implement the same coordinate lookup. Audit all other
callers that currently use coordinate-array presence as a policy flag.

The index should use arrays and packed rank data, not Python objects per
phantom. Capacity buckets, a fixed level directory, and active-length masks
should keep compiled shapes stable between intentional capacity growths.
Endpoint values and weights are dynamic inputs, never large captured
constants. Do not introduce host callbacks or per-seed disk reads.

Inspect lowering before accepting the design: vmapped conditionals can
materialize unused branches, merges can retain both buffer generations, and
generic pytree sorts can copy every seed coordinate even for a classic-only
view. Logical run merges must be decoupled from classic publication and must
not cause retracing once per accepted shell.

Persist the codec version, key implementation/configuration, anchor rule,
logical identities, validity, reservation state, and capacity information
needed for the reference resume behavior. Rebuilt indexes must give identical
ranks regardless of run partitioning. A physical storage change must not
silently compact/relabel rows or change the sampler's random split shapes.
Distributed workers receive decoded seed coordinates through the existing
request boundary; they need not replicate the complete phantom store.

Old checkpoints do not necessarily contain accepted step scalars or chain
keys. Do not convert them by rerunning likelihoods or infer missing records
from rounded coordinates. They can remain in the full-coordinate format,
with an explicit decoder tag for old rows if a future mixed-format resume is
needed. The cancelled ladder's artifacts remain in their original format.

Counts and identity encodings must cover `C*(P+1)` without overflow. Use the
existing sufficient integer width, or version an explicitly wider identity
representation. Do not narrow counts or coordinates to meet a memory target.
Changing integer width can also alter compiled behavior and requires parity
checks; no current overflow incident is asserted here.

## Alternatives and scope

A bounded global reservoir, fixed phantom subset, or one point per chain
would constitute a new sampling hypothesis. Uniformly retaining points before
an eligibility query does not preserve full A's realized candidate population,
its same-contour distinctness, or its small-mode survival. The eligible fraction
can become small at a deep contour, leaving few or no retained representatives.
A fixed representative is also especially questionable when a slice chain can
tunnel between modes: one chain need not represent one component.

Such policies may ultimately be worthwhile because A's gain is modest, but
their scientific benefit needs a fresh ablation. They are not the default in
this design. No clustering, geometric mode claims, scheduling change, or
posterior participation by phantoms is proposed.

Generic exact full-A storage cannot have a fixed population-sized memory bound
while retaining arbitrary historical points for arbitrary future contours.
This proposal exploits the isotropic trajectory's reconstructible structure.
It still retains O(NP) observations and does not solve memory use in the
separate evidence-analysis reducer.

## Validation and decision gates for later implementation

The next implementation experiment, when requested, should compare the
existing selector, lazy lookup alone, block indexing alone, and both together
on the same states and keys. This separates their contributions before
introducing a codec or hierarchical index. No new benchmark jobs or sampler
implementation are part of this design revision.

1. **Independent correctness.** Compare index counts and every selectable
   rank against literal enumeration of valid `b <= lambda < ell` intervals.
   Cover nonmonotone births, tied/adjacent contours, roots, partially valid
   rows, empty populations, tail visibility, block merges, and capacity
   growth. Compare every decoded coordinate and stored likelihood with the
   original trajectory, including repeated selection and resumes.
2. **Full selector parity.** Reuse the same random inputs and compare all
   selected identities, including mixed classic/phantom proposals, repeated
   contours, heterogeneous contours, exhaustion, in-flight reservations,
   unfinished start groups, source publication, and fallback. Then verify
   complete short-run classic trees, likelihood counts, goal progress,
   stopping, and evidence inputs. Tolerance-level coordinate agreement
   does not pass this gate.
3. **Separate performance measurements.** Start with the four selector
   variants above. If memory still dominates, compare full U with replay;
   if block queries dominate, compare fixed blocks with the hierarchical index.
   Use identical saved snapshots or newly recorded chains as appropriate,
   a pinned core, synchronized
   warmed calls, and separate compilation from execution. Measure decoded
   seeds, collision attempts, index build/query time, decode time, classic
   publication, checkpoint writing, peak RSS, compiler temporaries, buffer
   copies, and recompilations. Include multiple N, P, D, S and anchor counts.
   Include cheap likelihoods where decode/index overhead is most exposed.
4. **Whole-run confirmation.** With allocation, d0, targets, and seed streams
   fixed, exact storage should preserve full A's results and likelihood work.
   Check this on completed paper cohorts and short fixed-work continuations
   from preserved SS10 states. Do not require another multiweek ladder to
   establish an exact implementation equivalence. Scientific summaries include classic posterior
   mode mass, evidence bias/RMSE, and uncertainty calibration. If a proposed
   optimization changes trajectories or retains fewer identities, it needs
   a separately named 30-seed ablation against R and full A.

Accept an implementation only when measured end-to-end cost or feasible
concurrency improves after index storage, replay, merges, checkpoints, and
analysis are accounted for. Refit worker-memory forecasts from those actual
runs. Array compression ratios alone are insufficient grounds for raising
an experiment pool's active worker count. Forecast complete-cohort wall time
from bounded measurements before considering a new long run. The user's
cancellation is not permission to restart the ladder after optimization.
